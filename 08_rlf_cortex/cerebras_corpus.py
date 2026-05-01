"""Generate bilingual corpus via Cerebras Cloud API (Qwen3-235B teacher).

Companion to make_teacher_thoughts.py. The local Qwen-2.5-1.5B path is
tied to the JEPA workflow because it needs hidden states. Cerebras only
returns text — but it returns *much better* text from a *much bigger*
teacher (Qwen3-235B / 22B-active) at ~1200 tok/s vs ~12 KB/s locally.

Two output formats (`--format`):

  records (default): one JSON object per Cerebras response. Each record
      preserves the prompt and the *full multi-pair response* as the
      teacher returned it. This matches the JEPT record shape that
      make_teacher_thoughts.py writes, so a later `add_thoughts.py` can
      ingest these records, run them through local Qwen-2.5-1.5B forward,
      and emit the binary JEPT format with hiddens at byte-stride
      positions.

      {"question": "Write 8 parallel ... about travel.",
       "response": "Excuse me, could you tell me how to ...\\n
                    Disculpe, ¿podría decirme ...\\n..."}

  pairs: one bilingual pair per line, parsed out of every response. Use
      this for a flat bilingual corpus (training the byte-level LM with
      no JEPA, or as a drop-in for data/bilingual.txt).

      {"en": "Excuse me, could you tell me how to get to the train station?",
       "es": "Disculpe, ¿podría decirme cómo llegar a la estación de tren?"}

Storage target: durable disk (e.g. M4 Mac Mini). Vast.ai volumes are
ephemeral; we generate on Cerebras (cheap, fast) and persist on a box
that won't get recycled.

Run:
    export CEREBRAS_API_KEY=...
    python jepa/cerebras_corpus.py \\
        --target-mb 240 \\
        --out /Volumes/mac-mini/.../data/cerebras_corpus.jsonl \\
        --format records \\
        --concurrency 8

Args:
    --model            Cerebras model id (default: qwen-3-235b-a22b-instruct-2507)
    --target-mb        stop once output reaches this size in MB
    --out              output JSONL path (will be appended)
    --resume           skip if --out exists; pick up where it left off
    --concurrency      parallel in-flight requests (default 4)
    --pairs-per-prompt how many bilingual pairs per generation (default 12)
    --counting-fraction fraction of prompts that are natural counting (default 0.0)
    --temperature      sampling temperature (default 0.8)
    --seed             RNG seed for prompt selection
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib import request as urlrequest, error as urlerror

# Same prompt distribution as make_teacher_thoughts.py so the corpora
# are interchangeable. If we change one, the other must update too.

SYSTEM_PROMPT = (
    "You write high-quality parallel English-Spanish text. "
    "Output only lines of the format: <English sentence> :: <Spanish sentence>\\n. "
    "No headers, no numbering, no explanations. "
    "Use natural register that matches the topic — colloquial when colloquial, "
    "formal when formal. Cover a wide variety of vocabulary."
)

TOPIC_TEMPLATES = [
    "Write {n} parallel English-Spanish dialogue lines about {topic}.",
    "Write {n} parallel English-Spanish narrative sentences set in {topic}.",
    "Write {n} parallel English-Spanish question-and-answer pairs about {topic}.",
    "Write {n} parallel English-Spanish casual conversation lines about {topic}.",
    "Write {n} parallel English-Spanish instructional sentences about {topic}.",
]

TOPICS = [
    "cooking and recipes", "travel and directions",
    "science and discovery", "everyday family life",
    "school and learning", "work and the office",
    "weather and seasons", "music and art",
    "children's stories", "history and old times",
    "computers and technology", "sports and games",
    "health and feelings", "food at a restaurant",
    "shopping and money", "books and reading",
    "animals and nature", "trains and cars",
    "love and friendship", "fear and bravery",
    "morning routines", "rainy days",
    "moving to a new city", "an old friend visiting",
    "learning a new skill", "fixing something broken",
    "a long phone call", "an unexpected guest",
    "working from home", "a small misunderstanding",
    "celebrating a birthday", "missing a flight",
    "a quiet evening", "a noisy market",
    "dreaming and remembering", "asking for help",
    "asking for directions", "reading the newspaper",
    "watching the stars", "the first day of school",
]

COUNTING_TEMPLATES_EN = [
    "Count from 1 to {n}, one number per line, then stop.",
    "List the integers 1 through {n}, one per line.",
    "Number from 1 up to {n}, line by line, no extras.",
]
COUNTING_TEMPLATES_ES = [
    "Cuenta de 1 a {n}, un número por línea, luego detente.",
    "Enumera los enteros del 1 al {n}, uno por línea.",
    "Numera del 1 hasta {n}, línea por línea, sin extras.",
]


# ---------------------------------------------------------------------------
# Cerebras client (urllib only; no external deps)
# ---------------------------------------------------------------------------
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"


class CerebrasError(Exception):
    pass


def cerebras_chat(api_key: str, model: str, messages: list,
                  max_tokens: int = 1024, temperature: float = 0.8,
                  timeout: float = 60.0) -> dict:
    """One blocking POST. Returns the parsed JSON response.

    Raises CerebrasError on non-200 (the caller decides retry policy).
    """
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    req = urlrequest.Request(
        CEREBRAS_URL, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Cloudflare 1010 blocks the default Python-urllib UA. Send a
            # real-looking UA — the API itself doesn't care, only Cloudflare's
            # WAF does. Tested with curl and verified python with this UA
            # passes through.
            "User-Agent": "mamba3-hands-on/cerebras_corpus.py (urllib)",
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urlerror.HTTPError as e:
        # 429 rate limit, 5xx server, 401 bad key — distinguish at retry layer
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise CerebrasError(f"HTTP {e.code}: {body}") from e
    except (urlerror.URLError, TimeoutError) as e:
        raise CerebrasError(f"network: {e}") from e


# ---------------------------------------------------------------------------
# Output filtering
# ---------------------------------------------------------------------------
PAIR_RE = re.compile(r"^(.+?)\s+::\s+(.+?)\s*$")
MIN_LEN, MAX_LEN = 5, 250


def parse_pairs(text: str) -> list[tuple[str, str]]:
    """Extract well-formed `en :: es` pairs from teacher output."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # strip leading numbering / bullets
        line = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-*•]\s*)", "", line)
        m = PAIR_RE.match(line)
        if not m:
            continue
        en, es = m.group(1).strip(), m.group(2).strip()
        if not (MIN_LEN <= len(en) <= MAX_LEN and MIN_LEN <= len(es) <= MAX_LEN):
            continue
        out.append((en, es))
    return out


# ---------------------------------------------------------------------------
# Single generation request — runs in a worker thread
# ---------------------------------------------------------------------------
def one_generation(api_key: str, model: str, user_prompt: str,
                   max_tokens: int, temperature: float,
                   max_retries: int = 5) -> tuple[str, str, dict]:
    """Returns (user_prompt, raw_text, usage_info). Retries on 429/5xx."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = cerebras_chat(api_key, model, messages,
                                 max_tokens=max_tokens,
                                 temperature=temperature)
            text = resp["choices"][0]["message"]["content"]
            usage = resp.get("usage", {})
            return user_prompt, text, usage
        except CerebrasError as e:
            last_err = e
            msg = str(e)
            # Non-retriable: 401 (bad key), 400 (bad request)
            if "401" in msg or "400" in msg:
                raise
            # Retriable: 429, 5xx, network. Exponential backoff with jitter.
            sleep = (2 ** attempt) + random.random()
            time.sleep(min(sleep, 30.0))
    raise CerebrasError(f"max retries exceeded: {last_err}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-3-235b-a22b-instruct-2507")
    ap.add_argument("--target-mb", type=float, default=240.0)
    ap.add_argument("--out", default="data/cerebras_bilingual.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--pairs-per-prompt", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--counting-fraction", type=float, default=0.0,
                    help="0 by default — the late-training mode collapse "
                         "in run #1 traced back to the unary attractor; "
                         "this corpus deliberately omits it.")
    ap.add_argument("--format", choices=["records", "pairs"], default="records",
                    help="records: one prompt+full-response per line "
                         "(preserves context for later JEPA hidden capture). "
                         "pairs: one bilingual pair per line (flat corpus).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("error: CEREBRAS_API_KEY not set in environment", file=sys.stderr)
        print("  → set it with: export CEREBRAS_API_KEY=...", file=sys.stderr)
        print("  → or: set -a; source .env; set +a", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bytes_written = 0
    n_records = 0
    if args.resume and out_path.exists():
        bytes_written = out_path.stat().st_size
        # Count lines for record count.
        with open(out_path, "rb") as f:
            n_records = sum(1 for _ in f)
        print(f"resume: existing file {bytes_written/2**20:.1f} MB, "
              f"{n_records} records", flush=True)
    elif out_path.exists():
        print(f"error: {out_path} exists (use --resume to append)",
              file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    target_bytes = int(args.target_mb * 2**20)
    n_attempts = 0
    n_total_tokens = 0
    t0 = time.time()

    def make_prompt() -> str:
        """One prompt for one generation request."""
        if rng.random() < args.counting_fraction:
            n = rng.randint(3, 30)
            tmpl = (rng.choice(COUNTING_TEMPLATES_EN)
                    if rng.random() < 0.5
                    else rng.choice(COUNTING_TEMPLATES_ES))
            return tmpl.format(n=n)
        return rng.choice(TOPIC_TEMPLATES).format(
            n=args.pairs_per_prompt, topic=rng.choice(TOPICS),
        )

    # Concurrency via thread pool: urlopen is blocking but releases GIL on I/O,
    # so threads scale fine for network-bound work. ~4-8 concurrent requests
    # is the right band — Cerebras throttles harder past that on free tier.
    out_f = open(out_path, "ab")
    try:
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            in_flight = []
            while bytes_written < target_bytes:
                # Top up the pool.
                while len(in_flight) < args.concurrency and bytes_written < target_bytes:
                    prompt = make_prompt()
                    n_attempts += 1
                    in_flight.append(ex.submit(
                        one_generation, api_key, args.model, prompt,
                        args.max_tokens, args.temperature,
                    ))
                if not in_flight:
                    break
                # Drain any completed futures.
                done, pending = cf.wait(in_flight, timeout=2.0,
                                        return_when=cf.FIRST_COMPLETED)
                in_flight = list(pending)
                for fut in done:
                    try:
                        prompt, text, usage = fut.result()
                    except CerebrasError as e:
                        print(f"  drop: {e}", flush=True)
                        continue
                    n_total_tokens += int(usage.get("total_tokens", 0))
                    pairs = parse_pairs(text)
                    if not pairs:
                        # No well-formed pairs in this response — skip.
                        continue
                    if args.format == "records":
                        # Reconstruct the response from valid pairs only,
                        # so downstream consumers can rely on it.
                        clean_response = "\n".join(
                            f"{en} :: {es}" for en, es in pairs
                        ) + "\n"
                        rec = json.dumps(
                            {"question": prompt, "response": clean_response},
                            ensure_ascii=False,
                        )
                        line = rec.encode("utf-8") + b"\n"
                        out_f.write(line)
                        bytes_written += len(line)
                        n_records += 1
                    else:  # pairs
                        for en, es in pairs:
                            rec = json.dumps({"en": en, "es": es},
                                             ensure_ascii=False)
                            line = rec.encode("utf-8") + b"\n"
                            out_f.write(line)
                            bytes_written += len(line)
                            n_records += 1
                # Periodic flush + log.
                if n_attempts % 5 == 0:
                    out_f.flush()
                    pct = 100 * bytes_written / target_bytes
                    elapsed = time.time() - t0
                    rate = (bytes_written / max(elapsed, 1e-6)) / 1024
                    print(f"  attempts={n_attempts:5d}  records={n_records:6d}  "
                          f"tokens={n_total_tokens:>9,d}  "
                          f"size={bytes_written/2**20:6.2f}/{args.target_mb} MB  "
                          f"({pct:5.1f}%)  rate={rate:.1f} KB/s",
                          flush=True)
    finally:
        out_f.close()

    elapsed = time.time() - t0
    print(f"\ndone: {out_path} ({bytes_written/2**20:.2f} MB, "
          f"{n_records} records, {n_total_tokens:,} tokens, "
          f"{n_attempts} prompts, {elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
