"""
Seed amplifier — takes 10 seed examples per lesson and uses Cerebras
(Qwen-3 235B) to generate ~100 variations of each, producing ~1000
examples per lesson.

Usage:
    python amplify.py seeds/phase1_1_hanoi.jsonl --count 100
    python amplify.py seeds/ --count 100          # all seed files

Output goes to data/amplified/<lesson_name>.jsonl
"""
import json
import os
import sys
import time
import argparse
from pathlib import Path

import urllib.request
import urllib.error

# Load env (local .env first, then ~/.gaucho-code/.env as fallback)
for env_path in [Path(__file__).parent / ".env", Path.home() / ".gaucho-code" / ".env"]:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)

API_KEY = os.environ["CEREBRAS_API_KEY"]
BASE_URL = os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
MODEL = os.environ.get("GAUCHO_MODEL", "qwen-3-235b-a22b-instruct-2507")


def cerebras_chat(system: str, user: str, max_tokens: int = 2048,
                  temperature: float = 0.8) -> str:
    """Single chat completion call to Cerebras."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "mamba3-amplify/1.0",
        },
    )

    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                wait = int(e.headers.get("Retry-After", 2 ** attempt))
                print(f"  Rate limited ({e.code}), waiting {wait}s...",
                      flush=True)
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            print(f"  Error: {e}, retrying in {2**attempt}s...", flush=True)
            time.sleep(2 ** attempt)

    raise RuntimeError("Failed after 5 retries")


SYSTEM_PROMPT = """\
You are a training data generator for a reasoning AI model. You will be given \
seed examples in JSONL format. Your job is to generate new variations that:

1. Follow the EXACT same format (JSON with "lesson", "input", "output" fields)
2. Vary the difficulty, numbers, names, and phrasing
3. Alternate between English and Spanish roughly 50/50
4. Keep the ```thinking```, natural language answer, and ```python``` structure
5. Ensure all answers and code are CORRECT — verify your math/logic
6. Output ONLY valid JSONL — one JSON object per line, no extra text

IMPORTANT:
- Do NOT repeat the seed examples
- Each line must be a complete, valid JSON object
- The "output" field uses markdown code fences (```thinking```, ```python```)
- Escape newlines as \\n within JSON strings
- Vary difficulty: some easy, some harder than the seeds
"""


def amplify_file(seed_path: Path, count: int, out_dir: Path):
    """Amplify a single seed file."""
    seeds = []
    with open(seed_path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))

    lesson = seeds[0].get("lesson", seed_path.stem)
    out_path = out_dir / f"{lesson}.jsonl"
    print(f"\n{'='*60}", flush=True)
    print(f"Amplifying: {seed_path.name} → {out_path}", flush=True)
    print(f"Seeds: {len(seeds)}, Target: {count} new examples", flush=True)

    # Format seeds for the prompt
    seed_text = "\n".join(json.dumps(s, ensure_ascii=False) for s in seeds)

    generated = []
    batch_size = 20  # request 20 at a time to stay within token limits
    batches_needed = (count + batch_size - 1) // batch_size

    for batch_i in range(batches_needed):
        remaining = count - len(generated)
        n = min(batch_size, remaining)
        if n <= 0:
            break

        user_msg = f"""Here are the seed examples:

{seed_text}

Generate exactly {n} NEW variations. Output {n} lines of JSONL, nothing else.
Vary difficulty, language (EN/ES), names, numbers. All answers must be correct."""

        print(f"  Batch {batch_i+1}/{batches_needed}: requesting {n} examples...",
              end="", flush=True)
        t0 = time.time()

        try:
            response = cerebras_chat(SYSTEM_PROMPT, user_msg,
                                     max_tokens=8192, temperature=0.8)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)
            continue

        # Parse response lines as JSONL
        good = 0
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("```"):
                continue
            try:
                obj = json.loads(line)
                if "input" in obj and "output" in obj:
                    obj.setdefault("lesson", lesson)
                    generated.append(obj)
                    good += 1
            except json.JSONDecodeError:
                continue

        elapsed = time.time() - t0
        print(f" got {good} valid in {elapsed:.1f}s", flush=True)

        # Brief pause to avoid rate limits
        time.sleep(0.5)

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for obj in generated:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"  Written: {len(generated)} examples to {out_path}", flush=True)
    return len(generated)


def main():
    parser = argparse.ArgumentParser(description="Amplify seed data via Cerebras")
    parser.add_argument("input", help="Seed JSONL file or directory of seed files")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of new examples per seed file (default: 100)")
    parser.add_argument("--out", default="data/amplified",
                        help="Output directory (default: data/amplified)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

    print(f"Cerebras model: {MODEL}", flush=True)
    print(f"Files to amplify: {len(files)}", flush=True)
    print(f"Target per file: {args.count}", flush=True)

    total = 0
    for f in files:
        total += amplify_file(f, args.count, out_dir)

    print(f"\n{'='*60}", flush=True)
    print(f"Total generated: {total} examples", flush=True)


if __name__ == "__main__":
    main()
