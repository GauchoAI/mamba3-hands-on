"""Generate the textbook-style synthetic corpus for P2.

Reuses the verified Bedrock + Converse + cachePoint + Sonnet-4.6 pipeline
from the cache probes. Streams JSONL output (one example per line) so a
crash mid-run only loses the in-flight call.

Usage:
  AWS_PROFILE=cc .venv/bin/python experiments/jepa_structured_data/gen_textbook.py \
      --n 5 \
      --out experiments/jepa_structured_data/data/textbook_smoke.jsonl

  --n            number of examples
  --out          local path for JSONL output (parent dir auto-created)
  --topics-file  optional file, one topic per line; if absent, free-choice
  --max-tokens   per-call output cap (default 600)
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import boto3

from probe_prompt_cache import SYSTEM_PROMPT

REGION = "us-west-2"
MODEL = "us.anthropic.claude-sonnet-4-6"

# Default domain rotation — used when --topics-file is not provided. Each call
# is given an explicit topic to break the deterministic mode-collapse we saw
# with "free choice" prompts and a cached system prefix.
DEFAULT_TOPICS = [
    "modular arithmetic with addition",
    "modular arithmetic with subtraction",
    "modus ponens with concrete entities",
    "contrapositive of an implication",
    "disjunctive syllogism with two options",
    "variable tracing through three sequential assignments",
    "variable tracing through a conditional branch",
    "loop counter accumulation",
    "set intersection of small finite sets",
    "set union of small finite sets",
    "arithmetic sequence next-term",
    "geometric sequence next-term",
    "Fibonacci-like sequence next-term",
    "greatest common divisor by Euclidean steps",
    "least common multiple of two small integers",
    "parity of a sum of integers",
    "parity of a product of integers",
    "word problem reducing to addition and subtraction",
    "word problem reducing to multiplication",
    "word problem about distance, speed, and time",
    "counting moves in Tower of Hanoi for small n",
    "Tower of Hanoi single-step move legality",
    "balanced parentheses depth tracking",
    "stack push/pop final state",
    "small combinatorics: ways to choose k from n",
    "comparison of two fractions by cross-multiplication",
    "monotone-run detection in a short integer sequence",
    "rounding to the nearest ten",
    "even-vs-odd classification with reasoning",
    "small power of two recognition",
]

# Bedrock Sonnet 4.6 us-west-2 pricing (USD per 1k tokens)
PRICE_IN = 3.0e-3
PRICE_OUT = 15.0e-3
PRICE_CACHE_WRITE = PRICE_IN * 1.25
PRICE_CACHE_READ = PRICE_IN * 0.10


def call(client, user_prompt: str, max_tokens: int, temperature: float = 1.0) -> dict:
    t0 = time.time()
    resp = client.converse(
        modelId=MODEL,
        system=[
            {"text": SYSTEM_PROMPT},
            {"cachePoint": {"type": "default"}},
        ],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    return {"ms": (time.time() - t0) * 1000, "resp": resp}


def parse_example(text: str) -> dict | None:
    """Sonnet sometimes wraps JSON in ```json fences despite our instructions.
    Strip fences and extract the first balanced object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    required = {"concept", "question", "solution", "paraphrase"}
    if not required.issubset(obj):
        return None
    return obj


def usage_cost(u: dict) -> float:
    return (
        u.get("inputTokens", 0) * PRICE_IN
        + u.get("cacheWriteInputTokens", 0) * PRICE_CACHE_WRITE
        + u.get("cacheReadInputTokens", 0) * PRICE_CACHE_READ
        + u.get("outputTokens", 0) * PRICE_OUT
    ) / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--topics-file", type=str, default=None)
    ap.add_argument("--max-tokens", type=int, default=600)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.topics_file:
        topics = [t.strip() for t in Path(args.topics_file).read_text().splitlines() if t.strip()]
    else:
        topics = DEFAULT_TOPICS

    client = boto3.client("bedrock-runtime", region_name=REGION)

    n_ok = 0
    n_bad = 0
    total_cost = 0.0
    t0 = time.time()

    with open(out_path, "w", buffering=1) as f:  # line-buffered → durable per line
        for i in range(args.n):
            topic = topics[i % len(topics)]
            # Add a per-call salt: the iteration index. Forces a fresh user
            # message even when the topic repeats on the second pass through
            # the rotation, defeating any latent response cache.
            user_prompt = (
                f"Generate one example (call #{i + 1}). "
                f"Topic: {topic}. Pick fresh numbers/entities not used in earlier calls."
            )

            try:
                r = call(client, user_prompt, args.max_tokens)
            except Exception as e:
                print(f"[{i:03d}] API error: {e}", file=sys.stderr)
                n_bad += 1
                continue

            u = r["resp"]["usage"]
            cost = usage_cost(u)
            total_cost += cost
            text = r["resp"]["output"]["message"]["content"][0]["text"]
            ex = parse_example(text)

            cache_status = (
                f"W{u.get('cacheWriteInputTokens', 0)}"
                if u.get("cacheWriteInputTokens", 0) > 0
                else f"R{u.get('cacheReadInputTokens', 0)}"
            )
            if ex is None:
                n_bad += 1
                f.write(json.dumps({"_error": "parse_failed", "_raw": text, "_user_prompt": user_prompt}) + "\n")
                print(f"[{i:03d}] PARSE-FAIL  {r['ms']:.0f}ms  cache={cache_status}  ${cost:.5f}", file=sys.stderr)
            else:
                n_ok += 1
                ex["_meta"] = {"model": MODEL, "user_prompt": user_prompt, "cost_usd": round(cost, 6)}
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                print(
                    f"[{i:03d}] OK          {r['ms']:.0f}ms  cache={cache_status}  "
                    f"${cost:.5f}  concept={ex['concept'][:60]}"
                )

    dt = time.time() - t0
    print(file=sys.stderr)
    print(f"  wrote: {out_path}", file=sys.stderr)
    print(f"  ok / bad: {n_ok} / {n_bad}", file=sys.stderr)
    print(f"  total cost: ${total_cost:.4f}", file=sys.stderr)
    print(f"  elapsed: {dt:.1f}s ({dt / max(1, args.n):.1f}s/call avg)", file=sys.stderr)
    if n_ok > 0:
        print(f"  avg cost / good example: ${total_cost / n_ok:.5f}", file=sys.stderr)


if __name__ == "__main__":
    main()
