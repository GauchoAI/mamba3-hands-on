"""Extended cache probe: 1 cold + 10 warm calls in a row.

Verifies that the cache TTL holds across a realistic burst, not just two
calls back-to-back. Reports per-call usage and the steady-state savings.

Run: AWS_PROFILE=cc .venv/bin/python experiments/jepa_structured_data/probe_prompt_cache_burst.py
"""
import time

import boto3

from probe_prompt_cache import SYSTEM_PROMPT

REGION = "us-west-2"
MODEL = "us.anthropic.claude-sonnet-4-6"
N_WARM = 10

# Vary the user prompt across calls so we don't accidentally hit response cache.
USER_PROMPTS = [
    "Generate one example. Topic: free choice.",
    "Generate one example. Topic: parity of integers.",
    "Generate one example. Topic: Tower of Hanoi for n=3.",
    "Generate one example. Topic: modus ponens with concrete entities.",
    "Generate one example. Topic: greatest common divisor.",
    "Generate one example. Topic: arithmetic sequence.",
    "Generate one example. Topic: set union.",
    "Generate one example. Topic: variable trace with conditional.",
    "Generate one example. Topic: word problem about distance and speed.",
    "Generate one example. Topic: contrapositive of an implication.",
    "Generate one example. Topic: geometric sequence.",
]
assert len(USER_PROMPTS) >= 1 + N_WARM


def call(client, user_prompt: str) -> dict:
    t0 = time.time()
    resp = client.converse(
        modelId=MODEL,
        system=[
            {"text": SYSTEM_PROMPT},
            {"cachePoint": {"type": "default"}},
        ],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={"maxTokens": 400},
    )
    return {"ms": (time.time() - t0) * 1000, "u": resp["usage"]}


def main():
    client = boto3.client("bedrock-runtime", region_name=REGION)
    rows = []
    for i, prompt in enumerate(USER_PROMPTS[: 1 + N_WARM]):
        r = call(client, prompt)
        rows.append(r)
        u = r["u"]
        kind = "COLD" if i == 0 else f"warm{i:02d}"
        print(
            f"{kind:>6}  {r['ms']:>5.0f}ms  "
            f"in={u['inputTokens']:>4}  "
            f"cacheRead={u.get('cacheReadInputTokens', 0):>5}  "
            f"cacheWrite={u.get('cacheWriteInputTokens', 0):>5}  "
            f"out={u['outputTokens']}"
        )

    cold = rows[0]["u"]
    warm = [r["u"] for r in rows[1:]]
    cw = cold.get("cacheWriteInputTokens", 0) or 0
    avg_read = sum(w.get("cacheReadInputTokens", 0) or 0 for w in warm) / len(warm)
    avg_in = sum(w["inputTokens"] for w in warm) / len(warm)
    avg_out = sum(w["outputTokens"] for w in warm) / len(warm)

    # Bedrock Sonnet 4.6 pricing (us-west-2):
    in_per_kt = 3.0e-3
    out_per_kt = 15.0e-3
    cache_write_per_kt = in_per_kt * 1.25
    cache_read_per_kt = in_per_kt * 0.10

    cold_cost = (
        cold["inputTokens"] * in_per_kt
        + cw * cache_write_per_kt
        + cold["outputTokens"] * out_per_kt
    ) / 1000
    avg_warm_cost = (
        avg_in * in_per_kt
        + avg_read * cache_read_per_kt
        + avg_out * out_per_kt
    ) / 1000

    print()
    print(f"  cold call cost:                ${cold_cost:.5f}")
    print(f"  warm avg cost (n={len(warm)}):           ${avg_warm_cost:.5f}")
    print(f"  warm/cold cost ratio:          {avg_warm_cost / cold_cost:.2f}")
    print(
        f"  cache-hit consistency:         "
        f"{sum(1 for w in warm if (w.get('cacheReadInputTokens', 0) or 0) > 0)}/{len(warm)} warm calls hit"
    )

    daily_budget_usd = 1000_000 * in_per_kt / 1000  # rough: 1M Sonnet input-tokens/day
    print(
        f"  est steady-state corpus rate:  "
        f"~{int(daily_budget_usd / avg_warm_cost):,} examples/day at ~$3/1M-input-token budget"
    )


if __name__ == "__main__":
    main()
