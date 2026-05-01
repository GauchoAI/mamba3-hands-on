"""Same probe but via Bedrock's Converse API, which is the documented path for
prompt caching on Bedrock. cache_control becomes a structural element
(`cachePoint`) rather than a field on a content block.

Run: AWS_PROFILE=cc .venv/bin/python experiments/jepa_structured_data/probe_prompt_cache_converse.py
"""
import time

import boto3

from probe_prompt_cache import SYSTEM_PROMPT  # noqa: E402  reuse the same long system prompt

REGION = "us-west-2"
# Sonnet 4.6 — Haiku 4.5 silently drops cache_control on this Bedrock account.
MODEL = "us.anthropic.claude-sonnet-4-6"


def call(client, user_prompt: str) -> dict:
    t0 = time.time()
    resp = client.converse(
        modelId=MODEL,
        system=[
            {"text": SYSTEM_PROMPT},
            {"cachePoint": {"type": "default"}},
        ],
        messages=[
            {"role": "user", "content": [{"text": user_prompt}]},
        ],
        inferenceConfig={"maxTokens": 600},
    )
    dt = (time.time() - t0) * 1000
    return {"ms": dt, "resp": resp}


def report(label: str, r: dict) -> None:
    u = r["resp"].get("usage", {})
    print(f"\n=== {label} ===")
    print(f"  latency:                       {r['ms']:.0f} ms")
    print(f"  inputTokens:                   {u.get('inputTokens')}")
    print(f"  cacheReadInputTokens:          {u.get('cacheReadInputTokens')}")
    print(f"  cacheWriteInputTokens:         {u.get('cacheWriteInputTokens')}")
    print(f"  outputTokens:                  {u.get('outputTokens')}")
    out = r["resp"]["output"]["message"]["content"][0]["text"]
    print(f"  output preview:                {out[:120].replace(chr(10), ' ')}...")


def main():
    client = boto3.client("bedrock-runtime", region_name=REGION)

    print("Converse API — call #1 (expect cache MISS / write)")
    r1 = call(client, "Generate one example. Topic: free choice.")
    report("call #1", r1)

    print("\nConverse API — call #2 (expect cache HIT / read)")
    r2 = call(client, "Generate one example. Topic: parity of integers.")
    report("call #2", r2)

    print("\nConverse API — call #3 (expect cache HIT / read)")
    r3 = call(client, "Generate one example. Topic: Tower of Hanoi for n=3.")
    report("call #3", r3)

    u1 = r1["resp"].get("usage", {})
    u2 = r2["resp"].get("usage", {})
    cache_works = (u1.get("cacheWriteInputTokens", 0) or 0) > 0 and (
        u2.get("cacheReadInputTokens", 0) or 0
    ) > 0
    print("\n=== summary ===")
    print(f"  cache works (Converse path):   {cache_works}")


if __name__ == "__main__":
    main()
