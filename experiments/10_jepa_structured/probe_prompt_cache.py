"""Probe: does prompt caching work on Bedrock for our use case?

Two back-to-back calls with the SAME system prompt (marked cache_control:
ephemeral) and DIFFERENT user prompts. If caching works:
  call 1: cache_creation_input_tokens > 0, cache_read_input_tokens = 0
  call 2: cache_creation_input_tokens = 0, cache_read_input_tokens > 0

Run:  AWS_PROFILE=cc .venv/bin/python experiments/10_jepa_structured/probe_prompt_cache.py
"""
import json
import time

import boto3

REGION = "us-west-2"
# Sonnet 4.6 — Haiku 4.5 silently drops cache_control on this Bedrock account.
# See README "Caching" section for the empirical comparison.
MODEL = "us.anthropic.claude-sonnet-4-6"

# A realistic textbook-generator system prompt — also serves as a draft of
# what we'd actually use for the P2 corpus. Padded with worked examples to
# clear the ~2k-token minimum cacheable prefix on Haiku.
SYSTEM_PROMPT = """You are a synthetic data generator for a small language-model training corpus. Each call you make produces ONE training example.

Your output must be a single JSON object with exactly these fields, in this order:
- "concept": one sentence naming the single concept the example teaches
- "question": a concrete, self-contained question that tests the concept
- "solution": a step-by-step solution (3-7 numbered steps)
- "paraphrase": the same solution rewritten in different words but identical meaning

Constraints:
- Pick one concept per example. Do not bundle multiple concepts.
- Keep questions concrete: numbers, named entities, specific cases. Avoid abstractions.
- Solutions must be derivable from first principles — show the reasoning, do not appeal to outside knowledge.
- The paraphrase must change vocabulary and sentence structure but preserve every step and every numeric/logical fact.
- Output STRICT JSON only. No prose preamble, no markdown fences, no trailing commentary.

Domains to draw from, in rough order of preference:
1. Elementary arithmetic and modular arithmetic.
2. Propositional logic (modus ponens, contrapositive, disjunctive syllogism).
3. Variable tracing in tiny imperative programs.
4. Set operations and small combinatorics.
5. Sequence patterns (arithmetic, geometric, Fibonacci-like).
6. Word problems that compile to one of the above.

Quality bar:
- The student model is 1M parameters and byte-level. Examples should be SHORT — questions under 200 chars, solutions under 600 chars.
- Each step must follow from the previous one without leaps.
- Every numeric claim must be checkable. If you write "5 + 3 = 8" you are committing to 8 being correct.
- Paraphrase invariance is the entire point — if the paraphrase contradicts the solution in any detail, the example is worthless.

Example 1:
{
  "concept": "Modular arithmetic with addition: (a + b) mod n",
  "question": "Compute (17 + 25) mod 7.",
  "solution": "1. Add: 17 + 25 = 42. 2. Divide by 7: 42 / 7 = 6 remainder 0. 3. Therefore (17 + 25) mod 7 = 0.",
  "paraphrase": "Step one, sum 17 and 25 to get 42. Step two, 42 is exactly six groups of seven, so the remainder when divided by 7 is zero. Step three, the answer is 0."
}

Example 2:
{
  "concept": "Modus ponens",
  "question": "Given: If it rains, the street is wet. It is raining. What can we conclude?",
  "solution": "1. Premise 1: Rain implies wet street. 2. Premise 2: It is raining. 3. By modus ponens, applying premise 1 to premise 2, the street is wet.",
  "paraphrase": "First, we know rainfall makes the street wet. Second, we observe that rain is occurring. Third, combining these two facts via the rule of modus ponens, we conclude the street must be wet."
}

Example 3:
{
  "concept": "Variable tracing through assignment",
  "question": "Trace this program: x = 4; y = x + 2; z = y * x; what is z?",
  "solution": "1. x is assigned 4. 2. y is assigned x + 2 = 4 + 2 = 6. 3. z is assigned y * x = 6 * 4 = 24. 4. z = 24.",
  "paraphrase": "Begin by setting x to 4. Next, compute y as the sum of x and 2, giving y the value 6. Finally, set z to the product of y and x, which is 6 times 4, equal to 24. Therefore z holds 24."
}

Example 4:
{
  "concept": "Disjunctive syllogism",
  "question": "Given: Either Maria is at home or she is at the office. Maria is not at home. Where is she?",
  "solution": "1. Premise 1: Maria is at home OR Maria is at the office. 2. Premise 2: Maria is not at home. 3. By disjunctive syllogism, eliminating the false disjunct, Maria is at the office.",
  "paraphrase": "We know one of two things: Maria is at home, or Maria is at the office. We also know Maria is not at home. The first option is ruled out, so the remaining option must hold. Maria is at the office."
}

Example 5:
{
  "concept": "Arithmetic sequence next-term",
  "question": "What is the next term in the sequence 3, 7, 11, 15, ?",
  "solution": "1. Compute differences: 7-3=4, 11-7=4, 15-11=4. 2. The common difference is 4. 3. The next term is 15 + 4 = 19.",
  "paraphrase": "Look at consecutive gaps in the sequence. From 3 to 7 is a step of 4, from 7 to 11 is also 4, and from 11 to 15 is again 4. The pattern is a constant step of 4, so the term after 15 is 15 plus 4, namely 19."
}

Example 6:
{
  "concept": "Set intersection of small finite sets",
  "question": "Let A = {2, 4, 6, 8} and B = {3, 4, 5, 6, 7}. What is A ∩ B?",
  "solution": "1. List elements of A: 2, 4, 6, 8. 2. List elements of B: 3, 4, 5, 6, 7. 3. Find elements in both: 4 is in A and B; 6 is in A and B; 2 is only in A; 8 is only in A; 3, 5, 7 are only in B. 4. Therefore A ∩ B = {4, 6}.",
  "paraphrase": "First, write down what is in A: the numbers 2, 4, 6, and 8. Second, write down what is in B: the numbers 3, 4, 5, 6, and 7. Third, identify which numbers appear in both lists. The number 4 belongs to both, and the number 6 belongs to both. No other numbers are shared. The intersection is therefore the set containing 4 and 6."
}

Example 7:
{
  "concept": "Contrapositive",
  "question": "Given: If a number is divisible by 6, then it is divisible by 3. The number 17 is not divisible by 3. What can we conclude about 17 being divisible by 6?",
  "solution": "1. The original statement: divisible by 6 implies divisible by 3. 2. The contrapositive: not divisible by 3 implies not divisible by 6. 3. We are told 17 is not divisible by 3. 4. By the contrapositive, 17 is not divisible by 6.",
  "paraphrase": "Start with the rule that any multiple of 6 must also be a multiple of 3. The contrapositive flips this: if something is not a multiple of 3, then it cannot be a multiple of 6 either. We are given that 17 fails to be a multiple of 3. Applying the contrapositive directly, 17 cannot be a multiple of 6."
}

Example 8:
{
  "concept": "Loop counter variable trace",
  "question": "Trace this program: x = 0; repeat 4 times { x = x + 3 }; what is x?",
  "solution": "1. Start: x = 0. 2. Iteration 1: x = 0 + 3 = 3. 3. Iteration 2: x = 3 + 3 = 6. 4. Iteration 3: x = 6 + 3 = 9. 5. Iteration 4: x = 9 + 3 = 12. 6. Final value: x = 12.",
  "paraphrase": "Begin with x equal to zero. The loop body adds three to x and runs four times. After the first run x becomes 3. After the second run x becomes 6. After the third run x becomes 9. After the fourth run x becomes 12. The loop terminates and x is 12."
}

Example 9:
{
  "concept": "Geometric sequence next-term",
  "question": "What is the next term in the sequence 2, 6, 18, 54, ?",
  "solution": "1. Compute ratios: 6/2 = 3, 18/6 = 3, 54/18 = 3. 2. The common ratio is 3. 3. The next term is 54 * 3 = 162.",
  "paraphrase": "Examine consecutive quotients in the sequence. Dividing 6 by 2 gives 3; dividing 18 by 6 gives 3; dividing 54 by 18 gives 3. The pattern is multiplication by 3 at each step. The term following 54 is therefore 54 multiplied by 3, which is 162."
}

Example 10:
{
  "concept": "Word problem reducing to arithmetic",
  "question": "María has 5 apples. She gives 2 to Juan and then buys 4 more. How many apples does María have now?",
  "solution": "1. Start: María has 5 apples. 2. After giving 2 to Juan: 5 - 2 = 3 apples. 3. After buying 4 more: 3 + 4 = 7 apples. 4. Final: María has 7 apples.",
  "paraphrase": "Initially María holds five apples. The first event reduces her apple count by two, leaving three. The second event increases her apple count by four. Adding four to three gives seven. María ends with seven apples."
}

When asked for an example on a specific topic, follow the same format and quality bar. When asked for a free-choice example, vary the domain across calls. Always output strict JSON, nothing else. Do not wrap the JSON in markdown code fences. Do not include any text before or after the JSON object."""


def call(client, user_prompt: str) -> dict:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 600,
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_prompt}],
    }
    t0 = time.time()
    resp = client.invoke_model(modelId=MODEL, body=json.dumps(body))
    dt = (time.time() - t0) * 1000
    payload = json.loads(resp["body"].read())
    return {"ms": dt, "payload": payload}


def report(label: str, r: dict) -> None:
    u = r["payload"]["usage"]
    print(f"\n=== {label} ===")
    print(f"  latency:                       {r['ms']:.0f} ms")
    print(f"  input_tokens (uncached):       {u['input_tokens']}")
    print(f"  cache_creation_input_tokens:   {u['cache_creation_input_tokens']}")
    print(f"  cache_read_input_tokens:       {u['cache_read_input_tokens']}")
    print(f"  output_tokens:                 {u['output_tokens']}")
    text = r["payload"]["content"][0]["text"]
    snippet = text[:120].replace("\n", " ")
    print(f"  output preview:                {snippet}...")


def main():
    client = boto3.client("bedrock-runtime", region_name=REGION)

    print("calling #1 (expect cache MISS — first time the system prompt is sent)")
    r1 = call(client, "Generate one example. Topic: free choice.")
    report("call #1", r1)

    print("\ncalling #2 (expect cache HIT — same system prompt, different user)")
    r2 = call(client, "Generate one example. Topic: parity of integers.")
    report("call #2", r2)

    print("\ncalling #3 (expect cache HIT — same system prompt, different user)")
    r3 = call(client, "Generate one example. Topic: counting steps in Tower of Hanoi for n=3.")
    report("call #3", r3)

    u1, u2, u3 = (r["payload"]["usage"] for r in (r1, r2, r3))
    cache_works = (
        u1["cache_creation_input_tokens"] > 0
        and u2["cache_read_input_tokens"] > 0
        and u3["cache_read_input_tokens"] > 0
    )

    print("\n=== summary ===")
    print(f"  cache works:                   {cache_works}")
    if cache_works:
        per_call_input = u2["cache_read_input_tokens"] + u2["input_tokens"]
        cold_input = u1["cache_creation_input_tokens"] + u1["input_tokens"]
        print(f"  cold-call input tokens:        {cold_input}")
        print(f"  warm-call input tokens (cached prefix billed at 0.1x):")
        print(f"    raw read tokens:             {u2['cache_read_input_tokens']}")
        print(f"    new tokens this call:        {u2['input_tokens']}")
        print(f"    effective billable input:    "
              f"{u2['input_tokens'] + 0.1 * u2['cache_read_input_tokens']:.0f}")
        savings_ratio = cold_input / max(
            1, u2["input_tokens"] + 0.1 * u2["cache_read_input_tokens"]
        )
        print(f"  warm-vs-cold input savings:    {savings_ratio:.1f}x")


if __name__ == "__main__":
    main()
