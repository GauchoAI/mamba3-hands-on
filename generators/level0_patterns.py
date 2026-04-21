"""
Level 0 — Pattern Recognition data generator.

The most fundamental cognitive operation. Everything else bootstraps from this.

Task types:
  1. Sequence completion: A B A B A B ? → A
  2. Same/different: 3 3 → SAME, 3 5 → DIFF
  3. Odd one out: 2 2 2 7 2 → 7
  4. Repeat counting: A B A B A → count(A)=3
  5. Pattern period: 1 2 3 1 2 3 1 2 3 → period=3
  6. Next in arithmetic sequence: 2 5 8 11 ? → 14
  7. Mirror detection: 1 2 3 2 1 → MIRROR, 1 2 3 4 5 → NO

Usage:
    python generators/level0_patterns.py --count 10000 --out data/level0/
    python generators/level0_patterns.py --validate  # check correctness
"""
import json
import random
import argparse
from pathlib import Path


def gen_sequence_completion(max_alpha=10, max_period=5, min_repeats=2, max_repeats=6):
    """Generate a repeating pattern and ask for the next element."""
    period = random.randint(2, max_period)
    repeats = random.randint(min_repeats, max_repeats)
    pattern = [random.randint(0, max_alpha) for _ in range(period)]

    # Show full repeats, then partial (missing last element)
    cut = random.randint(1, period)  # how many of the last repeat to show
    seq = pattern * repeats + pattern[:cut]
    answer = pattern[cut % period]

    seq_str = " ".join(str(x) for x in seq)
    return {
        "type": "sequence_completion",
        "input": f"{seq_str} ?",
        "output": str(answer),
        "meta": {"period": period, "pattern": pattern}
    }


def gen_same_different(max_val=20):
    """Two values: are they the same or different?"""
    a = random.randint(0, max_val)
    if random.random() < 0.5:
        b = a  # same
        answer = "S"
    else:
        b = a
        while b == a:
            b = random.randint(0, max_val)
        answer = "D"

    return {
        "type": "same_different",
        "input": f"{a} {b}",
        "output": answer,
    }


def gen_odd_one_out(max_val=15, min_len=4, max_len=8):
    """All elements are the same except one. Find the odd one."""
    length = random.randint(min_len, max_len)
    common = random.randint(0, max_val)
    odd = common
    while odd == common:
        odd = random.randint(0, max_val)

    pos = random.randint(0, length - 1)
    seq = [common] * length
    seq[pos] = odd

    seq_str = " ".join(str(x) for x in seq)
    return {
        "type": "odd_one_out",
        "input": seq_str,
        "output": str(odd),
        "meta": {"position": pos}
    }


def gen_repeat_count(max_alpha=8, min_len=5, max_len=12):
    """Count how many times a specific value appears."""
    length = random.randint(min_len, max_len)
    vocab = list(range(max_alpha))
    seq = [random.choice(vocab) for _ in range(length)]
    target = random.choice(seq)  # pick a value that appears at least once
    count = seq.count(target)

    seq_str = " ".join(str(x) for x in seq)
    return {
        "type": "repeat_count",
        "input": f"{seq_str} COUNT {target}",
        "output": str(count),
    }


def gen_pattern_period(max_alpha=8, max_period=5, min_repeats=2, max_repeats=4):
    """What is the period of this repeating sequence?"""
    period = random.randint(2, max_period)
    repeats = random.randint(min_repeats, max_repeats)
    pattern = [random.randint(0, max_alpha) for _ in range(period)]
    seq = pattern * repeats

    seq_str = " ".join(str(x) for x in seq)
    return {
        "type": "pattern_period",
        "input": f"{seq_str} PERIOD",
        "output": str(period),
        "meta": {"pattern": pattern}
    }


def gen_arithmetic_next(max_start=20, max_step=10, min_len=3, max_len=6):
    """Next element in an arithmetic sequence."""
    start = random.randint(0, max_start)
    step = random.randint(1, max_step)
    if random.random() < 0.3:
        step = -step  # sometimes decreasing
    length = random.randint(min_len, max_len)
    seq = [start + i * step for i in range(length)]
    answer = start + length * step

    seq_str = " ".join(str(x) for x in seq)
    return {
        "type": "arithmetic_next",
        "input": f"{seq_str} ?",
        "output": str(answer),
        "meta": {"start": start, "step": step}
    }


def gen_geometric_next(max_base=5, min_len=3, max_len=5):
    """Next element in a geometric sequence."""
    base = random.randint(1, max_base)
    ratio = random.randint(2, 4)
    length = random.randint(min_len, max_len)
    seq = [base * (ratio ** i) for i in range(length)]
    answer = base * (ratio ** length)

    # Skip if numbers get too large
    if answer > 10000:
        return gen_arithmetic_next()  # fallback

    seq_str = " ".join(str(x) for x in seq)
    return {
        "type": "geometric_next",
        "input": f"{seq_str} ?",
        "output": str(answer),
        "meta": {"base": base, "ratio": ratio}
    }


def gen_mirror_detection(max_val=10, min_len=3, max_len=7):
    """Is this sequence a palindrome/mirror?"""
    length = random.randint(min_len, max_len)
    if random.random() < 0.5:
        # Generate mirror
        half = [random.randint(0, max_val) for _ in range(length // 2)]
        if length % 2 == 1:
            middle = [random.randint(0, max_val)]
        else:
            middle = []
        seq = half + middle + half[::-1]
        answer = "M"
    else:
        # Generate non-mirror (ensure it's not accidentally a mirror)
        seq = [random.randint(0, max_val) for _ in range(length)]
        while seq == seq[::-1]:
            seq = [random.randint(0, max_val) for _ in range(length)]
        answer = "N"

    seq_str = " ".join(str(x) for x in seq)
    return {
        "type": "mirror_detection",
        "input": seq_str,
        "output": answer,
    }


def gen_parity(min_len=3, max_len=8):
    """Count 1s in binary sequence → SAME if even, DIFF if odd (parity)."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, 1) for _ in range(length)]
    ones = sum(seq)
    answer = "S" if ones % 2 == 0 else "D"
    seq_str = " ".join(str(x) for x in seq)
    return {"type": "parity", "input": seq_str, "output": answer}


def gen_binary_pattern_next(min_repeats=2, max_repeats=4):
    """Binary pattern continuation: 0 0 1 1 0 0 1 1 → 0.
    Like IQ test pattern recognition with just 0s and 1s."""
    # Generate a binary pattern of period 2-6
    period = random.randint(2, 6)
    pattern = [random.randint(0, 1) for _ in range(period)]
    repeats = random.randint(min_repeats, max_repeats)
    # Show full repeats + partial
    cut = random.randint(1, period)
    seq = pattern * repeats + pattern[:cut]
    answer = pattern[cut % period]
    seq_str = " ".join(str(x) for x in seq)
    return {
        "type": "binary_pattern_next",
        "input": f"{seq_str} ?",
        "output": str(answer),
    }


def gen_run_length_next(max_run=5, max_val=3):
    """Predict next in a run-length pattern.
    E.g., 0 1 1 2 2 2 3 3 3 3 → ? (next would be 4)
    Or:   1 1 2 2 3 3 → ? (next is 4, each appears twice)"""
    pattern_type = random.choice(["increasing_runs", "fixed_runs"])

    if pattern_type == "increasing_runs":
        # Each value n appears n+1 times: 0, 1 1, 2 2 2, ...
        n_groups = random.randint(2, max_val)
        seq = []
        for i in range(n_groups):
            seq.extend([i] * (i + 1))
        # Next element starts the next group
        answer = n_groups
    else:
        # Fixed run length: each value appears k times
        k = random.randint(2, max_run)
        n_groups = random.randint(2, max_val)
        seq = []
        for i in range(n_groups):
            seq.extend([i] * k)
        answer = n_groups

    seq_str = " ".join(str(x) for x in seq)
    return {
        "type": "run_length_next",
        "input": f"{seq_str} ?",
        "output": str(answer),
    }


def gen_alternating_next(max_val=10):
    """Alternating pattern with two interleaved sequences.
    E.g., 1 10 2 9 3 8 ? → 4 (two sequences: 1,2,3,4 and 10,9,8)"""
    # Sequence A: arithmetic
    a_start = random.randint(0, max_val // 2)
    a_step = random.randint(1, 3)
    # Sequence B: different arithmetic
    b_start = random.randint(max_val // 2, max_val)
    b_step = random.randint(-3, -1)

    length = random.randint(3, 5)  # pairs
    seq = []
    for i in range(length):
        seq.append(a_start + i * a_step)
        seq.append(b_start + i * b_step)

    # Remove last element, that's the answer
    answer = seq[-1]
    seq = seq[:-1]

    seq_str = " ".join(str(x) for x in seq)
    return {
        "type": "alternating_next",
        "input": f"{seq_str} ?",
        "output": str(answer),
    }


def gen_logic_gate(max_inputs=2):
    """Evaluate a single logic gate: AND/OR/XOR/NOT on binary inputs."""
    gate = random.choice(["AND", "OR", "XOR", "NOT"])
    if gate == "NOT":
        a = random.randint(0, 1)
        result = 1 - a
        return {"type": "logic_gate", "input": f"NOT {a}", "output": str(result)}
    else:
        a = random.randint(0, 1)
        b = random.randint(0, 1)
        if gate == "AND":
            result = a & b
        elif gate == "OR":
            result = a | b
        else:  # XOR
            result = a ^ b
        return {"type": "logic_gate", "input": f"{gate} {a} {b}", "output": str(result)}


def gen_logic_chain(max_depth=2):
    """Evaluate a chain of logic gates: AND 1 1 OR 0 → OR(AND(1,1), 0) = 1.
    Reads left to right: first gate takes first two bits, result feeds next gate."""
    gates = ["AND", "OR", "XOR"]
    depth = random.randint(1, max_depth)

    # Start with two values
    val = random.randint(0, 1)
    parts = [str(val)]

    for _ in range(depth):
        gate = random.choice(gates)
        b = random.randint(0, 1)
        if gate == "AND":
            val = val & b
        elif gate == "OR":
            val = val | b
        else:
            val = val ^ b
        parts.extend([gate, str(b)])

    return {
        "type": "logic_chain",
        "input": " ".join(parts),
        "output": str(val),
    }


def gen_modus_ponens():
    """Modus ponens: IF p THEN q. Given p, what is q?
    Also: contrapositive, and cases where p is false (inconclusive → ?)."""
    p_val = random.randint(0, 1)
    q_val = random.randint(0, 1)

    variant = random.choice(["direct", "contrapositive", "false_premise"])

    if variant == "direct":
        # IF p THEN q. p is TRUE. → q
        # We encode: IF 1 THEN {q} → {q}
        return {
            "type": "modus_ponens",
            "input": f"IF 1 THEN {q_val}",
            "output": str(q_val),
        }
    elif variant == "contrapositive":
        # IF p THEN q. NOT q. → NOT p
        # IF {p} THEN 1 NOT 1 → NOT {p}
        return {
            "type": "modus_ponens",
            "input": f"IF {p_val} THEN 1 NOT 1",
            "output": str(1 - p_val),
        }
    else:
        # IF p THEN q. p is FALSE. → inconclusive (we say ?)
        return {
            "type": "modus_ponens",
            "input": f"IF 0 THEN {q_val}",
            "output": "?",
        }


GENERATORS = [
    (gen_sequence_completion, 3),
    (gen_same_different, 2),
    (gen_odd_one_out, 2),
    (gen_repeat_count, 2),
    (gen_pattern_period, 1),
    (gen_arithmetic_next, 2),
    (gen_geometric_next, 1),
    (gen_mirror_detection, 2),
    (gen_parity, 2),
    (gen_binary_pattern_next, 2),
    (gen_run_length_next, 1),
    (gen_alternating_next, 1),
    (gen_logic_gate, 2),
    (gen_logic_chain, 1),
    (gen_modus_ponens, 1),
]


def generate_dataset(count):
    """Generate a mixed dataset with weighted task sampling."""
    fns, weights = zip(*GENERATORS)
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]

    examples = []
    for _ in range(count):
        fn = random.choices(fns, weights=probs, k=1)[0]
        examples.append(fn())
    return examples


def validate(count=1000):
    """Generate examples and verify basic correctness."""
    examples = generate_dataset(count)
    errors = 0
    for ex in examples:
        t = ex["type"]
        inp = ex["input"]
        out = ex["output"]

        if t == "same_different":
            a, b = inp.split()
            expected = "S" if a == b else "D"
            if out != expected:
                print(f"ERROR: {inp} → {out}, expected {expected}")
                errors += 1

        elif t == "mirror_detection":
            nums = inp.split()
            is_mirror = nums == nums[::-1]
            expected = "M" if is_mirror else "N"
            if out != expected:
                print(f"ERROR: {inp} → {out}, expected {expected}")
                errors += 1

        elif t == "repeat_count":
            parts = inp.split(" COUNT ")
            seq = parts[0].split()
            target = parts[1]
            expected = str(seq.count(target))
            if out != expected:
                print(f"ERROR: {inp} → {out}, expected {expected}")
                errors += 1

    print(f"Validated {count} examples: {errors} errors")
    type_counts = {}
    for ex in examples:
        t = ex["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    return errors == 0


def main():
    parser = argparse.ArgumentParser(description="Level 0 pattern recognition data")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--out", default="data/level0/")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.validate:
        ok = validate()
        return 0 if ok else 1

    examples = generate_dataset(args.count)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "patterns.jsonl"

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(examples)} Level 0 examples → {out_path}")

    # Print distribution
    type_counts = {}
    for ex in examples:
        t = ex["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
