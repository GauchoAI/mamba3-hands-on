"""Level 1 reasoning generators — harder problems building on level 0 skills.

These require composition: arithmetic + logic, pattern + counting, etc.
"""
import random


def gen_cumulative_sum(max_val=10, min_len=3, max_len=6):
    """Running cumulative sum. Input: sequence, output: final cumulative sum.
    Example: '2 3 1' → '6' (2+3+1)"""
    length = random.randint(min_len, max_len)
    seq = [random.randint(1, max_val) for _ in range(length)]
    total = sum(seq)
    return {"type": "cumulative_sum", "input": " ".join(str(x) for x in seq), "output": str(total)}


def gen_max_element(max_val=20, min_len=3, max_len=8):
    """Find the maximum element in a sequence.
    Example: '3 7 2 9 1' → '9'"""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    return {"type": "max_element", "input": " ".join(str(x) for x in seq), "output": str(max(seq))}


def gen_min_element(max_val=20, min_len=3, max_len=8):
    """Find the minimum element in a sequence.
    Example: '3 7 2 9 1' → '1'"""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    return {"type": "min_element", "input": " ".join(str(x) for x in seq), "output": str(min(seq))}


def gen_sort_check(max_val=15, min_len=3, max_len=7):
    """Is the sequence sorted (ascending)? Output: S (sorted) or U (unsorted).
    Example: '1 3 5 7' → 'S', '3 1 5' → 'U'"""
    length = random.randint(min_len, max_len)
    if random.random() < 0.5:
        # Generate sorted
        seq = sorted(random.sample(range(max_val + 1), min(length, max_val + 1)))
        answer = "S"
    else:
        # Generate unsorted (ensure not accidentally sorted)
        seq = [random.randint(0, max_val) for _ in range(length)]
        if seq == sorted(seq):
            random.shuffle(seq)
            if seq == sorted(seq) and len(seq) > 1:
                seq[0], seq[-1] = seq[-1], seq[0]
        answer = "S" if seq == sorted(seq) else "U"
    return {"type": "sort_check", "input": " ".join(str(x) for x in seq), "output": answer}


def gen_duplicate_detect(max_val=10, min_len=3, max_len=8):
    """Does the sequence contain any duplicates? Output: D (duplicates) or U (unique).
    Example: '1 3 2 5 3' → 'D', '1 3 2 5' → 'U'"""
    length = random.randint(min_len, max_len)
    if random.random() < 0.5:
        # Generate with duplicates
        seq = [random.randint(0, max_val) for _ in range(length)]
        if len(set(seq)) == len(seq) and length > 1:
            seq[-1] = seq[0]  # force a duplicate
    else:
        # Generate unique
        pool = list(range(max_val + 1))
        random.shuffle(pool)
        seq = pool[:min(length, len(pool))]
    has_dup = len(set(seq)) < len(seq)
    return {"type": "duplicate_detect", "input": " ".join(str(x) for x in seq), "output": "D" if has_dup else "U"}


def gen_element_position(max_val=10, min_len=4, max_len=8):
    """Find the position (0-indexed) of the first occurrence of a target.
    Example: '3 7 2 9 TARGET 2' → '2'"""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    target = random.choice(seq)
    pos = seq.index(target)
    return {"type": "element_position", "input": " ".join(str(x) for x in seq) + " TARGET " + str(target), "output": str(pos)}


def gen_reverse_sequence(max_val=10, min_len=3, max_len=6):
    """Reverse a sequence.
    Example: '1 2 3' → '3 2 1'"""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    reversed_seq = list(reversed(seq))
    return {"type": "reverse_sequence", "input": " ".join(str(x) for x in seq), "output": " ".join(str(x) for x in reversed_seq)}


def gen_fibonacci_next(max_terms=8, min_terms=4):
    """Predict the next Fibonacci-like number. Sequence starts with two random seeds.
    Example: '1 1 2 3 5 ?' → '8'"""
    a = random.randint(1, 5)
    b = random.randint(1, 5)
    n_terms = random.randint(min_terms, max_terms)
    seq = [a, b]
    for _ in range(n_terms - 2):
        seq.append(seq[-1] + seq[-2])
    answer = seq[-1] + seq[-2]
    return {"type": "fibonacci_next", "input": " ".join(str(x) for x in seq) + " ?", "output": str(answer)}


def gen_modular_arithmetic(max_val=20, max_mod=10):
    """Compute a mod b.
    Example: '17 MOD 5' → '2'"""
    b = random.randint(2, max_mod)
    a = random.randint(0, max_val)
    result = a % b
    return {"type": "modular_arithmetic", "input": f"{a} MOD {b}", "output": str(result)}


def gen_comparison_chain(max_val=20, min_len=2, max_len=4):
    """Evaluate a chain of comparisons. All must be true for output T.
    Example: '3 < 5 < 9' → 'T', '3 < 5 > 9' → 'F'"""
    length = random.randint(min_len, max_len)
    values = [random.randint(0, max_val) for _ in range(length + 1)]
    ops = [random.choice(["<", ">"]) for _ in range(length)]

    parts = [str(values[0])]
    all_true = True
    for i, op in enumerate(ops):
        parts.append(op)
        parts.append(str(values[i + 1]))
        if op == "<" and not (values[i] < values[i + 1]):
            all_true = False
        elif op == ">" and not (values[i] > values[i + 1]):
            all_true = False

    return {"type": "comparison_chain", "input": " ".join(parts), "output": "T" if all_true else "F"}
