"""
Final Boss Tasks — unseen task types the model has never trained on.

If the model has truly learned pattern recognition algorithms (not just
memorized task formats), it should handle these with minimal examples.

Categories:
  - Set operations (union, intersection, subset)
  - Sorting and ordering
  - Arithmetic variants (modular, min/max, range)
  - Sequence transforms (reverse, rotate, interleave)
  - Logic (AND/OR/XOR on binary sequences)
  - Counting variants (unique count, majority element)
"""
import random


# ── Set operations ──────────────────────────────────────────────────

def gen_set_union(max_val=10, max_len=5):
    """Union of two sets (sorted, deduplicated)."""
    len_a = random.randint(2, max_len)
    len_b = random.randint(2, max_len)
    a = sorted(set(random.randint(0, max_val) for _ in range(len_a)))
    b = sorted(set(random.randint(0, max_val) for _ in range(len_b)))
    result = sorted(set(a) | set(b))
    a_str = " ".join(str(x) for x in a)
    b_str = " ".join(str(x) for x in b)
    r_str = " ".join(str(x) for x in result)
    return {"type": "set_union", "input": f"{a_str} | {b_str}", "output": r_str}


def gen_set_intersection(max_val=10, max_len=5):
    """Intersection of two sets."""
    len_a = random.randint(2, max_len)
    len_b = random.randint(2, max_len)
    # Ensure some overlap
    pool = list(range(max_val + 1))
    shared = random.sample(pool, min(2, len(pool)))
    a = sorted(set(shared + [random.randint(0, max_val) for _ in range(len_a)]))
    b = sorted(set(shared + [random.randint(0, max_val) for _ in range(len_b)]))
    result = sorted(set(a) & set(b))
    a_str = " ".join(str(x) for x in a)
    b_str = " ".join(str(x) for x in b)
    r_str = " ".join(str(x) for x in result) if result else "0"
    return {"type": "set_intersection", "input": f"{a_str} & {b_str}", "output": r_str}


def gen_is_subset(max_val=8, max_len=4):
    """Is set A a subset of set B?"""
    len_b = random.randint(3, max_len + 2)
    b = sorted(set(random.randint(0, max_val) for _ in range(len_b)))
    if random.random() < 0.5 and len(b) >= 2:
        # Make A a true subset
        a = sorted(random.sample(b, random.randint(1, max(1, len(b) - 1))))
        answer = "S"  # yes/subset
    else:
        a = sorted(set(random.randint(0, max_val) for _ in range(random.randint(1, max_len))))
        answer = "S" if set(a).issubset(set(b)) else "D"
    a_str = " ".join(str(x) for x in a)
    b_str = " ".join(str(x) for x in b)
    return {"type": "is_subset", "input": f"{a_str} ? {b_str}", "output": answer}


# ── Sorting and ordering ───────────────────────────────────────────

def gen_sort_sequence(max_val=15, min_len=3, max_len=6):
    """Sort a sequence in ascending order."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    result = sorted(seq)
    seq_str = " ".join(str(x) for x in seq)
    r_str = " ".join(str(x) for x in result)
    return {"type": "sort_sequence", "input": seq_str, "output": r_str}


def gen_find_min(max_val=20, min_len=3, max_len=7):
    """Find the minimum value in a sequence."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    result = min(seq)
    seq_str = " ".join(str(x) for x in seq)
    return {"type": "find_min", "input": seq_str, "output": str(result)}


def gen_find_max(max_val=20, min_len=3, max_len=7):
    """Find the maximum value in a sequence."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    result = max(seq)
    seq_str = " ".join(str(x) for x in seq)
    return {"type": "find_max", "input": seq_str, "output": str(result)}


def gen_find_range(max_val=20, min_len=3, max_len=7):
    """Find max - min of a sequence."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    result = max(seq) - min(seq)
    seq_str = " ".join(str(x) for x in seq)
    return {"type": "find_range", "input": seq_str, "output": str(result)}


# ── Arithmetic variants ────────────────────────────────────────────

def gen_sum_sequence(max_val=10, min_len=2, max_len=5):
    """Sum all values in a sequence."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    result = sum(seq)
    seq_str = " ".join(str(x) for x in seq)
    return {"type": "sum_sequence", "input": seq_str, "output": str(result)}


def gen_modular_arithmetic(max_val=15, max_mod=7):
    """a + b mod m."""
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    m = random.randint(2, max_mod)
    result = (a + b) % m
    return {"type": "modular_arithmetic", "input": f"{a} {b} {m}", "output": str(result)}


def gen_difference(max_val=20):
    """Absolute difference of two numbers."""
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    result = abs(a - b)
    return {"type": "difference", "input": f"{a} {b}", "output": str(result)}


# ── Sequence transforms ────────────────────────────────────────────

def gen_reverse_sequence(max_val=10, min_len=3, max_len=6):
    """Reverse a sequence."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    result = seq[::-1]
    seq_str = " ".join(str(x) for x in seq)
    r_str = " ".join(str(x) for x in result)
    return {"type": "reverse_sequence", "input": seq_str, "output": r_str}


def gen_rotate_sequence(max_val=10, min_len=4, max_len=7):
    """Rotate sequence left by 1 position."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    result = seq[1:] + seq[:1]
    seq_str = " ".join(str(x) for x in seq)
    r_str = " ".join(str(x) for x in result)
    return {"type": "rotate_sequence", "input": seq_str, "output": r_str}


def gen_deduplicate(max_val=8, min_len=4, max_len=8):
    """Remove duplicates preserving order."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    seen = set()
    result = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            result.append(x)
    seq_str = " ".join(str(x) for x in seq)
    r_str = " ".join(str(x) for x in result)
    return {"type": "deduplicate", "input": seq_str, "output": r_str}


# ── Logic ───────────────────────────────────────────────────────────

def gen_bitwise_xor(min_len=3, max_len=6):
    """XOR two binary sequences element-wise."""
    length = random.randint(min_len, max_len)
    a = [random.randint(0, 1) for _ in range(length)]
    b = [random.randint(0, 1) for _ in range(length)]
    result = [x ^ y for x, y in zip(a, b)]
    a_str = " ".join(str(x) for x in a)
    b_str = " ".join(str(x) for x in b)
    r_str = " ".join(str(x) for x in result)
    return {"type": "bitwise_xor", "input": f"{a_str} | {b_str}", "output": r_str}


def gen_count_ones(min_len=3, max_len=8):
    """Count the number of 1s in a binary sequence."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, 1) for _ in range(length)]
    result = sum(seq)
    seq_str = " ".join(str(x) for x in seq)
    return {"type": "count_ones", "input": seq_str, "output": str(result)}


# ── Counting variants ──────────────────────────────────────────────

def gen_unique_count(max_val=10, min_len=3, max_len=8):
    """Count unique values in a sequence."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    result = len(set(seq))
    seq_str = " ".join(str(x) for x in seq)
    return {"type": "unique_count", "input": seq_str, "output": str(result)}


def gen_majority_element(max_val=5, min_len=5, max_len=9):
    """Find the most frequent element (ties: smallest)."""
    length = random.randint(min_len, max_len)
    # Ensure there IS a clear majority
    majority = random.randint(0, max_val)
    count = length // 2 + 1
    others = [random.randint(0, max_val) for _ in range(length - count)]
    seq = [majority] * count + others
    random.shuffle(seq)
    # Find actual majority (handle ties)
    from collections import Counter
    counts = Counter(seq)
    max_count = max(counts.values())
    result = min(v for v, c in counts.items() if c == max_count)
    seq_str = " ".join(str(x) for x in seq)
    return {"type": "majority_element", "input": seq_str, "output": str(result)}


def gen_second_largest(max_val=20, min_len=3, max_len=6):
    """Find the second largest value."""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    unique_sorted = sorted(set(seq), reverse=True)
    result = unique_sorted[1] if len(unique_sorted) >= 2 else unique_sorted[0]
    seq_str = " ".join(str(x) for x in seq)
    return {"type": "second_largest", "input": seq_str, "output": str(result)}


# ── Registry ────────────────────────────────────────────────────────

BOSS_GENERATORS = {
    "set_union": gen_set_union,
    "set_intersection": gen_set_intersection,
    "is_subset": gen_is_subset,
    "sort_sequence": gen_sort_sequence,
    "find_min": gen_find_min,
    "find_max": gen_find_max,
    "find_range": gen_find_range,
    "sum_sequence": gen_sum_sequence,
    "modular_arithmetic": gen_modular_arithmetic,
    "difference": gen_difference,
    "reverse_sequence": gen_reverse_sequence,
    "rotate_sequence": gen_rotate_sequence,
    "deduplicate": gen_deduplicate,
    "bitwise_xor": gen_bitwise_xor,
    "count_ones": gen_count_ones,
    "unique_count": gen_unique_count,
    "majority_element": gen_majority_element,
    "second_largest": gen_second_largest,
}


if __name__ == "__main__":
    # Quick validation
    for name, fn in BOSS_GENERATORS.items():
        ex = fn()
        print(f"  {name}: {ex['input']} → {ex['output']}")
    print(f"\n{len(BOSS_GENERATORS)} boss tasks ready.")
