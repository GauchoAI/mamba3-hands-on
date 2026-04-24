"""Level 2 composition generators — problems requiring multiple skills.

These combine level 0 and level 1 abilities: counting + comparison,
arithmetic + logic, sequence analysis + output formatting.
"""
import random


def gen_count_above_threshold(max_val=10, max_threshold=8, min_len=3, max_len=8):
    """Count how many elements are above a threshold.
    Example: '3 7 2 9 1 ABOVE 5' → '2' (7 and 9 are above 5)"""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    threshold = random.randint(1, max_threshold)
    count = sum(1 for x in seq if x > threshold)
    return {
        "type": "count_above_threshold",
        "input": " ".join(str(x) for x in seq) + " ABOVE " + str(threshold),
        "output": str(count),
    }


def gen_second_largest(max_val=20, min_len=3, max_len=7):
    """Find the second largest element. Requires sorting mentally.
    Example: '5 2 8 3' → '5'"""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    # Ensure at least 2 distinct values
    if len(set(seq)) < 2:
        seq[-1] = (seq[0] + 1) % (max_val + 1)
    sorted_unique = sorted(set(seq), reverse=True)
    answer = sorted_unique[1] if len(sorted_unique) >= 2 else sorted_unique[0]
    return {
        "type": "second_largest",
        "input": " ".join(str(x) for x in seq),
        "output": str(answer),
    }


def gen_range_of_sequence(max_val=20, min_len=3, max_len=8):
    """Compute max - min of a sequence.
    Example: '3 7 2 9' → '7' (9-2)"""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    answer = max(seq) - min(seq)
    return {
        "type": "range_of_sequence",
        "input": " ".join(str(x) for x in seq),
        "output": str(answer),
    }


def gen_conditional_sum(max_val=10, min_len=3, max_len=8):
    """Sum only the even (or odd) numbers in a sequence.
    Example: '3 4 7 2 EVEN' → '6' (4+2)"""
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    if random.random() < 0.5:
        cond = "EVEN"
        total = sum(x for x in seq if x % 2 == 0)
    else:
        cond = "ODD"
        total = sum(x for x in seq if x % 2 == 1)
    return {
        "type": "conditional_sum",
        "input": " ".join(str(x) for x in seq) + " " + cond,
        "output": str(total),
    }


def gen_majority_element(max_val=5, min_len=5, max_len=9):
    """Find the element that appears most often. If tie, smallest wins.
    Example: '2 3 2 1 2' → '2'"""
    length = random.randint(min_len, max_len)
    # Sometimes guarantee a majority
    if random.random() < 0.6:
        majority = random.randint(0, max_val)
        n_majority = length // 2 + 1
        seq = [majority] * n_majority + [random.randint(0, max_val) for _ in range(length - n_majority)]
        random.shuffle(seq)
    else:
        seq = [random.randint(0, max_val) for _ in range(length)]
    # Count frequencies
    from collections import Counter
    counts = Counter(seq)
    max_count = max(counts.values())
    # Smallest element with max count
    answer = min(x for x, c in counts.items() if c == max_count)
    return {
        "type": "majority_element",
        "input": " ".join(str(x) for x in seq),
        "output": str(answer),
    }
