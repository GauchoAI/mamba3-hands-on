"""Level 3 long-range generators — problems that REQUIRE register usage.

These problems cannot be memorized. The model must learn to use the SSM
hidden state as working memory (registers) to track carries, disk positions,
or intermediate results across many tokens.

The SSM has 2,048 registers per layer (8 heads × 16 headdim × 16 d_state).
These problems are designed to push the model toward using them as a
Turing machine tape rather than pattern-matching.
"""
import random


def gen_addition(n_digits=2):
    """Multi-digit addition with carry propagation.
    Example: '47 + 38' → '85'
    Long: '3847 + 2965' → '6812'

    Carry propagation forces the model to track state across positions.
    """
    max_val = 10 ** n_digits - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    result = a + b
    return {"type": "addition", "input": f"{a} + {b}", "output": str(result)}


def gen_subtraction(n_digits=2):
    """Multi-digit subtraction with borrowing.
    Example: '85 - 38' → '47'

    Always produces non-negative result (larger - smaller).
    """
    max_val = 10 ** n_digits - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    if a < b:
        a, b = b, a
    result = a - b
    return {"type": "subtraction", "input": f"{a} - {b}", "output": str(result)}


def gen_multiplication(n_digits=2, max_second=1):
    """Multiplication — one number can be multi-digit, second is smaller.
    Example: '23 * 7' → '161'

    Start with single-digit multiplier, ramp to multi-digit.
    """
    max_a = 10 ** n_digits - 1
    max_b = 10 ** max_second - 1
    a = random.randint(1, max(1, max_a))
    b = random.randint(1, max(1, max_b))
    result = a * b
    return {"type": "multiplication", "input": f"{a} * {b}", "output": str(result)}


def gen_digit_sum(n_digits=3, max_digits=6):
    """Sum of individual digits of a number.
    Example: '4829' → '23' (4+8+2+9)

    Forces the model to process each digit and accumulate.
    """
    n = random.randint(n_digits, max_digits)
    number = random.randint(10 ** (n - 1), 10 ** n - 1)
    result = sum(int(d) for d in str(number))
    return {"type": "digit_sum", "input": str(number), "output": str(result)}


def gen_tower_of_hanoi(n_disks=2):
    """Tower of Hanoi — output the number of moves required.
    Example: 'HANOI 3' → '7' (2^n - 1)

    For small n: output the move count.
    This tests whether the model can learn the recurrence 2^n - 1.
    """
    n = random.randint(1, n_disks)
    moves = 2 ** n - 1
    return {"type": "tower_of_hanoi", "input": f"HANOI {n}", "output": str(moves)}


def gen_tower_of_hanoi_binary(n_disks=2):
    """Hanoi move-count in BINARY: 2^n - 1 = n ones.
    Example: 'HANOIBIN 3' → '111'

    Diagnostic for the trajectory-distillation experiment: the
    decimal-output Hanoi failed to extrapolate past the trained
    range (memorized templates). The binary form removes the
    binary→decimal converter — the model only needs to count
    to n and emit n '1's. If THIS extrapolates while decimal
    doesn't, the failure was the output head, not the program.
    If this also fails, the recurrence itself isn't generalising.
    """
    n = random.randint(1, n_disks)
    return {"type": "tower_of_hanoi_binary", "input": f"HANOIBIN {n}", "output": "1" * n}


def _fib(n: int) -> int:
    """Iterative Fibonacci: F(0)=0, F(1)=1, F(2)=1, F(3)=2, ..."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def gen_fib_unary(n_max=10):
    """Fibonacci in UNARY: 'FIB n' → '1' * F(n).

    Output length = F(n), which grows exponentially — F(10)=55,
    F(15)=610, F(20)=6765. So n_max should stay small.

    Sanity-check counterpart to HANOIBIN: instead of the counter
    being the parsed integer n, the counter is F(n), computed by
    the oracle from n. Tests whether the LoopCounter pattern
    transfers when the counter value is a non-trivial function
    of the input.
    """
    n = random.randint(1, n_max)
    fn = _fib(n)
    return {"type": "fib_unary", "input": f"FIB {n}", "output": "1" * fn}


def gen_fib_decimal(n_max=20):
    """Fibonacci in DECIMAL: 'FIBD n' → str(F(n)).

    Example: FIBD 10 → '55', FIBD 30 → '832040', FIBD 100 → 21 digits.

    The output is the actual decimal representation of F(n). Unlike
    fib_unary where every output token is the same character ('1'),
    here the iteration token *varies per position* — it's the digit
    at each position of F(n). This stress-tests the per-position
    iter_token extension of LoopCounter.

    Counter at SEP = digit_count(F(n)) (much smaller than F(n)
    itself: F(20)=6765 has 4 digits, F(100) has 21 digits).
    """
    n = random.randint(1, n_max)
    fn = _fib(n)
    return {"type": "fib_decimal", "input": f"FIBD {n}", "output": str(fn)}


def gen_string_length(max_len=10):
    """Count the number of elements in a sequence.
    Example: '5 3 8 1 4 2' → '6'

    Simple but forces sequential counting — a warmup for register usage.
    """
    length = random.randint(1, max_len)
    seq = [random.randint(0, 9) for _ in range(length)]
    return {"type": "string_length", "input": " ".join(str(x) for x in seq), "output": str(length)}


def gen_running_max(max_val=10, min_len=3, max_len=8):
    """Output the running maximum at each position.
    Example: '3 1 4 1 5' → '3 3 4 4 5'

    Multi-token output that requires tracking state across the sequence.
    """
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, max_val) for _ in range(length)]
    running = []
    current_max = 0
    for x in seq:
        current_max = max(current_max, x)
        running.append(current_max)
    return {
        "type": "running_max",
        "input": " ".join(str(x) for x in seq),
        "output": " ".join(str(x) for x in running),
    }
