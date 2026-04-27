"""hanoi_trace — Hanoi but the model emits the iterative computation trace
instead of just the final answer. Forces program-shape output.

The recurrence for the number of moves in Tower of Hanoi is:
    x_1 = 1
    x_{k+1} = 2 * x_k + 1

A program that runs the recurrence emits:
    n=1  →  1
    n=2  →  1 3
    n=3  →  1 3 7
    n=4  →  1 3 7 15
    n=10 →  1 3 7 15 31 63 127 255 511 1023

Output length scales linearly with n; the answer to "did the model
internalize the recurrence" is "for what n is the trace still correct."
A bounded pattern matcher fails at some n; a true program runs to any n.

This is the vertical-depth experiment: same task, but the output is the
program's execution trace, not just its terminal value. Sustained correct
trace length directly measures how much program the model holds.
"""
import random


def gen_tower_of_hanoi_trace(n_disks=4):
    """One Hanoi-trace example.

    Args:
      n_disks: max number of disks (curriculum knob). The actual n is
               sampled uniformly in [1, n_disks].

    Returns:
      {"input": "HANOITRACE n", "output": "x_1 x_2 ... x_n"}
    """
    n = random.randint(1, n_disks)
    trace = []
    x = 1
    trace.append(x)
    for _ in range(n - 1):
        x = 2 * x + 1
        trace.append(x)
    return {
        "type": "tower_of_hanoi_trace",
        "input": f"HANOITRACE {n}",
        "output": " ".join(str(v) for v in trace),
    }


if __name__ == "__main__":
    for n in [1, 2, 3, 5, 10, 20]:
        random.seed(n)
        ex = gen_tower_of_hanoi_trace(n_disks=n)
        print(f"  n_disks={n:3d}  input={ex['input']!r:25s} output={ex['output']!r}")
