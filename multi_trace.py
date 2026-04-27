"""multi_trace — Hanoi traces with the same recurrence run multiple times.

Each example asks for K traces back-to-back:

  Input:  "MULTI <n_1> <n_2> ... <n_K>"
  Output: "<trace_1> | <trace_2> | ... | <trace_K>"

  trace_i = "1 3 7 ... x_{n_i}"  (the iterative Hanoi trace)

The point: every trace starts with `1 3 7 ...` regardless of n_i. If the
model has memorized a per-n template, the FIRST trace is fine but the
second/third can't reuse that template — the model has to *restart*
the recurrence and run it again.

This is the input-independent-prefix experiment: a model running a
program emits `1 3 7 ...` from any starting condition (the | separator
acts as "reset the recurrence"). A model doing input-keyed lookup
fails the second trace because there's no template for "trace after a |".

Curriculum stages vary BOTH the maximum n_i and K (number of traces).
At the simplest stage K=2; at deeper stages K=4, 8, etc.
"""
import random


def gen_multi_trace(n_disks=4, max_traces=2):
    """One example with K traces (K random in [1, max_traces])."""
    K = random.randint(1, max_traces)
    ns = [random.randint(1, n_disks) for _ in range(K)]
    pieces = []
    for n in ns:
        x = 1
        trace = [str(x)]
        for _ in range(n - 1):
            x = 2 * x + 1
            trace.append(str(x))
        pieces.append(" ".join(trace))
    return {
        "type": "multi_trace",
        "input": f"MULTI {' '.join(str(n) for n in ns)}",
        "output": " | ".join(pieces),
    }


if __name__ == "__main__":
    for seed in range(8):
        random.seed(seed)
        ex = gen_multi_trace(n_disks=5, max_traces=4)
        print(f"  in={ex['input']!r:25s}  out={ex['output']!r}")
