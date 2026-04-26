"""dual_task — multi-specialist composition test.

Two independent sub-questions in one sequence; the router has to solve
both. Output is two characters: the gate result and the count result.

  Input  : DUAL op a b ; values list ABOVE threshold
  Output : "<gate_result> <count_result>"

  Example:
    input:  "DUAL XOR 1 0 ; 0 7 10 0 10 ABOVE 8"
    output: "1 2"

Two specialists are needed:
  - logic_gate (handles "op a b") — already mastered at 100%
  - count_above_threshold (handles "values list ABOVE threshold")
        — already mastered at 97%

The router has to attend to the *right specialist for the right
position* in the sequence. Two separate gates, two separate W_recv
matrices — that's what the architectural primitive validates: many
synapses, the router opens the right one for the right sub-region.

Bonus: this also tests whether the router can discriminate between
specialists rather than just opening every gate uniformly.
"""
import random


GATES = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]


def _apply_gate(op, a, b):
    if op == "AND":  return a & b
    if op == "OR":   return a | b
    if op == "XOR":  return a ^ b
    if op == "NAND": return 1 - (a & b)
    if op == "NOR":  return 1 - (a | b)
    if op == "XNOR": return 1 - (a ^ b)
    raise ValueError(op)


def gen_dual_task():
    op = random.choice(GATES)
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    gate_r = _apply_gate(op, a, b)

    # Match the count_above_threshold generator's profile.
    n_values = random.randint(3, 6)
    values = [random.randint(0, 10) for _ in range(n_values)]
    threshold = random.randint(0, 10)
    count_r = sum(1 for v in values if v > threshold)

    return {
        "type": "dual_task",
        "input": f"DUAL {op} {a} {b} ; {' '.join(map(str, values))} ABOVE {threshold}",
        "output": f"{gate_r} {count_r}",
    }


if __name__ == "__main__":
    for _ in range(8):
        print(gen_dual_task())
