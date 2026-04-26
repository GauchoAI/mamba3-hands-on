"""compose_logic_gate — composition task for the synapse falsification.

Each example chains two logic-gate operations:

    Input :   COMPOSE op1 a b op2 c
    Output:   r2

    where  r1 = op1(a, b)
           r2 = op2(r1, c)

`op1, op2 ∈ {AND, OR, XOR, NAND, NOR, XNOR}` (matching the logic_gate
specialist's vocabulary). `a, b, c ∈ {0, 1}`.

There are 6 × 6 × 8 = 288 distinct inputs; the model can't memorize the
truth table at any reasonable scale without paying for it. To get this
right at high accuracy a small router has to do the same primitive
twice — exactly what register-level invocation of a frozen logic_gate
specialist would help with.

This is the falsification target for the synapse: a tiny router that
solves it WITH the specialist but plateaus WITHOUT.
"""
import random


GATES = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]


def _apply(op, a, b):
    if op == "AND":  return a & b
    if op == "OR":   return a | b
    if op == "XOR":  return a ^ b
    if op == "NAND": return 1 - (a & b)
    if op == "NOR":  return 1 - (a | b)
    if op == "XNOR": return 1 - (a ^ b)
    raise ValueError(op)


def gen_compose_logic_gate():
    """One composition example."""
    op1 = random.choice(GATES)
    op2 = random.choice(GATES)
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    c = random.randint(0, 1)
    r1 = _apply(op1, a, b)
    r2 = _apply(op2, r1, c)
    return {
        "type": "compose_logic_gate",
        "input": f"COMPOSE {op1} {a} {b} {op2} {c}",
        "output": str(r2),
    }


def gen_compose_logic_gate_3():
    """Three-deep chain — tests how the synapse scales with chain depth.

        r3 = op3(op2(op1(a, b), c), d)
    """
    op1, op2, op3 = (random.choice(GATES) for _ in range(3))
    a, b, c, d = (random.randint(0, 1) for _ in range(4))
    r1 = _apply(op1, a, b)
    r2 = _apply(op2, r1, c)
    r3 = _apply(op3, r2, d)
    return {
        "type": "compose_logic_gate_3",
        "input": f"COMPOSE3 {op1} {a} {b} {op2} {c} {op3} {d}",
        "output": str(r3),
    }


if __name__ == "__main__":
    # Sanity sample
    for _ in range(8):
        print(gen_compose_logic_gate())
