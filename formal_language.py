"""formal_language — generators for tier-1 bidirectional language tasks.

Two tasks that are exact inverses of each other:

  bool_expr_to_truth_table:
    input:  "BOOLTAB <op> <arg1> [<arg2>]"
    output: 4 bits "abcd" (rows in order f(0,0) f(0,1) f(1,0) f(1,1))

  truth_table_to_bool_expr:
    input:  "TABEXPR <4-bit>"
    output: smallest canonical expression for that truth table

The two together let us check round-trip identity on every example,
which is the litmus test for whether the model learned the semantics
of boolean logic or just surface patterns.

Operators: AND, OR, XOR, NAND, NOR, XNOR (binary) and NOT (unary).
Variables: a, b. Negated literals (NOT a, NOT b) are also allowed
as args to a binary op.

There are 16 distinct truth tables for f(a, b). For each, we hard-pick
a canonical smallest expression. This makes the reverse direction a
deterministic function (rather than one-to-many).
"""
import random


# 16 truth tables → canonical smallest expressions.
# Truth table notation: 4 bits in order f(0,0), f(0,1), f(1,0), f(1,1).
TT_TO_EXPR = {
    "0000": "AND a NOT a",         # constant FALSE
    "0001": "AND a b",
    "0010": "AND a NOT b",
    "0011": "a",                    # f = a
    "0100": "AND NOT a b",
    "0101": "b",                    # f = b
    "0110": "XOR a b",
    "0111": "OR a b",
    "1000": "NOR a b",
    "1001": "XNOR a b",
    "1010": "NOT b",
    "1011": "OR a NOT b",
    "1100": "NOT a",
    "1101": "OR NOT a b",
    "1110": "NAND a b",
    "1111": "OR a NOT a",          # constant TRUE
}
EXPR_TO_TT = {v: k for k, v in TT_TO_EXPR.items()}


def _eval(expr_tokens, a, b):
    """Evaluate a tokenized expression at given a,b values. Returns 0/1."""
    # Recursive descent on a token list (operator-prefix).
    pos = [0]

    def parse_arg():
        tok = expr_tokens[pos[0]]
        if tok == "a":
            pos[0] += 1; return a
        if tok == "b":
            pos[0] += 1; return b
        if tok == "NOT":
            pos[0] += 1
            v = parse_arg()
            return 1 - v
        # Otherwise it's a binary op
        op = tok
        pos[0] += 1
        x = parse_arg()
        y = parse_arg()
        if op == "AND":  return x & y
        if op == "OR":   return x | y
        if op == "XOR":  return x ^ y
        if op == "NAND": return 1 - (x & y)
        if op == "NOR":  return 1 - (x | y)
        if op == "XNOR": return 1 - (x ^ y)
        raise ValueError(f"unknown op {op}")

    return parse_arg()


def _truth_table(expr_str):
    """Compute the 4-bit truth table for an expression string."""
    toks = expr_str.split()
    bits = []
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        bits.append(str(_eval(toks, a, b)))
    return "".join(bits)


# Pool of expressions to sample from (for the forward direction).
_EXPRESSION_POOL = []


def _build_pool():
    if _EXPRESSION_POOL:
        return
    OPS = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]
    LITS = ["a", "b", "NOT a", "NOT b"]
    # Binary op over two literals (each can be NOT'd)
    for op in OPS:
        for lhs in LITS:
            for rhs in LITS:
                expr = f"{op} {lhs} {rhs}"
                _EXPRESSION_POOL.append(expr)
    # NOT a, NOT b alone
    _EXPRESSION_POOL.extend(["NOT a", "NOT b", "a", "b"])


def gen_bool_expr_to_truth_table():
    """Forward direction: expression → 4-bit truth table."""
    _build_pool()
    expr = random.choice(_EXPRESSION_POOL)
    tt = _truth_table(expr)
    return {
        "type": "bool_expr_to_truth_table",
        "input": f"BOOLTAB {expr}",
        "output": tt,
    }


def gen_truth_table_to_bool_expr():
    """Reverse direction: 4-bit truth table → smallest canonical expression."""
    tt = random.choice(list(TT_TO_EXPR.keys()))
    expr = TT_TO_EXPR[tt]
    return {
        "type": "truth_table_to_bool_expr",
        "input": f"TABEXPR {tt}",
        "output": expr,
    }


# ── Tier 2 — depth-2 expressions evaluated to truth tables ────────────
#
# A depth-2 expression composes two depth-1 expressions with an outer op:
#
#   <outer_op> <inner1> <inner2>
#
# where each <inner> is itself one of the depth-1 forms (op + literals).
# The truth table over (a, b) is computed end-to-end. A router solving
# this can use the tier-1 bool_expr_to_truth_table specialist on each
# <inner> sub-expression and combine the results — exactly the
# function-call composition we want to demonstrate.

def _depth1_expression():
    """Sample one depth-1 expression as a string."""
    OPS = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]
    LITS = ["a", "b", "NOT a", "NOT b"]
    if random.random() < 0.2:
        return random.choice(LITS)  # bare literal (depth 0)
    op = random.choice(OPS)
    lhs = random.choice(LITS)
    rhs = random.choice(LITS)
    return f"{op} {lhs} {rhs}"


def gen_bool_expr_depth2():
    """Depth-2: outer op over two depth-1 sub-expressions."""
    OUTER_OPS = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]
    op = random.choice(OUTER_OPS)
    inner1 = _depth1_expression()
    inner2 = _depth1_expression()

    # Build a parser-friendly expression string. Inner expressions are
    # already in operator-prefix form so concatenation just works.
    expr = f"{op} {inner1} {inner2}"
    tt = _truth_table(expr)
    return {
        "type": "bool_expr_depth2",
        "input": f"BOOLTAB2 {expr}",
        "output": tt,
    }


def _depth2_expression():
    """Sample one depth-2 expression as a string (no marker)."""
    OUTER_OPS = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]
    if random.random() < 0.3:
        return _depth1_expression()  # sometimes depth-1 to keep it varied
    op = random.choice(OUTER_OPS)
    return f"{op} {_depth1_expression()} {_depth1_expression()}"


def gen_bool_simplify():
    """Symbolic simplification: arbitrary depth-1 or depth-2 boolean
    expression → smallest equivalent canonical form.

    Many expressions map to the same truth table; the canonical form
    is the unique smallest representative for that equivalence class.
    Example mappings:

      'AND a a'        → 'a'          (idempotence)
      'OR a NOT a'     → 'OR a NOT a' (canonical for TRUE; surface stays)
      'NAND NOT a NOT b' → 'OR a b'   (De Morgan)
      'XOR a NOT b'    → 'XNOR a b'   (rewrite)
      'AND OR a b TRUE' → ... (model has to reduce nested literals)

    The mapping is deterministic: any expression evaluates to one of
    16 truth tables (2 vars), and each truth table has exactly one
    canonical form. So this is a many-to-one task.
    """
    if random.random() < 0.5:
        expr = _depth1_expression()
    else:
        OUTER_OPS = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]
        op = random.choice(OUTER_OPS)
        expr = f"{op} {_depth1_expression()} {_depth1_expression()}"
    tt = _truth_table(expr)
    canonical = TT_TO_EXPR[tt]
    return {
        "type": "bool_simplify",
        "input": f"SIMPLIFY {expr}",
        "output": canonical,
    }


def _arith_expr(max_depth=2):
    """Sample an infix arithmetic expression as a list of tokens.
    Returns (infix_tokens, postfix_tokens, value)."""
    OPS = ["+", "-", "*"]
    if max_depth == 0 or random.random() < 0.4:
        n = random.randint(0, 9)
        return [str(n)], [str(n)], n
    op = random.choice(OPS)
    li, lp, lv = _arith_expr(max_depth - 1)
    ri, rp, rv = _arith_expr(max_depth - 1)
    if op == "+":   v = lv + rv
    elif op == "-": v = lv - rv
    else:           v = lv * rv
    # Infix with explicit parens to keep parse unambiguous (and match
    # the byte tokenizer's vocabulary).
    infix = ["("] + li + [op] + ri + [")"]
    postfix = lp + rp + [op]
    return infix, postfix, v


def gen_infix_to_postfix():
    """Forward direction: infix expression → postfix.
    Same expression, two notations — tests parsing without semantics."""
    inf, post, _ = _arith_expr(max_depth=2)
    return {
        "type": "infix_to_postfix",
        "input": "INFIX2POST " + " ".join(inf),
        "output": " ".join(post),
    }


def gen_postfix_to_infix():
    """Reverse direction: postfix → fully-parenthesized infix.
    The mapping is deterministic (one canonical infix per postfix)."""
    inf, post, _ = _arith_expr(max_depth=2)
    return {
        "type": "postfix_to_infix",
        "input": "POST2INFIX " + " ".join(post),
        "output": " ".join(inf),
    }


def gen_bool_expr_depth3():
    """Depth-3: outer op over two depth-2 sub-expressions.

    The natural tier-3 task: a router solving this should use the
    bool_expr_depth2 specialist on each inner sub-expression and
    combine via the outer op. Tests whether the synapse architecture
    composes through one more layer.
    """
    OUTER_OPS = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]
    op = random.choice(OUTER_OPS)
    inner1 = _depth2_expression()
    inner2 = _depth2_expression()
    expr = f"{op} {inner1} {inner2}"
    tt = _truth_table(expr)
    return {
        "type": "bool_expr_depth3",
        "input": f"BOOLTAB3 {expr}",
        "output": tt,
    }


if __name__ == "__main__":
    print("=== forward (BOOLTAB) ===")
    for _ in range(8):
        print(gen_bool_expr_to_truth_table())
    print()
    print("=== reverse (TABEXPR) ===")
    for _ in range(8):
        print(gen_truth_table_to_bool_expr())
    print()
    # Sanity check: all 16 reverse mappings produce expressions that
    # forward-evaluate back to the original truth table.
    print("=== round-trip identity check ===")
    ok = 0
    for tt, expr in TT_TO_EXPR.items():
        rt = _truth_table(expr)
        flag = "OK" if rt == tt else f"MISMATCH ({rt})"
        if rt == tt: ok += 1
        print(f"  {tt} → {expr!r:25s}  → {rt}  {flag}")
    print(f"  {ok}/16 round-trip clean")
