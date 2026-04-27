"""hanoi_oracle — programmatic expectation of the register trajectory.

For the Hanoi task (input "HANOI n", output "2^n − 1"), we know the
correct answer analytically. The hypothesis: if the model has an
explicit register bank, the natural place to store the answer is r0,
and r0 should hold a value-derived embedding of 2^n − 1 throughout the
output span.

The oracle gives us a *target trajectory* for the register write values
at every position. Auxiliary MSE loss against this target turns the
end-to-end learning problem ("find ANY weights that match outputs")
into a richer signal ("match this specific trajectory at this register
at every timestep"). This is the trajectory-distillation variant of
"FD-style training" — fine-grained intermediate supervision rather
than just terminal-output supervision.

Encoding: binary-bits of (2^n − 1) into a d_register-length float
vector. For n ≤ d_register this is exact; for n > d_register the
oracle truncates (the model has to compute beyond what the register
can hold — which is the architectural limit we want to expose).
"""
import re
import torch


HANOI_INPUT_RE = re.compile(rb"HANOI (\d+)")
HANOITRACE_INPUT_RE = re.compile(rb"HANOITRACE (\d+)")
HANOIBIN_INPUT_RE = re.compile(rb"HANOIBIN (\d+)")


SEP_TOKEN = 258


def parse_n_from_tokens(token_tensor):
    """Given an int64 token tensor (B, L), parse the integer 'n' from the
    HANOI / HANOITRACE / HANOIBIN prefix in each row. Stops at SEP so we
    don't bleed answer bytes into the parse. Returns a list of ints,
    one per batch.
    """
    Bn = token_tensor.shape[0]
    out = []
    for i in range(Bn):
        row = token_tensor[i].tolist()
        # Stop at the SEP token so the answer bytes don't leak in.
        if SEP_TOKEN in row:
            row = row[: row.index(SEP_TOKEN)]
        bytes_list = [b for b in row if 32 <= b < 127]
        text = bytes(bytes_list)
        m = (HANOI_INPUT_RE.search(text)
             or HANOITRACE_INPUT_RE.search(text)
             or HANOIBIN_INPUT_RE.search(text))
        out.append(int(m.group(1)) if m else 0)
    return out


def find_sep_positions(token_tensor):
    """Per-row position of the (first) SEP token. -1 if not found."""
    out = []
    rows = token_tensor.tolist()
    for row in rows:
        out.append(row.index(SEP_TOKEN) if SEP_TOKEN in row else -1)
    return out


def hanoibin_counter_trajectory(token_tensor, sentinel: int, device="cpu"):
    """Build the per-position counter trajectory for the unary-output
    HANOIBIN task.

    For each batch item with parsed integer n at SEP position s:
        positions [0, s)        -> sentinel  (input span, counter inactive)
        position  s             -> n         (predicts first '1', count=n)
        position  s+k  (1..n-1) -> n-k       (predicts (k+1)-th '1')
        position  s+n           -> 0         (predicts EOS)
        positions s+n+1..       -> sentinel  (irrelevant past EOS)

    Returns int64 (B, L) tensor and parsed ns (list).

    Why these positions: at training the model's prediction at
    position p targets position p+1. So at p=s the model is about to
    emit the first answer '1' and the counter says "n remaining."
    At p=s+n the model is about to emit EOS; counter is 0.
    """
    B, L = token_tensor.shape
    counter = torch.full((B, L), sentinel, dtype=torch.long, device=device)
    ns = parse_n_from_tokens(token_tensor)
    seps = find_sep_positions(token_tensor)
    for i, (n, s) in enumerate(zip(ns, seps)):
        if n <= 0 or s < 0:
            continue
        # Counter is n at SEP, then n-1, n-2, ... 0 at SEP+n.
        for k in range(n + 1):
            p = s + k
            if 0 <= p < L:
                counter[i, p] = n - k
    return counter, ns


def expected_register_target(n: int, d_register: int = 32) -> list[float]:
    """Binary-bit encoding of 2^n − 1 into a d_register-length float vector.
    bits[i] = (((2^n − 1) >> i) & 1). The first n bits are 1.0, rest 0.0.
    For n ≥ d_register, all d_register bits are 1.0 (saturated)."""
    if n <= 0:
        return [0.0] * d_register
    return [1.0 if i < min(n, d_register) else 0.0 for i in range(d_register)]


def expected_register_trajectory(token_tensor, d_register=32, device="cpu"):
    """Build the expected register-write trajectory for a batch.

    Shape: (B, L, d_register). For every position in the OUTPUT SPAN
    (positions ≥ sep_pos+1, i.e. where the model is emitting answer
    bytes), r0 should hold expected_register_target(n).

    Outside the answer span the target is zero (don't supervise; the
    mask in the loss handles those positions).

    NOTE: we don't attempt to model "value remaining" — for now r0 is
    a constant across the whole answer span = the encoded answer value.
    The model's job is to load it once at SEP and read it digit-by-digit.
    """
    B, L = token_tensor.shape
    ns = parse_n_from_tokens(token_tensor)
    target = torch.zeros(B, L, d_register, device=device)
    for i, n in enumerate(ns):
        bits = expected_register_target(n, d_register)
        bits_t = torch.tensor(bits, device=device)
        # Fill the entire row; mask handles which positions are supervised.
        target[i] = bits_t.unsqueeze(0).expand(L, d_register)
    return target, ns
