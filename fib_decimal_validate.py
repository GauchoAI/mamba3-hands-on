"""fib_decimal_validate — byte-for-byte cross-check that the trained
FIBD model produces exactly str(F(n)) for every n in the supported
range, and obeys the per-position iter_token oracle.

Three suites, mirroring fish_validate (HANOIBIN) and fib_validate
(FIB-unary):

  1. Length-gen: every n in [1, max_n], compare model output to
     str(F(n)) byte-for-byte.

  2. Counterfactual: feed input "FIBD n_input" but oracle the trajectory
     for str(F(n_target)). The model must follow the per-position
     iter_token, not the input.

  3. Edge cases: n=0 (F=0, single-digit 0), n=1, near-OOD, far-OOD.

Step-decoder for O(L) per query.
"""
import argparse, sys
from pathlib import Path

import torch
sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, ByteTokenizer

EOS = 257
SEP_TOKEN = 258
BOS_TOKEN = 256


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def python_reference(n: int) -> str:
    return str(fib(n))


def load_model(pt_path: str, device: str):
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = ck["config"]; sd = ck["model"]
    has_lc = any(k.startswith("loop_counter.") for k in sd.keys())
    if not has_lc:
        raise SystemExit(f"checkpoint {pt_path} has no LoopCounter")
    lc_max = sd["loop_counter.c_emb.weight"].shape[0] - 2
    model = ProgressiveModel(
        d_model=cfg["d_model"], d_state=cfg["d_state"],
        expand=2, headdim=cfg["headdim"],
        use_loop_counter=True, loop_counter_max=lc_max,
    )
    for _ in range(cfg["n_kernel_layers"]):
        model.add_kernel_layer()
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, lc_max


def step_decode_fibd(model, n_input: int, target_digits: str,
                     max_count: int, device: str):
    """Step-mode autoregressive decode for FIBD with per-position
    iter_token. The oracle exposes the answer DIGITS to the model
    (counter trajectory + iter_token = digit at each position).
    """
    sentinel = max_count + 1
    D = len(target_digits)
    inp = list(f"FIBD {n_input}".encode("utf-8"))
    prefix = [BOS_TOKEN] + inp + [SEP_TOKEN]
    sep_pos = len(prefix) - 1
    cur_len = len(prefix)
    states = model.init_decode_state(1)

    def cv_at(p):
        k = p - sep_pos
        return min(D - k, max_count) if 0 <= k <= D else sentinel

    def itok_at(p):
        k = p - sep_pos
        return ord(target_digits[k]) if 0 <= k < D else 0

    last_logits = None
    # Prime: step through the prefix
    for i, tok_id in enumerate(prefix):
        c_t = torch.tensor([[cv_at(i)]], dtype=torch.long, device=device)
        it_t = torch.tensor([[itok_at(i)]], dtype=torch.long, device=device)
        t = torch.tensor([[tok_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            last_logits, states = model.forward_step(
                t, counter_value=c_t, states=states,
                iter_token_per_pos=it_t,
            )

    nxt = int(last_logits[0, -1].argmax().item())
    out = []
    max_steps = D + 8
    for _ in range(max_steps):
        if nxt == EOS:
            break
        out.append(nxt)
        cur_len += 1
        c_t = torch.tensor([[cv_at(cur_len - 1)]], dtype=torch.long, device=device)
        it_t = torch.tensor([[itok_at(cur_len - 1)]], dtype=torch.long, device=device)
        t = torch.tensor([[nxt]], dtype=torch.long, device=device)
        with torch.no_grad():
            last_logits, states = model.forward_step(
                t, counter_value=c_t, states=states,
                iter_token_per_pos=it_t,
            )
        nxt = int(last_logits[0, -1].argmax().item())
    return bytes(b for b in out if b < 256)


def suite_1_length_gen(model, max_n, max_count, device):
    fails = []
    for n in range(1, max_n + 1):
        target = python_reference(n)
        actual = step_decode_fibd(model, n, target, max_count, device)
        expected = target.encode("ascii")
        if actual != expected:
            fails.append((n, expected, actual))
    return fails


def suite_2_counterfactual(model, max_count, device):
    """Feed input 'FIBD n_input' but oracle for F(n_target)'s digits."""
    pairs = [(5, 1), (5, 10), (10, 5), (1, 20),
             (20, 1), (3, 30), (30, 3),
             (15, 35), (35, 15)]
    fails = []
    for ni, nt in pairs:
        target = python_reference(nt)
        actual = step_decode_fibd(model, ni, target, max_count, device)
        expected = target.encode("ascii")
        if actual != expected:
            fails.append((ni, nt, expected, actual))
    return fails


def suite_3_edges(model, max_count, device):
    cases = [0, 1, 2, 7, 10, 20, 30, 40]
    fails = []
    for n in cases:
        target = python_reference(n)
        actual = step_decode_fibd(model, n, target, max_count, device)
        expected = target.encode("ascii")
        if actual != expected:
            fails.append((n, expected, actual))
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="checkpoints/specialists/fib_decimal.pt")
    ap.add_argument("--max-n", type=int, default=40)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    if not Path(args.pt).exists():
        raise SystemExit(f"checkpoint not found: {args.pt}")

    import time
    t_total = time.time()
    model, lc_max = load_model(args.pt, args.device)
    print(f"Model: {args.pt}")
    print(f"Counter table max: {lc_max}")
    print(f"Reference: FIBD n -> str(F(n))")
    print()

    t = time.time()
    fails1 = suite_1_length_gen(model, args.max_n, lc_max, args.device)
    print(f"Suite 1: length-gen sweep n=1..{args.max_n} ({time.time()-t:.1f}s)")
    print(f"  {args.max_n - len(fails1)}/{args.max_n} match")
    for n, exp, act in fails1[:5]:
        print(f"  ✗ n={n}: expected {exp!r} got {act!r}")
    print()

    t = time.time()
    fails2 = suite_2_counterfactual(model, lc_max, args.device)
    n_total_2 = 9
    print(f"Suite 2: counterfactual ({time.time()-t:.1f}s)")
    print(f"  {n_total_2 - len(fails2)}/{n_total_2} pass")
    for ni, nt, exp, act in fails2[:5]:
        print(f"  ✗ input={ni} target=F({nt}): expected {exp!r} got {act!r}")
    print()

    t = time.time()
    fails3 = suite_3_edges(model, lc_max, args.device)
    n_total_3 = 8
    print(f"Suite 3: edge cases ({time.time()-t:.1f}s)")
    print(f"  {n_total_3 - len(fails3)}/{n_total_3} pass")
    for n, exp, act in fails3:
        print(f"  ✗ n={n}: expected {exp!r} got {act!r}")
    print()

    total_fails = len(fails1) + len(fails2) + len(fails3)
    print(f"Total wall: {time.time()-t_total:.1f}s")
    if total_fails == 0:
        print("FIBD FISH PASS — model matches str(F(n)) byte-for-byte.")
        return 0
    else:
        print(f"FIBD FISH FAIL — {total_fails} mismatches.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
