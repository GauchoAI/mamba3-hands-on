"""hanoi_tool_validate — autoregressive validation of the tool-use
Hanoi model.

The Python HanoiTool tracks state. After every byte the model emits,
the tool updates and produces the next feedback value. The model
sees this feedback as input embedding addition; it decides the next
byte. Loop until EOS (forced by LoopCounter at counter==0).

Compare emitted bytes byte-for-byte to Python's reference Hanoi.
"""
import argparse, sys, time
from pathlib import Path
import torch

sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, BOS, SEP, EOS
from hanoi_tool import HanoiTool, hanoi_moves, PEG_NAMES


def expected_trace(n: int) -> bytes:
    out = bytearray()
    for k, src, dst in hanoi_moves(n):
        out.extend(f"{k} {PEG_NAMES[src]} {PEG_NAMES[dst]}\n".encode("ascii"))
    return bytes(out)


def load_model(pt_path, device):
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    sd = ck["model"]
    has_lc = any(k.startswith("loop_counter.") for k in sd.keys())
    has_rb = any(k.startswith("register_bank.") for k in sd.keys())
    n_reg = sd["register_bank.read_addr_head.weight"].shape[0] - 1 if has_rb else 1
    val_range = cfg.get("value_range", 256)
    if has_rb and "register_bank.value_emb.weight" in sd:
        val_range = sd["register_bank.value_emb.weight"].shape[0]
    model = ProgressiveModel(
        d_model=cfg["d_model"], d_state=16, expand=2, headdim=16,
        use_register_bank=has_rb, reg_n_registers=n_reg,
        reg_value_range=val_range,
        use_loop_counter=has_lc, lc_iteration_token=None,
    )
    for _ in range(cfg["n_kernel_layers"]):
        model.add_kernel_layer()
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, val_range


def autoregressive_decode(model, n: int, value_range: int, device: str):
    """Step-by-step AR decode with tool-managed state feedback."""
    inp_bytes = list(f"HANOI {n}".encode("utf-8"))
    prefix = [BOS] + inp_bytes + [SEP]
    sep_pos = len(prefix) - 1
    total_bytes = sum(len(f"{k} {PEG_NAMES[s]} {PEG_NAMES[d]}\n".encode())
                      for k, s, d in hanoi_moves(n))

    tool = HanoiTool(n)
    cur = list(prefix)
    emitted = []
    state_fb_history = [0]  # initial feedback at sep_pos (state=0)

    max_steps = total_bytes + 8
    for step in range(max_steps):
        L = len(cur)
        toks = torch.tensor([cur], dtype=torch.long, device=device)

        # Build state-feedback input: at each answer-span position p,
        # feedback[p] is what the tool produced after the byte at p-1.
        fb = torch.zeros(1, L, dtype=torch.long, device=device)
        for k, v in enumerate(state_fb_history):
            p = sep_pos + k
            if 0 <= p < L:
                fb[0, p] = min(max(0, v), value_range - 1)

        # LoopCounter: counter at sep_pos+k = total_bytes - k.
        cv = torch.full((1, L), -1, dtype=torch.long, device=device)
        for k in range(total_bytes + 1):
            p = sep_pos + k
            if 0 <= p < L:
                cv[0, p] = total_bytes - k

        with torch.no_grad():
            out = model(toks, counter_values=cv, register_read_values=fb)
        logits = out["token_logits"] if isinstance(out, dict) else out
        nxt = int(logits[0, -1].argmax().item())
        if nxt == EOS:
            break
        emitted.append(nxt)
        cur.append(nxt)
        # Tool processes this newly emitted byte.
        tool.step(nxt)
        state_fb_history.append(tool.feedback_value())

    return bytes(b for b in emitted if b < 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="checkpoints/specialists/hanoi_tool.pt")
    ap.add_argument("--ns", default="2,3,4,5,6")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    if not Path(args.pt).exists():
        raise SystemExit(f"checkpoint not found: {args.pt}")

    model, val_range = load_model(args.pt, args.device)
    print(f"Model: {args.pt}  value_range={val_range}")
    print(f"Reference: HANOI(n) recursive moves -> bytes\n")
    print(f"{'n':>3}  {'expected':>10}  {'got_len':>8}  {'match':>6}  {'first_diff':>11}")
    print("-" * 55)

    fails = []
    for s in args.ns.split(","):
        n = int(s)
        expected = expected_trace(n)
        t0 = time.time()
        got = autoregressive_decode(model, n, val_range, args.device)
        dt = time.time() - t0
        match = (got == expected)
        first_diff = "—"
        if not match:
            for i in range(min(len(got), len(expected))):
                if got[i] != expected[i]:
                    first_diff = f"@{i}"
                    break
            else:
                first_diff = "len-only"
        verdict = "✓" if match else "✗"
        print(f"{n:>3}  {len(expected):>10}  {len(got):>8}  {verdict:>5}  {first_diff:>11}  ({dt:.1f}s)")
        if not match:
            fails.append((n, expected, got))

    print()
    if not fails:
        print("HANOI-TOOL PASS — all n match Python reference byte-for-byte.")
        return 0
    n0, exp0, got0 = fails[0]
    print(f"HANOI-TOOL FAIL — {len(fails)} of {len(args.ns.split(','))}")
    print(f"\n  n={n0} first 80 bytes:")
    print(f"    expected: {exp0[:80]!r}")
    print(f"    got:      {got0[:80]!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
