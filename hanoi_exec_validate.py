"""hanoi_exec_validate — autoregressive validation of the Hanoi
register-execution model.

Unlike teacher-forced eval (which feeds ground-truth context at every
step), this runs the model autoregressively:
  - At step t, model emits a token + read_addr + write_addr + write_val
    via argmax of each head.
  - Apply the write to the register state.
  - Emit the token.
  - Compute the read_value from the (possibly just-written) register
    state and feed back as input to step t+1.

Then compare the emitted token sequence to Python's reference Hanoi
trace, byte-for-byte.

This is the real test: does the model EXECUTE the algorithm using
its registers, not memorize the trace?
"""
import argparse, sys
from pathlib import Path

import torch
sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, ByteTokenizer, BOS, SEP, EOS, PAD
from hanoi_exec_oracle import gen_exec_trace, hanoi_moves, encode_move


def load_model(pt_path: str, device: str):
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    sd = ck["model"]
    if not any(k.startswith("register_bank.") for k in sd.keys()):
        raise SystemExit(f"checkpoint {pt_path} has no RegisterBank")
    n_reg = sd["register_bank.read_addr_head.weight"].shape[0] - 1
    val_range = sd["register_bank.write_val_head.weight"].shape[0]
    model = ProgressiveModel(
        d_model=cfg["d_model"], d_state=16, expand=2, headdim=16,
        use_register_bank=True,
        reg_n_registers=n_reg, reg_value_range=val_range,
    )
    for _ in range(cfg["n_kernel_layers"]):
        model.add_kernel_layer()
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, n_reg, val_range


def autoregressive_decode(model, n: int, n_registers: int, value_range: int,
                          device: str, max_steps: int = None):
    """Step-by-step autoregressive decode for HANOI(n).

    Returns:
      emitted_bytes: the bytes the model produced as the trace
      register_history: list of register states after each step
      head_history: list of (read_addr, write_addr, write_val) the model chose
    """
    if max_steps is None:
        max_steps = (2 ** n - 1) * 8 + 16   # generous

    inp_bytes = list(f"HANOI {n}".encode("utf-8"))
    prefix = [BOS] + inp_bytes + [SEP]

    # Initial register state: all zeros
    registers = torch.zeros(1, n_registers, dtype=torch.long, device=device)
    last_read_value = torch.zeros(1, dtype=torch.long, device=device)

    # We use the FULL forward (not step-mode) since we want simplicity here.
    # Inefficient but correct. Step-mode could speed it up later.
    cur = list(prefix)
    emitted = []
    head_history = []

    NO_REG = n_registers
    for step in range(max_steps):
        # Build input: tokens are `cur`. read_input is per-position:
        # 0 for prefix, then last_read_value at the answer-span positions.
        L = len(cur)
        toks = torch.tensor([cur], dtype=torch.long, device=device)
        # Build read_input vector: at position p, read_input[p] is what
        # was read by the read_addr at position p-1.
        read_in = torch.zeros(1, L, dtype=torch.long, device=device)
        # Fill in read_input for the answer span:
        # head_history[k] was emitted at position sep_pos+k (predicting sep_pos+k+1).
        # The READ_VALUE that comes from head_history[k]['read_addr'] is the
        # input at position sep_pos+k+1.
        sep_pos = len(prefix) - 1
        # Need to reconstruct register values at each step too
        regs_running = torch.zeros(n_registers, dtype=torch.long, device=device)
        for k, (ra, wa, wv) in enumerate(head_history):
            p_curr = sep_pos + k
            p_next = p_curr + 1
            # Read happens at p_curr; the read result is what was at the
            # register BEFORE this step's write.
            if ra < NO_REG and p_next < L:
                read_in[0, p_next] = regs_running[ra].item()
            # Apply write
            if wa < NO_REG:
                regs_running[wa] = wv

        with torch.no_grad():
            out = model(toks, register_read_values=read_in)

        # Predict at last position
        last_idx = L - 1
        next_tok = int(out["token_logits"][0, last_idx].argmax().item())
        read_addr = int(out["read_logits"][0, last_idx].argmax().item())
        write_addr = int(out["write_logits"][0, last_idx].argmax().item())
        write_val = int(out["val_logits"][0, last_idx].argmax().item())

        if next_tok == EOS:
            break
        emitted.append(next_tok)
        cur.append(next_tok)
        head_history.append((read_addr, write_addr, write_val))

    return bytes(b for b in emitted if b < 256), head_history


def expected_trace(n: int) -> bytes:
    """Python reference: emit all moves as bytes."""
    moves = hanoi_moves(n)
    out = bytearray()
    for k, src, dst in moves:
        out.extend(encode_move(k, src, dst))
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="checkpoints/specialists/hanoi_exec.pt")
    ap.add_argument("--ns", default="2,3,4,5,6,7,8,10",
                    help="Comma-separated n values to test")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    if not Path(args.pt).exists():
        raise SystemExit(f"checkpoint not found: {args.pt}")

    model, n_reg, val_range = load_model(args.pt, args.device)
    print(f"Model: {args.pt}  registers={n_reg}  value_range={val_range}")
    print(f"Reference: HANOI(n) recursive moves -> bytes\n")
    print(f"{'n':>3}  {'expected_len':>12}  {'got_len':>8}  {'match':>6}  {'first_diff_byte':>15}")
    print("-" * 65)

    import time
    fails = []
    for s in args.ns.split(","):
        n = int(s)
        expected = expected_trace(n)
        t0 = time.time()
        got, _ = autoregressive_decode(model, n, n_reg, val_range, args.device)
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
        print(f"{n:>3}  {len(expected):>12}  {len(got):>8}  {verdict:>5}  {first_diff:>15}  ({dt:.1f}s)")
        if not match:
            fails.append((n, expected, got))

    print()
    if not fails:
        print("HANOI-EXEC PASS — all n match Python reference byte-for-byte.")
        return 0
    else:
        print(f"HANOI-EXEC FAIL — {len(fails)} of {len(args.ns.split(','))} mismatched.")
        # Show first few mismatched bytes for the first failing n
        n, exp, got = fails[0]
        print(f"\n  n={n} first 80 bytes:")
        print(f"    expected: {exp[:80]!r}")
        print(f"    got:      {got[:80]!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
