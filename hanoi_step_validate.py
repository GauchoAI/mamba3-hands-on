"""hanoi_step_validate — autoregressive validation for the
multi-channel step-function Hanoi model.

The Python HanoiTool tracks per-disk pegs. After each emitted byte
the tool returns a K-vector of per-disk peg values. The model
receives this as multi-channel state feedback. AR decode loops:
emit byte -> tool updates -> next-step feedback.
"""
import argparse, sys, time
from pathlib import Path
import torch

sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, BOS, SEP, EOS
from hanoi_tool import HanoiTool, hanoi_moves, PEG_NAMES


def expected_trace(n):
    out = bytearray()
    for k, src, dst in hanoi_moves(n):
        out.extend(f"{k} {PEG_NAMES[src]} {PEG_NAMES[dst]}\n".encode("ascii"))
    return bytes(out)


def load_model(pt_path, device):
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    sd = ck["model"]
    has_lc = any(k.startswith("loop_counter.") for k in sd.keys())
    has_sf = any(k.startswith("state_feedback.") for k in sd.keys())
    K = cfg.get("K", 10)
    sf_vr = sd["state_feedback.value_emb.weight"].shape[0] if has_sf else 4
    model = ProgressiveModel(
        d_model=cfg["d_model"], d_state=16, expand=2, headdim=16,
        use_loop_counter=has_lc, lc_iteration_token=None,
        use_state_feedback=has_sf, sf_value_range=sf_vr,
    )
    for _ in range(cfg["n_kernel_layers"]):
        model.add_kernel_layer()
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, K


def autoregressive_decode(model, n, K, device, mode="pegs"):
    inp_bytes = list(f"HANOI {n}".encode("utf-8"))
    prefix = [BOS] + inp_bytes + [SEP]
    sep_pos = len(prefix) - 1
    total_bytes = sum(6 for _ in hanoi_moves(n))

    tool = HanoiTool(n)
    cur = list(prefix)
    emitted = []
    if mode == "pegs":
        fb_fn = lambda: tool.feedback_channels(K)
        n_ch = K
    elif mode == "pegs_npar":
        fb_fn = lambda: tool.feedback_channels_with_n(K)
        n_ch = K + 1
    else:  # full
        fb_fn = lambda: tool.feedback_channels_full(K)
        n_ch = K + 3
    fb_history = [fb_fn()]

    max_steps = total_bytes + 8
    for step in range(max_steps):
        L = len(cur)
        toks = torch.tensor([cur], dtype=torch.long, device=device)

        ch = torch.full((1, L, n_ch), 3, dtype=torch.long, device=device)
        for k, vec in enumerate(fb_history):
            p = sep_pos + k
            if 0 <= p < L:
                ch[0, p, :] = torch.tensor(vec, dtype=torch.long, device=device)

        # LoopCounter: counter at sep_pos+k = total_bytes - k
        cv = torch.full((1, L), -1, dtype=torch.long, device=device)
        for k in range(total_bytes + 1):
            p = sep_pos + k
            if 0 <= p < L:
                cv[0, p] = total_bytes - k

        with torch.no_grad():
            out = model(toks, counter_values=cv, state_channels=ch)
        logits = out["token_logits"] if isinstance(out, dict) else out
        nxt = int(logits[0, -1].argmax().item())
        if nxt == EOS:
            break
        emitted.append(nxt)
        cur.append(nxt)
        tool.step(nxt)
        fb_history.append(fb_fn())

    return bytes(b for b in emitted if b < 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="checkpoints/specialists/hanoi_step.pt")
    ap.add_argument("--ns", default="2,3,4,5,6,7,8,9,10")
    ap.add_argument("--state-mode", choices=["pegs", "pegs_npar", "full"], default="pegs")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    if not Path(args.pt).exists():
        raise SystemExit(f"checkpoint not found: {args.pt}")

    model, K = load_model(args.pt, args.device)
    print(f"Model: {args.pt}  K={K}")
    print(f"{'n':>3}  {'expected':>10}  {'got':>8}  {'prefix':>8}  {'%':>5}  match")
    print("-" * 50)

    fails = []
    for s in args.ns.split(","):
        n = int(s)
        expected = expected_trace(n)
        t0 = time.time()
        got = autoregressive_decode(model, n, K, args.device, mode=args.state_mode)
        dt = time.time() - t0
        match = (got == expected)
        prefix = 0
        for i in range(min(len(got), len(expected))):
            if got[i] != expected[i]:
                break
            prefix += 1
        pct = 100 * prefix / max(len(expected), 1)
        verdict = "✓" if match else "✗"
        print(f"{n:>3}  {len(expected):>10}  {len(got):>8}  {prefix:>8}  {pct:>5.1f}  {verdict}  ({dt:.0f}s)")
        if not match:
            fails.append((n, expected, got))

    print()
    if not fails:
        print("HANOI-STEP PASS — byte-for-byte match on all n.")
        return 0
    print(f"HANOI-STEP FAIL — {len(fails)} of {len(args.ns.split(','))}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
