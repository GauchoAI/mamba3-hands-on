"""length_gen_hanoi_binary — extrapolation test for the binary-output
Hanoi diagnostic.

Trained-on-decimal Hanoi failed to extrapolate past the curriculum's
n boundary; trajectory distillation stabilized training but didn't
fix the extrapolation. The hypothesis under test here: was the
failure the binary→decimal output head, or the recurrence itself?

The binary task is trivial in shape: input HANOIBIN n → output "1"*n.
If the model learns "count to n and emit n ones," it should
extrapolate. If it can't even do that, the recurrence isn't being
generalised.

Train n ≤ 20, evaluate n ∈ [1, 100].
"""
import argparse, json, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, ByteTokenizer, PAD


def autoregressive_predict(model, tokens, sep_pos, max_new, device):
    EOS = 257
    cur = list(tokens[: sep_pos + 1])
    out_bytes = []
    for _ in range(max_new):
        x = torch.tensor([cur], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        next_tok = int(logits[0, -1].argmax().item())
        if next_tok == EOS:
            break
        out_bytes.append(next_tok)
        cur.append(next_tok)
    return bytes(b for b in out_bytes if 32 <= b < 127).decode("ascii", errors="replace")


def make_example(n, bidir=False):
    inp = f"HANOIBIN {n}"
    if bidir:
        inp = inp + " " + inp[::-1]
    return {"type": "tower_of_hanoi_binary", "input": inp,
            "output": "1" * n}


def evaluate_n(model, tok, n, max_new, device, bidir=False):
    ex = make_example(n, bidir=bidir)
    toks, sep = tok.encode_curriculum(ex)
    pred = autoregressive_predict(model, toks, sep, max_new=max_new, device=device)
    target = "1" * n
    correct = (pred == target)
    # Diagnostic: length of correct-prefix and total length, so we can see
    # *how* it fails (right shape, wrong length? wrong content?)
    correct_prefix = 0
    for c in pred:
        if c == "1":
            correct_prefix += 1
        else:
            break
    return correct, pred, target, correct_prefix


def load_specialist(pt_path, device):
    ck = torch.load(str(pt_path), map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    sd = ck.get("model", {})
    has_history_attn = any(k.startswith("history_attn.") for k in sd.keys())
    history_d_attn = (sd["history_attn.q_proj.weight"].shape[0]
                      if has_history_attn else 32)
    has_registers = any(k.startswith("registers.") for k in sd.keys())
    n_registers = (sd["registers.read_query.weight"].shape[0]
                   if has_registers else 8)
    d_register = (sd["registers.read_proj.weight"].shape[1]
                  if has_registers else 32)
    model = ProgressiveModel(
        d_model=cfg.get("d_model", 64),
        d_state=cfg.get("d_state", 16),
        expand=2,
        headdim=cfg.get("headdim", 16),
        use_history_attn=has_history_attn,
        history_d_attn=history_d_attn,
        use_explicit_registers=has_registers,
        n_registers=n_registers,
        d_register=d_register,
    ).to(device)
    for _ in range(cfg.get("n_kernel_layers", 1)):
        model.add_kernel_layer()
    model.load_state_dict(sd)
    model.eval()
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="checkpoints/specialists/tower_of_hanoi_binary.pt")
    ap.add_argument("--n-max", type=int, default=100)
    ap.add_argument("--trained-up-to", type=int, default=20,
                    help="In-distribution boundary for the report.")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--bidir", action="store_true",
                    help="Apply the same input + ' ' + reversed(input) "
                         "rewrite the trainer used. Required if the "
                         "checkpoint was trained with --bidir-input.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not Path(args.pt).exists():
        raise SystemExit(f"checkpoint not found: {args.pt}")

    print(f"Loading {args.pt} on {args.device} ...")
    model, ck = load_specialist(args.pt, args.device)
    cfg = ck.get("config", {})
    print(f"Model: d={cfg.get('d_model')}, L={cfg.get('n_kernel_layers')}, "
          f"trained_acc={ck.get('accuracy', 0):.0%}, params="
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"Trained up to n_disks={args.trained_up_to}\n")

    tok = ByteTokenizer()
    rows = []
    print(f"{'n':>4}  {'expected len':>12}  {'pred_len':>8}  {'prefix_ok':>9}  {'verdict':>7}  pred (truncated)")
    print("-" * 80)
    t_start = time.time()
    # max_new generously bigger than n_max so EOS prediction can fail openly
    max_new = max(args.n_max + 20, 64)
    for n in range(1, args.n_max + 1):
        correct, pred, target, prefix = evaluate_n(model, tok, n, max_new, args.device, bidir=args.bidir)
        rows.append({"n": n, "correct": correct, "pred_len": len(pred),
                     "prefix_ones": prefix, "pred_sample": pred[:50]})
        in_dist = n <= args.trained_up_to
        verdict = "✓" if correct else ("≈" if prefix >= n else "✗")
        marker = "  " if in_dist else "* "  # mark out-of-distribution
        pred_show = pred if len(pred) <= 50 else pred[:47] + "..."
        print(f"{marker}{n:>2}  {n:>12}  {len(pred):>8}  {prefix:>9}  {verdict:>7}  {pred_show!r}")

    total = time.time() - t_start
    out = Path("/tmp/length_gen_hanoi_binary.json")
    out.write_text(json.dumps({
        "pt": args.pt, "device": args.device,
        "n_max": args.n_max, "trained_up_to": args.trained_up_to,
        "rows": rows, "total_wall_s": round(total, 1),
    }, indent=2))
    print(f"\nTotal {total:.1f}s. Saved {out}")

    in_dist_correct = sum(1 for r in rows if r["n"] <= args.trained_up_to and r["correct"])
    out_dist_correct = sum(1 for r in rows if r["n"] > args.trained_up_to and r["correct"])
    in_dist_total = sum(1 for r in rows if r["n"] <= args.trained_up_to)
    out_dist_total = sum(1 for r in rows if r["n"] > args.trained_up_to)
    print(f"In-distribution  n∈[1,{args.trained_up_to}]:  {in_dist_correct}/{in_dist_total}")
    print(f"Out-of-distribution n∈[{args.trained_up_to+1},{args.n_max}]: {out_dist_correct}/{out_dist_total}")


if __name__ == "__main__":
    main()
