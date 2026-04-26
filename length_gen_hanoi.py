"""length_gen_hanoi — fixed-model length-generalization experiment.

Loads checkpoints/specialists/tower_of_hanoi.pt (trained on n_disks ∈ [1,8]
per the curriculum) and evaluates it WITHOUT any further training on
n_disks values from 1 up to N_MAX. Each n gets ACC_TRIALS samples.

Hypothesis:
- If the model learned the algorithm (number of moves = 2^n − 1), accuracy
  stays high beyond n=8.
- If it memorized 8 input→output pairs, accuracy crashes the moment we
  query n=9.

The model size is FIXED (it's the saved .pt; no GA, no resizing). What's
changing is the test-time difficulty.
"""
import argparse, json, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, ByteTokenizer, PAD, VOCAB_SIZE


N_MAX_DEFAULT = 25
ACC_TRIALS_DEFAULT = 100


def autoregressive_predict(model, tokens, sep_pos, max_new=8, device="cpu"):
    """Greedy decode after the separator until EOS or max_new tokens.

    Returns the decoded byte string (the answer). Stops at byte 257 (EOS)
    or after max_new bytes — whichever first. The model is run with the
    full prefix [BOS, input bytes, SEP] and we keep extending one byte
    at a time.
    """
    EOS = 257
    cur = list(tokens[:sep_pos + 1])  # up to and including SEP
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
    # Decode the answer bytes back to a string. Filter to printable ASCII
    # so a junk byte doesn't break str().
    return bytes(b for b in out_bytes if 32 <= b < 127).decode("ascii", errors="replace")


def make_hanoi_example(n):
    """Mirror gen_tower_of_hanoi but with the n forced to a specific value."""
    moves = 2 ** n - 1
    return {"type": "tower_of_hanoi", "input": f"HANOI {n}", "output": str(moves)}


def evaluate_n(model, tok, n, n_trials, device):
    """Run the model on `n_trials` samples of HANOI(n). Returns (acc, sample_correct, sample_predicted)."""
    correct = 0
    sample_pred = None
    sample_target = None
    for _ in range(n_trials):
        ex = make_hanoi_example(n)
        toks, sep = tok.encode_curriculum(ex)
        # Predict the answer string
        pred = autoregressive_predict(model, toks, sep, max_new=12, device=device)
        target = str(2 ** n - 1)
        if pred == target:
            correct += 1
        if sample_pred is None:
            sample_pred = pred
            sample_target = target
    return correct / n_trials, sample_pred, sample_target


def load_specialist(pt_path, device):
    ck = torch.load(str(pt_path), map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    model = ProgressiveModel(
        d_model=cfg.get("d_model", 64),
        d_state=cfg.get("d_state", 16),
        expand=2,
        headdim=cfg.get("headdim", 16),
    ).to(device)
    for _ in range(cfg.get("n_kernel_layers", 1)):
        model.add_kernel_layer()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="checkpoints/specialists/tower_of_hanoi.pt")
    ap.add_argument("--n-max", type=int, default=N_MAX_DEFAULT)
    ap.add_argument("--trials", type=int, default=ACC_TRIALS_DEFAULT)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=12345)
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
    print(f"Curriculum trained up to n_disks=8 (per problems/tower_of_hanoi/problem.yaml)\n")

    tok = ByteTokenizer()
    rows = []
    print(f"{'n':>3}  {'2^n-1':>10}  {'acc':>7}   {'sample (pred → target)':<30}  {'cycle':>6}s")
    print("-" * 70)
    t_start = time.time()
    for n in range(1, args.n_max + 1):
        t0 = time.time()
        acc, sp, st = evaluate_n(model, tok, n, args.trials, args.device)
        dt = time.time() - t0
        rows.append({"n": n, "moves": 2 ** n - 1, "acc": acc,
                     "sample_pred": sp, "sample_target": st,
                     "wall_s": round(dt, 2)})
        verdict = "✓" if acc >= 0.95 else ("≈" if acc >= 0.5 else "✗")
        print(f"{n:>3}  {2**n - 1:>10}  {acc:>6.1%} {verdict}  "
              f"{sp!r:>14} → {st!r:<14}  {dt:>5.1f}")

    total = time.time() - t_start
    out = Path("/tmp/length_gen_hanoi.json")
    out.write_text(json.dumps({
        "pt": args.pt, "device": args.device, "trials": args.trials,
        "n_max": args.n_max, "rows": rows, "total_wall_s": round(total, 1),
    }, indent=2))
    print(f"\nTotal {total:.1f}s. Saved {out}")

    # Find the cliff
    cliffs = [r for r in rows if r["acc"] < 0.5]
    in_dist = [r for r in rows if r["n"] <= 8 and r["acc"] >= 0.95]
    print(f"In-distribution (n≤8) at ≥95%: {len(in_dist)}/8")
    if cliffs:
        first = cliffs[0]["n"]
        print(f"Generalization cliff: first n with acc<50% is n={first}")
    else:
        print(f"No cliff in [1, {args.n_max}] — model generalized across the entire range")


if __name__ == "__main__":
    main()
