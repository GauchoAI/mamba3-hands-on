"""length_gen_general — apply the Hanoi cliff-test methodology to any task
with a tunable difficulty knob.

For each task we test, we sweep the difficulty knob across a range that
includes both the trained-on values and values past them. The pattern
of failures at the boundary tells us whether the model memorized or
computed.

Tasks:
  parity      — knob: n_bits           (trained 1..6, test 1..16)
  addition    — knob: max_digits       (trained 1..3, test 1..8)
  count_above — knob: n_values         (trained ~3..6, test 3..15)
"""
import argparse, json, random, sys, time
from pathlib import Path

import torch

sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, ByteTokenizer, PAD


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


def autoregressive_predict(model, tokens, sep_pos, max_new, device):
    EOS = 257
    cur = list(tokens[:sep_pos + 1])
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


# ── Per-task example builders + their difficulty axis ─────────────────
#
# Each entry calls the REAL registry generator with kwargs that pin the
# difficulty knob, so the example format matches what the specialist
# was trained on byte-for-byte.

def _registry_gen(task):
    sys.path.insert(0, ".")
    from registry.problem_registry import ProblemRegistry
    reg = ProblemRegistry()
    reg.discover(["problems"])
    return reg.get_generator(task)


_PARITY = None
_ADDITION = None
_COUNT_ABOVE = None


def parity_example(n):
    global _PARITY
    if _PARITY is None: _PARITY = _registry_gen("parity")
    return _PARITY(min_len=n, max_len=n)


def addition_example(n):
    global _ADDITION
    if _ADDITION is None: _ADDITION = _registry_gen("addition")
    return _ADDITION(n_digits=n)


def count_above_example(n):
    global _COUNT_ABOVE
    if _COUNT_ABOVE is None: _COUNT_ABOVE = _registry_gen("count_above_threshold")
    return _COUNT_ABOVE(min_len=n, max_len=n)


TASKS = {
    "parity":      {"fn": parity_example,      "knob": "n_bits",     "range": (1, 16), "max_new": 4},
    "addition":    {"fn": addition_example,    "knob": "n_digits",   "range": (1, 8),  "max_new": 16},
    "count_above_threshold": {"fn": count_above_example, "knob": "n_values",   "range": (3, 15), "max_new": 4},
}


def evaluate_at(model, tok, task_fn, knob_value, n_trials, max_new, device):
    correct = 0
    sample_pred = None
    sample_target = None
    for _ in range(n_trials):
        ex = task_fn(knob_value)
        ex["type"] = "_"
        toks, sep = tok.encode_curriculum(ex)
        pred = autoregressive_predict(model, toks, sep, max_new, device)
        target = ex["output"]
        if pred == target:
            correct += 1
        if sample_pred is None:
            sample_pred = pred
            sample_target = target
    return correct / n_trials, sample_pred, sample_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS.keys()))
    ap.add_argument("--pt", default=None)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    pt_path = args.pt or f"checkpoints/specialists/{args.task}.pt"
    if not Path(pt_path).exists():
        raise SystemExit(f"checkpoint not found: {pt_path}")

    spec = TASKS[args.task]
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, ck = load_specialist(pt_path, args.device)
    cfg = ck.get("config", {})
    print(f"Task:    {args.task}")
    print(f"Knob:    {spec['knob']}")
    print(f"Model:   d={cfg.get('d_model')}, L={cfg.get('n_kernel_layers')}, "
          f"trained_acc={ck.get('accuracy', 0):.0%}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print()

    tok = ByteTokenizer()
    rows = []
    print(f"{spec['knob']:>8}  {'acc':>7}   {'sample (pred → target)':<30}  {'wall':>5}s")
    print("-" * 65)
    for k in range(spec["range"][0], spec["range"][1] + 1):
        t0 = time.time()
        acc, sp, st = evaluate_at(model, tok, spec["fn"], k,
                                  args.trials, spec["max_new"], args.device)
        dt = time.time() - t0
        rows.append({"k": k, "acc": acc, "sample_pred": sp,
                     "sample_target": st, "wall_s": round(dt, 2)})
        verdict = "✓" if acc >= 0.95 else ("≈" if acc >= 0.5 else "✗")
        print(f"{k:>8}  {acc:>6.1%} {verdict}  {sp!r:>14} → {st!r:<14}  {dt:>4.1f}")

    out = Path(f"/tmp/length_gen_{args.task}.json")
    out.write_text(json.dumps({
        "task": args.task, "pt": pt_path, "trials": args.trials,
        "rows": rows,
    }, indent=2))

    cliffs = [r for r in rows if r["acc"] < 0.5]
    print()
    if cliffs:
        print(f"Cliff: first k with acc<50% is k={cliffs[0]['k']}")
    else:
        print(f"No cliff in tested range — generalized across {spec['range']}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
