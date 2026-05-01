"""round_trip_test — bidirectional consistency probe.

If A→B and B→A specialists genuinely learned the *semantics* of the
mapping (vs surface patterns), then A→B→A should be identity for any
A in the input space. This is a much stronger signal than per-direction
accuracy — a model can be 90% accurate at A→B and 90% at B→A and still
have only 70% round-trip identity.

Usage:
  python3 round_trip_test.py \\
    --forward checkpoints/specialists/bool_expr_to_truth_table.pt \\
    --reverse checkpoints/specialists/truth_table_to_bool_expr.pt \\
    --start-domain expr     # round-trips: expr → tt → expr
                             # (or 'tt' for: tt → expr → tt)

For the boolean tier-1 pair specifically, expr→tt→expr collapses to
canonical: many expressions map to the same truth table, then the
reverse step produces the *canonical* smallest expression. So the
"round-trip" identity is "did the forward step preserve the truth
table, and did the reverse step produce the canonical form?"

The cleaner round-trip is tt→expr→tt: the reverse direction picks
ONE canonical expression for each tt, and the forward direction
should evaluate it back to the original tt.
"""
import argparse, json, sys, time, random
from pathlib import Path

import torch

sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, ByteTokenizer, PAD


def load_specialist(pt_path, device):
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    sd = ck.get("model", {})
    if any(torch.isnan(v).any().item() if v.is_floating_point() else False for v in sd.values()):
        raise SystemExit(f"specialist has NaN weights: {pt_path}")
    model = ProgressiveModel(
        d_model=cfg.get("d_model", 64),
        d_state=cfg.get("d_state", 16),
        expand=2,
        headdim=cfg.get("headdim", 16),
    ).to(device)
    for _ in range(cfg.get("n_kernel_layers", 1)):
        model.add_kernel_layer()
    model.load_state_dict(sd)
    model.eval()
    return model, cfg, ck.get("accuracy", 0.0)


def autoregressive_decode(model, prefix_tokens, sep_pos, max_new, device, eos=257):
    cur = list(prefix_tokens[:sep_pos + 1])
    out = []
    for _ in range(max_new):
        x = torch.tensor([cur], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        nxt = int(logits[0, -1].argmax().item())
        if nxt == eos:
            break
        out.append(nxt)
        cur.append(nxt)
    return bytes(b for b in out if 32 <= b < 127).decode("ascii", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", default="checkpoints/specialists/bool_expr_to_truth_table.pt")
    ap.add_argument("--reverse", default="checkpoints/specialists/truth_table_to_bool_expr.pt")
    ap.add_argument("--start-domain", choices=["tt", "expr"], default="tt")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    fwd, _, fwd_acc = load_specialist(args.forward, args.device)
    rev, _, rev_acc = load_specialist(args.reverse, args.device)
    print(f"Forward (A→B):  {args.forward}  trained_acc={fwd_acc:.0%}")
    print(f"Reverse (B→A):  {args.reverse}  trained_acc={rev_acc:.0%}")
    print(f"Round-trip:     {args.start_domain} → ? → {args.start_domain}")
    print()

    tok = ByteTokenizer()

    # Domain samples
    if args.start_domain == "tt":
        from formal_language import TT_TO_EXPR
        domain = list(TT_TO_EXPR.keys())  # all 16
    else:
        from formal_language import _build_pool, _EXPRESSION_POOL
        _build_pool()
        domain = list(_EXPRESSION_POOL)

    # Statistics
    rt_correct = 0           # round-trip identity
    fwd_step_consistent = 0  # the intermediate B was a valid B
    n_total = 0

    samples = []
    for _ in range(args.n_trials):
        a = random.choice(domain)
        # Step 1: A → B via the appropriate specialist
        if args.start_domain == "tt":
            ex_step1 = {"type": "_", "input": f"TABEXPR {a}", "output": ""}
            spec_step1 = rev   # tt→expr
            spec_step2 = fwd   # expr→tt (forward verification)
            input_marker_step2 = "BOOLTAB"
        else:
            ex_step1 = {"type": "_", "input": f"BOOLTAB {a}", "output": ""}
            spec_step1 = fwd   # expr→tt
            spec_step2 = rev   # tt→expr
            input_marker_step2 = "TABEXPR"

        toks1, sep1 = tok.encode_curriculum(ex_step1)
        b_pred = autoregressive_decode(spec_step1, toks1, sep1, max_new=24, device=args.device)
        # Step 2: B → A via the other specialist
        ex_step2 = {"type": "_", "input": f"{input_marker_step2} {b_pred}", "output": ""}
        toks2, sep2 = tok.encode_curriculum(ex_step2)
        a_pred = autoregressive_decode(spec_step2, toks2, sep2, max_new=24, device=args.device)

        # For tt→expr→tt: round-trip is correct if a_pred == a (the tt is recovered)
        # For expr→tt→expr: round-trip identity is canonical-form: a_pred should be
        # the canonical expression for the tt corresponding to a's evaluation.
        if args.start_domain == "tt":
            is_rt = (a_pred == a)
        else:
            from formal_language import _truth_table, TT_TO_EXPR
            tt_of_a = _truth_table(a)
            canonical = TT_TO_EXPR[tt_of_a]
            is_rt = (a_pred == canonical)

        rt_correct += int(is_rt)
        n_total += 1

        if len(samples) < 6:
            samples.append({"a": a, "b_pred": b_pred, "a_pred": a_pred,
                            "round_trip_ok": is_rt})

    print(f"Round-trip accuracy: {rt_correct}/{n_total} = {rt_correct/n_total:.1%}")
    print(f"\nSamples:")
    for s in samples:
        ok = "✓" if s["round_trip_ok"] else "✗"
        print(f"  {ok}  {args.start_domain}={s['a']!r}  →  ?={s['b_pred']!r}  →  "
              f"{args.start_domain}={s['a_pred']!r}")

    Path("/tmp/round_trip.json").write_text(json.dumps({
        "forward": args.forward, "reverse": args.reverse,
        "start_domain": args.start_domain, "n_trials": n_total,
        "round_trip_correct": rt_correct,
        "round_trip_acc": rt_correct / n_total,
        "samples": samples,
    }, indent=2))


if __name__ == "__main__":
    main()
