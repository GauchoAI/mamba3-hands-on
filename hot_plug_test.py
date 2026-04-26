"""hot_plug_test — extend a deployed router with a new specialist at
inference-time, no retraining of the base.

Validates the strongest property the ecology architecture claims: a
router trained without a particular specialist can have that
specialist plugged in *post-hoc*, with only the new bridge's
parameters (W_recv + gate + scale; ~1.1k floats) optimized over a
brief fine-tune. The router's base SSM stays frozen.

Procedure:
  1. Load a saved router from --start-from <ckpt> (built via
     `train_router.py --save-to`). Its base + any pre-existing
     bridges are frozen.
  2. Append a NEW AttendBridge to a NEW specialist loaded from
     --new-specialist <ckpt>.
  3. Optimize ONLY the new bridge's parameters for --steps steps on
     --task. Report accuracy at intervals.

Result interpretation:
  - If acc_after > acc_before, the new specialist was successfully
    integrated — capability grows at runtime without disturbing the
    rest.
  - If acc stays flat, the specialist's expertise wasn't useful for
    this task; the gate should converge toward closed.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from progressive_model import ByteTokenizer, PAD
from synapse import AttendBridge, RouterModel, load_specialist, PlaceholderSpecialist


def build_batch(gen, tok, batch_size, device):
    examples = [tok.encode_curriculum(gen()) for _ in range(batch_size)]
    max_len = max(len(t) for t, _ in examples)
    tokens = torch.full((batch_size, max_len), PAD, dtype=torch.long, device=device)
    seps = []
    for i, (toks, sep) in enumerate(examples):
        tokens[i, :len(toks)] = torch.tensor(toks, device=device)
        seps.append(sep)
    return tokens, torch.tensor(seps, device=device, dtype=torch.long)


def loss_and_acc(router, tokens, seps, device):
    logits = router(tokens)
    B, L, V = logits.shape
    pos = torch.arange(L, device=device).unsqueeze(0)
    sep_t = seps.unsqueeze(1)
    mask = ((pos >= sep_t) & (pos < L - 1)).float()
    pad_mask = (tokens != PAD).float()
    pred_mask = mask[:, :L - 1] * pad_mask[:, 1:]
    if pred_mask.sum() < 1:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    logits_flat = logits[:, :L - 1].reshape(-1, V)
    targets_flat = tokens[:, 1:].reshape(-1)
    mask_flat = pred_mask.reshape(-1)
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    loss = (ce * mask_flat).sum() / (mask_flat.sum() + 1e-8)
    with torch.no_grad():
        pred = logits[:, :L - 1].argmax(dim=-1)
        actual = tokens[:, 1:]
        per_pos_correct = (pred == actual).float() * pred_mask
        n_supervised = pred_mask.sum(dim=-1)
        n_correct = per_pos_correct.sum(dim=-1)
        example_correct = (n_correct == n_supervised) & (n_supervised > 0)
        acc = float(example_correct.float().mean())
    return loss, acc


def evaluate(router, gen, tok, n_examples, device):
    eval_acc = 0.0
    n_batches = max(1, n_examples // 256)
    for _ in range(n_batches):
        eval_tokens, eval_seps = build_batch(gen, tok, 256, device)
        with torch.no_grad():
            _, acc = loss_and_acc(router, eval_tokens, eval_seps, device)
        eval_acc += acc / n_batches
    return eval_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-from", required=True,
                    help="Saved router .pt to start from (build via "
                         "train_router.py --save-to).")
    ap.add_argument("--new-specialist", required=True,
                    help="New specialist .pt to plug in via a fresh bridge.")
    ap.add_argument("--task", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-lr", type=float, default=0.0,
                    help="If >0, ALSO unfreeze the base (and any pre-existing "
                         "bridges) and fine-tune them at this lr while the "
                         "new bridge trains at --lr. Realistic 'ecology grows' "
                         "regime: peer joins, system briefly re-equilibrates.")
    ap.add_argument("--swap-slot", type=int, default=None,
                    help="If set, REPLACE the specialist at this slot index "
                         "with the new one (instead of appending a new slot). "
                         "Designed for placeholder→real swaps: the saved "
                         "router was trained with a reserved synapse slot at "
                         "this index, and we now fill it with a real "
                         "specialist. The bridge for this slot is also "
                         "re-initialized.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    # 1. Load the saved router. Use specialist_spec if present (includes
    # placeholders); fall back to legacy specialist_paths (real only).
    ck = torch.load(args.start_from, map_location=args.device, weights_only=False)
    spec_list = ck.get("specialist_spec")
    if spec_list:
        inner_specialists = []
        for s in spec_list:
            if s["type"] == "real":
                inner_specialists.append(load_specialist(s["path"], device=args.device))
            elif s["type"] == "placeholder":
                inner_specialists.append(
                    PlaceholderSpecialist(d_model=s["d_model"]).to(args.device))
            else:
                raise ValueError(f"unknown spec type: {s['type']}")
    else:
        inner_specialists = [
            load_specialist(p, device=args.device) for p in ck.get("specialist_paths", [])
        ]
    router = RouterModel(
        router_d_model=ck["router_d_model"],
        router_d_state=ck["router_d_state"],
        router_headdim=ck["router_headdim"],
        router_n_layers=ck["router_n_layers"],
        specialists=inner_specialists,
        bridge_kind=ck.get("bridge_kind", "attend"),
    ).to(args.device)
    router.load_state_dict(ck["router_state_dict"])

    # 2. Freeze EVERYTHING about the loaded router — base + existing bridges.
    for p in router.parameters():
        p.requires_grad = False

    # 3. Either SWAP a placeholder slot or APPEND a new slot.
    new_sp = load_specialist(args.new_specialist, device=args.device)
    if args.swap_slot is not None:
        if args.swap_slot >= len(router.specialists):
            raise SystemExit(f"--swap-slot {args.swap_slot} out of range "
                             f"(router has {len(router.specialists)} slots)")
        old = router.specialists[args.swap_slot]
        if new_sp.d_model != router.bridges[args.swap_slot].recv.in_features:
            raise SystemExit(f"d_specialist mismatch: existing slot expects "
                             f"d={router.bridges[args.swap_slot].recv.in_features}, "
                             f"new specialist has d={new_sp.d_model}")
        router.specialists[args.swap_slot] = new_sp
        # Re-initialize THIS slot's bridge — the prior bridge was shaped
        # for placeholder zeros; the new specialist's hidden state has
        # different statistics. Fresh open-init bridge gives the swap a
        # clean differentiable foothold.
        new_bridge = AttendBridge(d_router=router.base.d_model,
                                  d_specialist=new_sp.d_model,
                                  init_open=True).to(args.device)
        router.bridges[args.swap_slot] = new_bridge
        print(f"SWAP: slot {args.swap_slot}: "
              f"{type(old).__name__} → {args.new_specialist}")
    else:
        router.specialists.append(new_sp)
        new_bridge = AttendBridge(d_router=router.base.d_model,
                                  d_specialist=new_sp.d_model,
                                  init_open=True).to(args.device)
        router.bridges.append(new_bridge)
    # Only the new bridge trains.
    for p in new_bridge.parameters():
        p.requires_grad = True

    # 4. Set up generators + optimizer.
    from registry.problem_registry import ProblemRegistry
    reg = ProblemRegistry()
    reg.discover(["problems"])
    gen = reg.get_generator(args.task)
    tok = ByteTokenizer()

    # Optional: also unfreeze base + existing bridges at a low LR
    base_params = []
    if args.base_lr > 0:
        # Unfreeze: every param that's not in the new bridge.
        new_bridge_param_ids = {id(p) for p in new_bridge.parameters()}
        for p in router.parameters():
            if id(p) not in new_bridge_param_ids:
                p.requires_grad = True
                base_params.append(p)

    trainable = [p for p in router.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"Loaded router from:    {args.start_from}")
    print(f"  base d_model:        {ck['router_d_model']}, layers: {ck['router_n_layers']}")
    print(f"  pre-existing specs:  {len(inner_specialists)}")
    print(f"New specialist:        {args.new_specialist} (d={new_sp.d_model})")
    if args.base_lr > 0:
        print(f"Trainable now:         {n_train:,}  (new bridge @ lr={args.lr} + "
              f"base @ lr={args.base_lr})")
    else:
        print(f"Trainable now:         {n_train:,}  (only the new bridge — base is frozen)")
    print(f"Total router params:   {sum(p.numel() for p in router.parameters()):,}")
    print()

    # 5. Acc BEFORE the hot-plug fine-tune (the new bridge starts open;
    # gate σ(0)=0.5, recv weights small-random — so the router is
    # already perturbed by the new bridge's initial output).
    acc_before = evaluate(router, gen, tok, 256, args.device)
    print(f"acc_before fine-tune = {acc_before:.1%}  (new bridge at init, base unchanged)")

    if args.base_lr > 0 and base_params:
        opt = torch.optim.AdamW([
            {"params": list(new_bridge.parameters()), "lr": args.lr},
            {"params": base_params, "lr": args.base_lr},
        ], weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    history = [{"step": 0, "acc": acc_before,
                "gate_new": float(torch.sigmoid(new_bridge.gate.bias).mean())}]

    t0 = time.time()
    for step in range(1, args.steps + 1):
        tokens, seps = build_batch(gen, tok, args.batch_size, args.device)
        loss, _ = loss_and_acc(router, tokens, seps, args.device)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % args.eval_every == 0:
            with torch.no_grad():
                eval_tokens, eval_seps = build_batch(gen, tok, 256, args.device)
                _, acc = loss_and_acc(router, eval_tokens, eval_seps, args.device)
                gate_mean = router.gate_stats(eval_tokens)
            elapsed = time.time() - t0
            print(f"step {step:5d}  loss={float(loss.detach()):.4f}  acc={acc:.1%}  "
                  f"({elapsed:.0f}s)  gates={[round(g,3) for g in gate_mean]}", flush=True)
            history.append({
                "step": step, "loss": float(loss.detach()),
                "acc": acc, "gates": gate_mean,
            })

    acc_after = evaluate(router, gen, tok, 256, args.device)
    print()
    print(f"acc_before = {acc_before:.1%}")
    print(f"acc_after  = {acc_after:.1%}")
    print(f"Δ = {acc_after - acc_before:+.1%}")

    out = Path("/tmp/hot_plug_result.json")
    out.write_text(json.dumps({
        "start_from": args.start_from,
        "new_specialist": args.new_specialist,
        "task": args.task,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "delta": acc_after - acc_before,
        "n_trainable_added": n_train,
        "history": history,
    }, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
