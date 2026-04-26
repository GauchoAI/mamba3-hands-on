"""train_router — falsifiable test for register-space synaptic invocation.

A router model that is INTENTIONALLY too small to solve the task alone
is given a frozen specialist of the SAME task as a callable resource.
If the synapse mechanism works, the router learns to gate the specialist
on and reaches mastery; if not, it stays where the size limit puts it.

A control run with the same router but no specialist (--no-synapse)
establishes the floor.

Usage:
  python3 train_router.py --task logic_gate --steps 800
  python3 train_router.py --task logic_gate --no-synapse --steps 800   # control
"""
import argparse, json, os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn

from progressive_model import ByteTokenizer, PAD
from synapse import RouterModel, load_specialist


def load_generator(task):
    """Mirror of specialist_trainer's load_generators path — but minimal."""
    sys.path.insert(0, ".")
    from registry.problem_registry import ProblemRegistry
    reg = ProblemRegistry()
    reg.discover(["problems"])
    if task not in reg.list_problems():
        raise SystemExit(f"unknown task: {task}")
    return reg.get_generator(task)


def build_batch(gen, tok, batch_size, device):
    """Build a batch the same way specialist_trainer does (proven pattern):
    logits[:, :L-1] predict tokens[:, 1:], with a mask covering the
    supervised span (positions ≥ sep, < L-1, non-pad target).
    Returns (tokens, sep_positions) — caller computes mask + loss.
    """
    examples = [tok.encode_curriculum(gen()) for _ in range(batch_size)]
    max_len = max(len(t) for t, _ in examples)
    tokens = torch.full((batch_size, max_len), PAD,
                        dtype=torch.long, device=device)
    seps = []
    for i, (toks, sep) in enumerate(examples):
        tokens[i, :len(toks)] = torch.tensor(toks, device=device)
        seps.append(sep)
    return tokens, torch.tensor(seps, device=device, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="logic_gate")
    ap.add_argument("--specialist-pt", default=None,
                    help="Path to a frozen specialist .pt (defaults to "
                         "checkpoints/specialists/{task}.pt)")
    ap.add_argument("--no-synapse", action="store_true",
                    help="Control: router alone, no specialist invocation")
    ap.add_argument("--bridge-kind", choices=["attend", "project"], default="attend",
                    help="attend: specialist runs on original tokens, router "
                         "attends to its hidden state (v2, recommended). "
                         "project: router projects own state into specialist "
                         "via W_send (v1, marginal).")
    ap.add_argument("--router-d-model", type=int, default=32)
    ap.add_argument("--router-layers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args()

    print(f"Task:           {args.task}")
    print(f"Router:         d={args.router_d_model}, L={args.router_layers}")
    print(f"Synapse:        {'OFF (control)' if args.no_synapse else 'ON'}")
    print(f"Device:         {args.device}")

    gen = load_generator(args.task)
    tok = ByteTokenizer()

    specialists = []
    if not args.no_synapse:
        pt = args.specialist_pt or f"checkpoints/specialists/{args.task}.pt"
        if not Path(pt).exists():
            raise SystemExit(f"specialist not found: {pt} — run "
                             f"specialist_trainer first to produce one")
        sp = load_specialist(pt, device=args.device)
        specialists.append(sp)
        print(f"Specialist:     {pt} (d={sp.d_model}, frozen)")

    router = RouterModel(
        router_d_model=args.router_d_model,
        router_d_state=16,
        router_headdim=16,
        router_n_layers=args.router_layers,
        specialists=specialists,
        bridge_kind=args.bridge_kind,
    ).to(args.device)
    n_train = sum(p.numel() for p in router.parameters() if p.requires_grad)
    print(f"Router params:  {n_train:,} trainable")

    opt = torch.optim.AdamW([p for p in router.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.1)

    def loss_and_acc(tokens, seps):
        """Match specialist_trainer's masked next-token loss exactly."""
        logits = router(tokens)
        B, L, V = logits.shape
        pos = torch.arange(L, device=args.device).unsqueeze(0)
        sep_t = seps.unsqueeze(1)
        mask = ((pos >= sep_t) & (pos < L - 1)).float()
        pad_mask = (tokens != PAD).float()
        pred_mask = mask[:, :L - 1] * pad_mask[:, 1:]
        logits_flat = logits[:, :L - 1].reshape(-1, V)
        targets_flat = tokens[:, 1:].reshape(-1)
        mask_flat = pred_mask.reshape(-1)
        if mask_flat.sum() < 1:
            return torch.tensor(0.0, device=args.device, requires_grad=True), 0.0
        ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = (ce * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        # Per-example "all answer tokens correct" accuracy
        with torch.no_grad():
            pred = logits[:, :L - 1].argmax(dim=-1)
            actual = tokens[:, 1:]
            per_pos_correct = (pred == actual).float() * pred_mask
            n_supervised = pred_mask.sum(dim=-1)
            n_correct = per_pos_correct.sum(dim=-1)
            example_correct = (n_correct == n_supervised) & (n_supervised > 0)
            acc = float(example_correct.float().mean())
        return loss, acc

    t0 = time.time()
    history = []
    for step in range(1, args.steps + 1):
        tokens, seps = build_batch(gen, tok, args.batch_size, args.device)
        loss, _ = loss_and_acc(tokens, seps)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in router.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == 1:
            router.eval()
            with torch.no_grad():
                eval_tokens, eval_seps = build_batch(gen, tok, 256, args.device)
                _, acc = loss_and_acc(eval_tokens, eval_seps)
                gates = router.gate_stats(eval_tokens) if not args.no_synapse else []
            router.train()
            elapsed = time.time() - t0
            gate_str = f"  gates={[round(g, 3) for g in gates]}" if gates else ""
            print(f"step {step:5d}  loss={float(loss.detach()):.4f}  acc={acc:.1%}  "
                  f"({elapsed:.0f}s){gate_str}", flush=True)
            history.append({
                "step": step, "loss": float(loss), "acc": acc,
                "gates": gates, "elapsed_s": round(elapsed, 1),
            })

    out = {
        "task": args.task,
        "synapse_on": not args.no_synapse,
        "router_d_model": args.router_d_model,
        "router_layers": args.router_layers,
        "n_trainable": n_train,
        "history": history,
    }
    out_path = Path("/tmp") / f"router_{args.task}_{'syn' if not args.no_synapse else 'ctrl'}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResult: final acc = {history[-1]['acc']:.1%}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
