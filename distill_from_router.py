"""distill_from_router — distill a synapse-router into a solo specialist.

A synapse-router solves a task via composition (e.g. tiny router +
frozen logic_gate specialist). Its forward pass is a runtime graph,
not a single set of weights. This script asks: can a solo model
(no synapses, just a ProgressiveModel) be trained to *match the
router's predictions* via KL distillation, ending up with a single
.pt that captures the composed capability?

If yes, the ecology has a way to **crystallize compositions** into new
specialists. The library compounds: a router that solved depth-2
gates becomes a depth-2-gate specialist available to all future
routers, with no runtime synapse cost.

Usage:
  python3 distill_from_router.py \\
    --teacher checkpoints/routers/lvl1_compose.pt \\
    --task compose_logic_gate \\
    --student-d-model 32 --student-layers 2 \\
    --steps 2000 --save-to checkpoints/specialists/compose_logic_gate.pt
"""
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, ByteTokenizer, PAD, VOCAB_SIZE
from synapse import RouterModel, load_specialist, PlaceholderSpecialist


def load_router_teacher(pt_path, device):
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    spec_list = ck.get("specialist_spec")
    if spec_list:
        inner = []
        for s in spec_list:
            if s["type"] == "real":
                inner.append(load_specialist(s["path"], device=device))
            elif s["type"] == "placeholder":
                inner.append(PlaceholderSpecialist(d_model=s["d_model"]).to(device))
    else:
        inner = [load_specialist(p, device=device) for p in ck.get("specialist_paths", [])]
    teacher = RouterModel(
        router_d_model=ck["router_d_model"],
        router_d_state=ck["router_d_state"],
        router_headdim=ck["router_headdim"],
        router_n_layers=ck["router_n_layers"],
        specialists=inner,
        bridge_kind=ck.get("bridge_kind", "attend"),
    ).to(device)
    teacher.load_state_dict(ck["router_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def build_batch(gen, tok, batch_size, device):
    examples = [tok.encode_curriculum(gen()) for _ in range(batch_size)]
    max_len = max(len(t) for t, _ in examples)
    tokens = torch.full((batch_size, max_len), PAD, dtype=torch.long, device=device)
    seps = []
    for i, (toks, sep) in enumerate(examples):
        tokens[i, :len(toks)] = torch.tensor(toks, device=device)
        seps.append(sep)
    return tokens, torch.tensor(seps, device=device, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="Saved router .pt")
    ap.add_argument("--task", required=True)
    ap.add_argument("--student-d-model", type=int, default=32)
    ap.add_argument("--student-layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--ce-weight", type=float, default=0.3,
                    help="weight on hard CE loss (rest goes to soft KL)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-to", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    teacher = load_router_teacher(args.teacher, args.device)
    print(f"Teacher loaded:   {args.teacher}")
    print(f"  router d_model: {teacher.base.d_model}, "
          f"specialists: {len(teacher.specialists)}")

    student = ProgressiveModel(
        d_model=args.student_d_model, d_state=16, expand=2, headdim=16,
    ).to(args.device)
    for _ in range(args.student_layers):
        student.add_kernel_layer()
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student:          d={args.student_d_model}, L={args.student_layers}, "
          f"{n_params:,} params")

    sys.path.insert(0, ".")
    from registry.problem_registry import ProblemRegistry
    reg = ProblemRegistry()
    reg.discover(["problems"])
    gen = reg.get_generator(args.task)
    tok = ByteTokenizer()

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.1)

    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        tokens, seps = build_batch(gen, tok, args.batch_size, args.device)
        # Teacher logits (no_grad, runs the full synapse graph)
        with torch.no_grad():
            t_logits = teacher(tokens)
        s_logits = student(tokens)

        B, L, V = s_logits.shape
        pos = torch.arange(L, device=args.device).unsqueeze(0)
        sep_t = seps.unsqueeze(1)
        mask = ((pos >= sep_t) & (pos < L - 1)).float()
        pad_mask = (tokens != PAD).float()
        pred_mask = mask[:, :L - 1] * pad_mask[:, 1:]
        if pred_mask.sum() < 1:
            continue

        s_flat = s_logits[:, :L - 1].reshape(-1, V)
        t_flat = t_logits[:, :L - 1].reshape(-1, V)
        targets_flat = tokens[:, 1:].reshape(-1)
        mask_flat = pred_mask.reshape(-1)

        # CE on hard targets (positive samples only)
        ce = F.cross_entropy(s_flat, targets_flat, reduction="none")
        loss_ce = (ce * mask_flat).sum() / (mask_flat.sum() + 1e-8)

        # KL on soft teacher distribution
        T = args.temperature
        soft_t = F.softmax(t_flat / T, dim=-1)
        log_s = F.log_softmax(s_flat / T, dim=-1)
        kl = F.kl_div(log_s, soft_t, reduction="none").sum(dim=-1)
        loss_kl = (kl * mask_flat).sum() / (mask_flat.sum() + 1e-8) * (T * T)

        loss = args.ce_weight * loss_ce + (1 - args.ce_weight) * loss_kl

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0:
            student.eval()
            with torch.no_grad():
                eval_tokens, eval_seps = build_batch(gen, tok, 256, args.device)
                eval_logits = student(eval_tokens)
                B, L, V = eval_logits.shape
                pos = torch.arange(L, device=args.device).unsqueeze(0)
                sep_t = eval_seps.unsqueeze(1)
                mask = ((pos >= sep_t) & (pos < L - 1)).float()
                pad_mask = (eval_tokens != PAD).float()
                pred_mask = mask[:, :L - 1] * pad_mask[:, 1:]
                pred = eval_logits[:, :L - 1].argmax(dim=-1)
                actual = eval_tokens[:, 1:]
                per_pos_correct = (pred == actual).float() * pred_mask
                n_supervised = pred_mask.sum(dim=-1)
                n_correct = per_pos_correct.sum(dim=-1)
                example_correct = (n_correct == n_supervised) & (n_supervised > 0)
                acc = float(example_correct.float().mean())
            student.train()
            elapsed = time.time() - t0
            print(f"step {step:5d}  loss={float(loss.detach()):.4f}  "
                  f"ce={float(loss_ce.detach()):.4f} kl={float(loss_kl.detach()):.4f}  "
                  f"acc={acc:.1%}  ({elapsed:.0f}s)", flush=True)
            history.append({"step": step, "acc": acc,
                            "loss": float(loss.detach())})

    if args.save_to:
        save_dir = Path(args.save_to).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": student.state_dict(),
            "task": args.task,
            "config": {
                "d_model": args.student_d_model, "d_state": 16,
                "headdim": 16, "n_kernel_layers": args.student_layers,
                "device": args.device, "loss_fn": "kl_distill_from_router",
            },
            "accuracy": history[-1]["acc"] if history else 0.0,
            "cycles": args.steps // 100,
            "n_params": n_params,
            "distilled_from": args.teacher,
        }, args.save_to)
        print(f"Saved student → {args.save_to} ({history[-1]['acc']:.0%})")

    out_path = Path(f"/tmp/distill_{args.task}.json")
    out_path.write_text(json.dumps({
        "teacher": args.teacher, "task": args.task,
        "student_d_model": args.student_d_model,
        "student_layers": args.student_layers,
        "history": history,
        "final_acc": history[-1]["acc"] if history else 0.0,
    }, indent=2))


if __name__ == "__main__":
    main()
