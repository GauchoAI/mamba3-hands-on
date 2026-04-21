"""
H100 head-to-head: Plain Mamba-3 vs Augmented Mamba-3.

Both use the adaptive teacher. Same data, same curriculum, same eval.
Longer training (10K steps), larger model option, proper checkpointing,
diagnostic logging on the augmented model.

Usage:
    python exp_h100_comparison.py                         # defaults
    python exp_h100_comparison.py --steps 20000 --d-model 128  # bigger
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

from mamba3_minimal import Mamba3Block, Mamba3Config
from mamba3_augmented import AugmentedMamba3
from train_bootstrap import (
    SPECIAL_TOKENS, VOCAB_SIZE, tokenize, detokenize,
    BootstrapDataset, compute_loss, evaluate_on_examples, make_fresh_eval_set,
)
from generators.teacher import AdaptiveTeacher


# ── Models ───────────────────────────────────────────────────────────

class PlainModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.block = Mamba3Block(cfg)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.block(x)
        x = self.norm(x)
        return self.head(x)


class AugmentedModel(nn.Module):
    def __init__(self, cfg, n_registers=8, n_memory=16):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.block = AugmentedMamba3(cfg, n_registers, n_memory)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.block(x)
        return self.head(x)


# ── Shared training loop ────────────────────────────────────────────

def train_with_teacher(model, name, dataset, device, args, ckpt_prefix):
    """
    Train one model with adaptive teacher, cycle-based LR, diagnostics.
    Returns best_fresh accuracy.
    """
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}", flush=True)
    print(f"Training: {name} ({n_params:,} params)", flush=True)
    print(f"  steps={args.steps}  batch={args.batch_size}  lr={args.lr}", flush=True)
    print(f"  cycles: {args.cycle_learn} learn + {args.cycle_digest} digest", flush=True)
    print(f"{'='*60}", flush=True)

    base_lr = args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
    teacher = AdaptiveTeacher()

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    best_fresh = 0.0
    current_lr = base_lr
    global_step = 0

    learn_steps = args.cycle_learn
    digest_steps = args.cycle_digest
    cycle_len = learn_steps + digest_steps
    gap_throttle = 0.10

    # History for final report
    history = []

    while global_step < args.steps:
        cycle_num = global_step // cycle_len + 1

        # Regenerate data via teacher
        raw = teacher.generate(len(dataset.examples))
        dataset.examples = []
        for ex in raw:
            inp = [SPECIAL_TOKENS["<BOS>"]] + tokenize(ex["input"]) + [SPECIAL_TOKENS["<SEP>"]]
            out = tokenize(ex["output"]) + [SPECIAL_TOKENS["<EOS>"]]
            dataset.examples.append((inp, out, ex.get("type", "")))

        # Learning phase
        for param_group in opt.param_groups:
            param_group["lr"] = current_lr
        phase = "LEARN"

        for step_in_cycle in range(cycle_len):
            global_step += 1
            if global_step > args.steps:
                break

            if step_in_cycle == learn_steps:
                phase = "DIGEST"
                for param_group in opt.param_groups:
                    param_group["lr"] = current_lr * 0.1

            tokens, sep_pos = dataset.get_batch(args.batch_size)
            logits = model(tokens)
            loss = compute_loss(logits, tokens, sep_pos)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if global_step % args.eval_every == 0:
                train_acc = evaluate_on_examples(model, dataset.examples, device, 500)
                fresh = make_fresh_eval_set(500)
                fresh_acc = evaluate_on_examples(model, fresh, device, 500)
                gap = train_acc - fresh_acc

                # Augmented model diagnostics
                extra = ""
                if hasattr(model, 'block') and hasattr(model.block, '_last_reg_spikes'):
                    rs = model.block._last_reg_spikes
                    ms = model.block._last_mem_spikes
                    extra = f"  reg_spk={rs:.2f}  mem_spk={ms:.2f}"

                print(f"  [{name}] step {global_step:5d}  loss={loss.item():.3f}  "
                      f"train={train_acc:.1%}  fresh={fresh_acc:.1%}  "
                      f"gap={gap:+.1%}  lr={opt.param_groups[0]['lr']:.1e}  "
                      f"[{phase}]{extra}", flush=True)

                history.append({
                    "step": global_step, "loss": loss.item(),
                    "train_acc": train_acc, "fresh_acc": fresh_acc,
                    "gap": gap, "phase": phase,
                })

                # Gap throttle
                if gap > gap_throttle and phase == "LEARN":
                    current_lr = max(current_lr * 0.7, base_lr * 0.01)
                    for param_group in opt.param_groups:
                        param_group["lr"] = current_lr
                    print(f"    throttle LR → {current_lr:.1e}", flush=True)
                elif gap < 0.05 and current_lr < base_lr:
                    current_lr = min(current_lr * 1.2, base_lr)
                    if phase == "LEARN":
                        for param_group in opt.param_groups:
                            param_group["lr"] = current_lr

                # Checkpoint
                ckpt_data = {
                    "model": model.state_dict(),
                    "step": global_step,
                    "train_acc": train_acc,
                    "fresh_acc": fresh_acc,
                    "best_fresh": best_fresh,
                    "name": name,
                }
                torch.save(ckpt_data, ckpt_dir / f"{ckpt_prefix}_step{global_step}.pt")
                torch.save(ckpt_data, ckpt_dir / f"{ckpt_prefix}_latest.pt")

                if fresh_acc > best_fresh:
                    best_fresh = fresh_acc
                    ckpt_data["best_fresh"] = best_fresh
                    torch.save(ckpt_data, ckpt_dir / f"{ckpt_prefix}_best.pt")
                    print(f"    ★ new best fresh={fresh_acc:.1%}", flush=True)

            # Per-type eval + teacher feedback (every 4 evals)
            if global_step % (args.eval_every * 4) == 0:
                fresh_typed = make_fresh_eval_set(800)
                by_type = defaultdict(list)
                for inp, out, t in fresh_typed:
                    if len(by_type[t]) < 100:
                        by_type[t].append((inp, out, t))
                type_accs = {}
                for t, exs in sorted(by_type.items()):
                    type_accs[t] = evaluate_on_examples(model, exs, device, len(exs))
                    print(f"    {t}: {type_accs[t]:.0%}", flush=True)
                teacher.observe(type_accs)
                print(f"  [{name}] teacher:\n{teacher.get_status()}", flush=True)

    # Save history
    with open(ckpt_dir / f"{ckpt_prefix}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return best_fresh


# ── Diagnostic survey (augmented only) ──────────────────────────────

def run_survey(model, device, n_examples=500):
    """Quick diagnostic on the trained augmented model."""
    print(f"\n{'='*60}", flush=True)
    print(f"AUGMENTED MODEL SURVEY ({n_examples} fresh examples)", flush=True)
    print(f"{'='*60}", flush=True)

    fresh = make_fresh_eval_set(n_examples)
    by_type = defaultdict(list)
    for ex in fresh:
        by_type[ex[2]].append(ex)

    for task_type in sorted(by_type.keys()):
        examples = by_type[task_type]
        seqs = []
        sep_positions = []
        for inp, out, _ in examples:
            seqs.append(inp + out)
            sep_positions.append(len(inp) - 1)
        max_len = max(len(s) for s in seqs)
        tokens = torch.full((len(seqs), max_len), SPECIAL_TOKENS["<PAD>"],
                           dtype=torch.long, device=device)
        for i, seq in enumerate(seqs):
            tokens[i, :len(seq)] = torch.tensor(seq)

        # Forward with diagnostic capture
        model.eval()
        with torch.no_grad():
            x = model.embed(tokens)
            B, L, D = x.shape
            block = model.block
            ssm_out = block.ssm(x)
            reg_state = x.new_zeros(B, block.registers.n_registers, D)
            mem_state = x.new_zeros(B, block.memory.n_slots, D)

            reg_gates_sum = torch.zeros(B, device='cpu')
            mem_gates_sum = torch.zeros(B, device='cpu')
            outputs = []

            for t in range(L):
                h_t = ssm_out[:, t]
                reg_read, reg_state, reg_gate = block.registers(h_t, reg_state)
                mem_read, mem_state, mem_gate = block.memory(h_t, mem_state)
                combined = block.combine(torch.cat([h_t, reg_read, mem_read], dim=-1))
                combined = block.norm(combined + h_t)
                outputs.append(combined)
                reg_gates_sum += reg_gate.cpu()
                mem_gates_sum += mem_gate.cpu()

            logits = model.head(torch.stack(outputs, dim=1))

        # Check correctness
        correct_reg = []
        wrong_reg = []
        correct_mem = []
        wrong_mem = []

        for i in range(B):
            sep = sep_positions[i]
            ok = True
            for t in range(sep, L - 1):
                target = tokens[i, t + 1].item()
                if target == SPECIAL_TOKENS["<PAD>"]:
                    break
                if logits[i, t].argmax().item() != target:
                    ok = False
                    break
            rs = (reg_gates_sum[i] / L).item()
            ms = (mem_gates_sum[i] / L).item()
            if ok:
                correct_reg.append(rs)
                correct_mem.append(ms)
            else:
                wrong_reg.append(rs)
                wrong_mem.append(ms)

        n_correct = len(correct_reg)
        n_total = len(correct_reg) + len(wrong_reg)
        acc = n_correct / max(n_total, 1)

        def mean(lst):
            return sum(lst) / max(len(lst), 1)

        cr = mean(correct_reg) if correct_reg else 0
        cm = mean(correct_mem) if correct_mem else 0
        wr = mean(wrong_reg) if wrong_reg else 0
        wm = mean(wrong_mem) if wrong_mem else 0

        print(f"  {task_type}: {acc:.0%} ({n_correct}/{n_total})"
              f"  correct[reg={cr:.3f} mem={cm:.3f}]"
              f"  wrong[reg={wr:.3f} mem={wm:.3f}]", flush=True)

    model.train()


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H100 comparison: Plain vs Augmented Mamba-3")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument("--n-registers", type=int, default=8)
    parser.add_argument("--n-memory", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--cycle-learn", type=int, default=500)
    parser.add_argument("--cycle-digest", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Device detection: CUDA (H100) > MPS (Apple) > CPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: {torch.cuda.get_device_name()}", flush=True)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}", flush=True)

    cfg = Mamba3Config(
        d_model=args.d_model,
        d_state=args.d_state,
        expand=2,
        headdim=args.headdim,
    )

    # ── Train PLAIN ──
    dataset = BootstrapDataset("data/level0/patterns.jsonl", device)
    torch.manual_seed(args.seed)
    plain = PlainModel(cfg).to(device)
    plain_best = train_with_teacher(plain, "PLAIN", dataset, device, args, "plain")

    # ── Train AUGMENTED ──
    dataset = BootstrapDataset("data/level0/patterns.jsonl", device)
    torch.manual_seed(args.seed)
    aug = AugmentedModel(cfg, args.n_registers, args.n_memory).to(device)
    aug_best = train_with_teacher(aug, "AUGMENTED", dataset, device, args, "augmented")

    # ── Final comparison ──
    n_plain = sum(p.numel() for p in plain.parameters())
    n_aug = sum(p.numel() for p in aug.parameters())
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Plain:     best fresh={plain_best:.1%}  ({n_plain:,} params)", flush=True)
    print(f"  Augmented: best fresh={aug_best:.1%}  ({n_aug:,} params)", flush=True)
    print(f"  Δ: {aug_best - plain_best:+.1%}", flush=True)

    # Survey on augmented model
    run_survey(aug, device, n_examples=500)

    print(f"\nCheckpoints saved to checkpoints/plain_* and checkpoints/augmented_*", flush=True)
    print(f"Training histories saved to checkpoints/*_history.json", flush=True)
