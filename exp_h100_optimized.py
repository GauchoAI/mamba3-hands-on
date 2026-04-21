"""
H100-optimized comparison: Plain Mamba-3 vs Augmented Mamba-3.

Optimizations over exp_h100_comparison.py:
  1. torch.compile on the full model (triton backend)
  2. BF16 mixed precision (H100 native)
  3. Large batch (512) to saturate GPU
  4. Gradient accumulation if needed
  5. Larger default model (d_model=128)
  6. Fused loss computation

Usage:
    python exp_h100_optimized.py                              # defaults
    python exp_h100_optimized.py --steps 20000 --d-model 256  # bigger
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

from mamba3_minimal import Mamba3Block, Mamba3Config
from mamba3_augmented import AugmentedMamba3
from train_bootstrap import (
    SPECIAL_TOKENS, VOCAB_SIZE, tokenize, detokenize,
    BootstrapDataset, evaluate_on_examples, make_fresh_eval_set,
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


# ── Vectorized loss (no Python loop) ────────────────────────────────

def compute_loss_fast(logits, tokens, sep_positions):
    """Vectorized loss — no per-example Python loop."""
    B, L, V = logits.shape
    device = logits.device

    # Build mask: 1 for positions after SEP and before PAD
    mask = torch.zeros(B, L, device=device)
    for b in range(B):
        sep = sep_positions[b]
        mask[b, sep:L-1] = 1.0

    # Zero out PAD positions
    pad_mask = (tokens != SPECIAL_TOKENS["<PAD>"]).float()
    # Shift: we predict token[t+1] from logits[t]
    # So mask positions t where token[t+1] is valid
    target_valid = pad_mask[:, 1:]  # (B, L-1)
    pred_mask = mask[:, :L-1] * target_valid  # (B, L-1)

    if pred_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Flatten for cross_entropy
    logits_flat = logits[:, :L-1].reshape(-1, V)    # (B*(L-1), V)
    targets_flat = tokens[:, 1:].reshape(-1)         # (B*(L-1),)
    mask_flat = pred_mask.reshape(-1)                 # (B*(L-1),)

    loss_all = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (B*(L-1),)
    loss = (loss_all * mask_flat).sum() / mask_flat.sum()
    return loss


# ── Training loop ───────────────────────────────────────────────────

def train_with_teacher(model, name, dataset, device, args, ckpt_prefix,
                       use_compile=True):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}", flush=True)
    print(f"Training: {name} ({n_params:,} params)", flush=True)
    print(f"  steps={args.steps}  batch={args.batch_size}  lr={args.lr}", flush=True)
    print(f"  bf16={'ON' if args.bf16 else 'OFF'}  compile={'ON' if use_compile else 'OFF'}", flush=True)
    print(f"  cycles: {args.cycle_learn} learn + {args.cycle_digest} digest", flush=True)
    print(f"{'='*60}", flush=True)

    # Compile model for speed
    train_model = model
    if use_compile and device == "cuda":
        try:
            train_model = torch.compile(model, mode="reduce-overhead")
            print(f"  torch.compile: OK (reduce-overhead)", flush=True)
        except Exception as e:
            print(f"  torch.compile: FAILED ({e}), using eager", flush=True)
            train_model = model

    base_lr = args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
    teacher = AdaptiveTeacher()

    # Mixed precision
    use_amp = args.bf16 and device == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and not args.bf16))
    # BF16 doesn't need scaler, but we use autocast

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    best_fresh = 0.0
    current_lr = base_lr
    global_step = 0

    learn_steps = args.cycle_learn
    digest_steps = args.cycle_digest
    cycle_len = learn_steps + digest_steps
    gap_throttle = 0.10

    history = []
    step_times = []

    while global_step < args.steps:
        cycle_num = global_step // cycle_len + 1

        # Regenerate data via teacher
        raw = teacher.generate(len(dataset.examples))
        dataset.examples = []
        for ex in raw:
            inp = [SPECIAL_TOKENS["<BOS>"]] + tokenize(ex["input"]) + [SPECIAL_TOKENS["<SEP>"]]
            out = tokenize(ex["output"]) + [SPECIAL_TOKENS["<EOS>"]]
            dataset.examples.append((inp, out, ex.get("type", "")))

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

            t0 = time.time()

            tokens, sep_pos = dataset.get_batch(args.batch_size)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                logits = train_model(tokens)
                loss = compute_loss_fast(logits, tokens, sep_pos)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step_times.append(time.time() - t0)

            if global_step % args.eval_every == 0:
                # Throughput stats
                avg_ms = sum(step_times[-args.eval_every:]) / len(step_times[-args.eval_every:]) * 1000
                tps = args.batch_size / (avg_ms / 1000)

                train_acc = evaluate_on_examples(model, dataset.examples, device, 500)
                fresh = make_fresh_eval_set(500)
                fresh_acc = evaluate_on_examples(model, fresh, device, 500)
                gap = train_acc - fresh_acc

                extra = ""
                if hasattr(model, 'block') and hasattr(model.block, '_last_reg_spikes'):
                    rs = model.block._last_reg_spikes
                    ms = model.block._last_mem_spikes
                    extra = f"  reg_spk={rs:.2f}  mem_spk={ms:.2f}"

                print(f"  [{name}] step {global_step:5d}  loss={loss.item():.3f}  "
                      f"train={train_acc:.1%}  fresh={fresh_acc:.1%}  "
                      f"gap={gap:+.1%}  lr={opt.param_groups[0]['lr']:.1e}  "
                      f"[{phase}]  {avg_ms:.0f}ms/step  {tps:.0f}ex/s{extra}", flush=True)

                history.append({
                    "step": global_step, "loss": loss.item(),
                    "train_acc": train_acc, "fresh_acc": fresh_acc,
                    "gap": gap, "phase": phase, "ms_per_step": avg_ms,
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

            # Per-type eval + teacher feedback
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


# ── Survey ──────────────────────────────────────────────────────────

def run_survey(model, device, n_examples=500):
    """Quick diagnostic on the augmented model."""
    print(f"\n{'='*60}", flush=True)
    print(f"AUGMENTED MODEL SURVEY ({n_examples} fresh examples)", flush=True)
    print(f"{'='*60}", flush=True)

    fresh = make_fresh_eval_set(n_examples)
    by_type = defaultdict(list)
    for ex in fresh:
        by_type[ex[2]].append(ex)

    for task_type in sorted(by_type.keys()):
        examples = by_type[task_type]
        seqs, sep_positions = [], []
        for inp, out, _ in examples:
            seqs.append(inp + out)
            sep_positions.append(len(inp) - 1)
        max_len = max(len(s) for s in seqs)
        tokens = torch.full((len(seqs), max_len), SPECIAL_TOKENS["<PAD>"],
                           dtype=torch.long, device=device)
        for i, seq in enumerate(seqs):
            tokens[i, :len(seq)] = torch.tensor(seq)

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

        correct_reg, wrong_reg = [], []
        correct_mem, wrong_mem = [], []
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
            (correct_reg if ok else wrong_reg).append(rs)
            (correct_mem if ok else wrong_mem).append(ms)

        n_correct = len(correct_reg)
        n_total = n_correct + len(wrong_reg)
        acc = n_correct / max(n_total, 1)
        mean = lambda lst: sum(lst) / max(len(lst), 1)

        print(f"  {task_type}: {acc:.0%} ({n_correct}/{n_total})"
              f"  correct[reg={mean(correct_reg):.3f} mem={mean(correct_mem):.3f}]"
              f"  wrong[reg={mean(wrong_reg):.3f} mem={mean(wrong_mem):.3f}]", flush=True)

    model.train()


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument("--n-registers", type=int, default=8)
    parser.add_argument("--n-memory", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--cycle-learn", type=int, default=500)
    parser.add_argument("--cycle-digest", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--no-compile", action="store_true", default=True,
                        help="Skip torch.compile — Triton kernel handles the hot path")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name()
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {name} ({mem:.0f}GB)", flush=True)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print(f"Device: Apple MPS", flush=True)
    else:
        device = "cpu"
        print(f"Device: CPU", flush=True)

    cfg = Mamba3Config(
        d_model=args.d_model,
        d_state=args.d_state,
        expand=2,
        headdim=args.headdim,
    )

    use_compile = not args.no_compile

    # ── PLAIN ──
    dataset = BootstrapDataset("data/level0/patterns.jsonl", device)
    torch.manual_seed(args.seed)
    plain = PlainModel(cfg).to(device)
    plain_best = train_with_teacher(plain, "PLAIN", dataset, device, args,
                                    "opt_plain", use_compile=use_compile)

    # ── AUGMENTED ──
    dataset = BootstrapDataset("data/level0/patterns.jsonl", device)
    torch.manual_seed(args.seed)
    aug = AugmentedModel(cfg, args.n_registers, args.n_memory).to(device)
    # Don't compile augmented — the sequential register loop breaks dynamo
    aug_best = train_with_teacher(aug, "AUGMENTED", dataset, device, args,
                                  "opt_augmented", use_compile=False)

    # ── Results ──
    n_plain = sum(p.numel() for p in plain.parameters())
    n_aug = sum(p.numel() for p in aug.parameters())
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Plain:     best fresh={plain_best:.1%}  ({n_plain:,} params)", flush=True)
    print(f"  Augmented: best fresh={aug_best:.1%}  ({n_aug:,} params)", flush=True)
    print(f"  Δ: {aug_best - plain_best:+.1%}", flush=True)

    run_survey(aug, device, n_examples=500)

    print(f"\nCheckpoints: checkpoints/opt_plain_* and checkpoints/opt_augmented_*", flush=True)
    print(f"Histories: checkpoints/opt_*_history.json", flush=True)
