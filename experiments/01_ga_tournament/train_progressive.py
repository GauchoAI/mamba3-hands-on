"""
Progressive training: kernel (curriculum) + cortex (language).

Alternates between curriculum training (kernel layers) and language
training (cortex layers). Shared embedding learns from both.

Uses StableMax + PerpGrad for grokking-friendly training.
Starts with 1 kernel layer, grows on demand.

Usage:
    # Curriculum only (test on mini):
    python train_progressive.py --kernel-steps 500 --cortex-steps 0 --total-cycles 20

    # Full alternating:
    python train_progressive.py --kernel-steps 500 --cortex-steps 500 \
        --language-data data/bilingual.txt --total-cycles 100

    # H100 with growth:
    python train_progressive.py --kernel-steps 1000 --cortex-steps 1000 \
        --total-cycles 500 --grow-kernel-at 10,50 --grow-cortex-at 5,30
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

from progressive_model import (
    ProgressiveModel, ByteTokenizer, VOCAB_SIZE, BOS, EOS, SEP, PAD,
)
from grokking import stable_cross_entropy, PerpGradOptimizer
from generators.teacher import AdaptiveTeacher


# ── Data loaders ────────────────────────────────────────────────────

class CurriculumLoader:
    """Loads curriculum examples, encodes as bytes."""
    def __init__(self, teacher, device):
        self.teacher = teacher
        self.device = device
        self.tok = ByteTokenizer()
        self.examples = []  # cached encoded examples

    def refresh(self, count=10000):
        """Generate fresh curriculum data from teacher."""
        raw = self.teacher.generate(count)
        self.examples = []
        for ex in raw:
            tokens, sep_pos = self.tok.encode_curriculum(ex)
            self.examples.append((tokens, sep_pos, ex.get("type", "")))

    def get_batch(self, batch_size):
        indices = torch.randint(0, len(self.examples), (batch_size,))
        max_len = 0
        batch = []
        for idx in indices:
            tokens, sep_pos, _ = self.examples[idx.item()]
            batch.append((tokens, sep_pos))
            max_len = max(max_len, len(tokens))

        token_tensor = torch.full((batch_size, max_len), PAD,
                                  dtype=torch.long, device=self.device)
        sep_positions = []
        for i, (tokens, sep_pos) in enumerate(batch):
            token_tensor[i, :len(tokens)] = torch.tensor(tokens)
            sep_positions.append(sep_pos)

        return token_tensor, sep_positions


class LanguageLoader:
    """Loads raw text as bytes for next-byte prediction."""
    def __init__(self, path, device, seq_len=64):
        with open(path, "rb") as f:
            self.data = f.read()
        self.device = device
        self.seq_len = seq_len
        print(f"LanguageLoader: {len(self.data):,} bytes from {path}", flush=True)

    def get_batch(self, batch_size):
        max_start = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,))
        tokens = torch.zeros(batch_size, self.seq_len + 1,
                            dtype=torch.long, device=self.device)
        for i, s in enumerate(starts):
            chunk = self.data[s:s + self.seq_len + 1]
            tokens[i, :len(chunk)] = torch.tensor(list(chunk))
        inputs = tokens[:, :-1]   # (B, seq_len)
        targets = tokens[:, 1:]   # (B, seq_len)
        return inputs, targets


# ── Loss functions ──────────────────────────────────────────────────

def curriculum_loss(logits, tokens, sep_positions):
    """StableMax loss on output portion only (after SEP)."""
    B, L, V = logits.shape
    device = logits.device

    mask = torch.zeros(B, L, device=device)
    for b in range(B):
        sep = sep_positions[b]
        mask[b, sep:L-1] = 1.0

    pad_mask = (tokens != PAD).float()
    target_valid = pad_mask[:, 1:]
    pred_mask = mask[:, :L-1] * target_valid

    if pred_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits_flat = logits[:, :L-1].reshape(-1, V)
    targets_flat = tokens[:, 1:].reshape(-1)
    mask_flat = pred_mask.reshape(-1)

    loss_all = stable_cross_entropy(logits_flat, targets_flat, reduction='none')
    return (loss_all * mask_flat).sum() / mask_flat.sum()


def language_loss(logits, targets):
    """StableMax loss on all positions (next-byte prediction)."""
    B, L, V = logits.shape
    return stable_cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction='mean'
    )


# ── Evaluation ──────────────────────────────────────────────────────

def evaluate_curriculum(model, teacher, device, n_eval=300):
    """Evaluate per-type accuracy on fresh curriculum examples."""
    from generators.level0_patterns import generate_dataset
    tok = ByteTokenizer()

    raw = generate_dataset(n_eval)
    by_type = defaultdict(list)
    for ex in raw:
        by_type[ex["type"]].append(ex)

    type_accs = {}
    model.eval()
    with torch.no_grad():
        for task_type, examples in sorted(by_type.items()):
            correct = 0
            total = 0
            # Batch
            batch_tokens = []
            batch_seps = []
            batch_outputs = []
            for ex in examples[:50]:
                tokens, sep_pos = tok.encode_curriculum(ex)
                out_bytes = list(ex["output"].encode("utf-8"))
                batch_tokens.append(tokens)
                batch_seps.append(sep_pos)
                batch_outputs.append(out_bytes)

            max_len = max(len(t) for t in batch_tokens)
            token_tensor = torch.full((len(batch_tokens), max_len), PAD,
                                     dtype=torch.long, device=device)
            for i, t in enumerate(batch_tokens):
                token_tensor[i, :len(t)] = torch.tensor(t)

            logits = model(token_tensor)

            for i in range(len(batch_tokens)):
                sep = batch_seps[i]
                out = batch_outputs[i]
                ok = True
                for j, expected_byte in enumerate(out):
                    pos = sep + j  # predict byte at sep+j+1 from logits at sep+j
                    if pos >= logits.shape[1]:
                        ok = False
                        break
                    pred = logits[i, pos].argmax().item()
                    if pred != expected_byte:
                        ok = False
                        break
                if ok:
                    correct += 1
                total += 1

            type_accs[task_type] = correct / max(total, 1)

    model.train()
    return type_accs


def evaluate_language(model, loader, n_batches=10):
    """Evaluate BPC on language data."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = loader.get_batch(64)
            logits = model(x)
            loss = language_loss(logits, y)
            total_loss += loss.item() * x.shape[0] * x.shape[1]
            total_tokens += x.shape[0] * x.shape[1]
    model.train()
    bpc = total_loss / max(total_tokens, 1) / 0.6931  # nats → bits
    return bpc


# ── Main training loop ─────────────────────────────────────────────

def train(args):
    # Device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: {torch.cuda.get_device_name()}", flush=True)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print(f"Device: Apple MPS", flush=True)
    else:
        device = "cpu"
        print(f"Device: CPU", flush=True)

    # Model
    model = ProgressiveModel(
        d_model=args.d_model, d_state=args.d_state,
        expand=2, headdim=args.headdim,
    ).to(device)

    # Parse growth schedule
    grow_kernel = set(int(x) for x in args.grow_kernel_at.split(",") if x) if args.grow_kernel_at else set()
    grow_cortex = set(int(x) for x in args.grow_cortex_at.split(",") if x) if args.grow_cortex_at else set()

    # Start with 1 kernel layer (after .to(device) so it lands on the right device)
    model.add_kernel_layer()

    # Teacher + loaders
    teacher = AdaptiveTeacher(sequential_unlock=True)
    cur_loader = CurriculumLoader(teacher, device)
    cur_loader.refresh()

    lang_loader = None
    if args.language_data and args.cortex_steps > 0:
        lang_loader = LanguageLoader(args.language_data, device)

    # Optimizer + PerpGrad
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    perp = PerpGradOptimizer(model)

    # Checkpoint dir
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # Resume
    start_cycle = 0
    best_fresh = 0.0
    best_bpc = 99.0
    if args.resume:
        print(f"Resuming from {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Rebuild layers
        for _ in range(ckpt.get("n_kernel_layers", 1) - len(model.kernel_layers)):
            model.add_kernel_layer()
        for _ in range(ckpt.get("n_cortex_layers", 0)):
            model.add_cortex_layer()
        model.load_state_dict(ckpt["model"])
        if "teacher" in ckpt:
            teacher = AdaptiveTeacher.from_dict(ckpt["teacher"])
            cur_loader.teacher = teacher
        start_cycle = ckpt.get("cycle", 0)
        best_fresh = ckpt.get("best_fresh", 0.0)
        best_bpc = ckpt.get("best_bpc", 99.0)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        print(f"Resumed at cycle {start_cycle}", flush=True)

    print(f"\n{model.summary()}", flush=True)
    print(f"Kernel steps/cycle: {args.kernel_steps}", flush=True)
    print(f"Cortex steps/cycle: {args.cortex_steps}", flush=True)
    print(f"Total cycles: {args.total_cycles}", flush=True)
    print(f"Growth: kernel at {sorted(grow_kernel)}, cortex at {sorted(grow_cortex)}", flush=True)
    print(f"PerpGrad: ON, StableMax: ON, weight_decay={args.weight_decay}", flush=True)
    print(flush=True)

    global_step = start_cycle * (args.kernel_steps + args.cortex_steps)

    for cycle in range(start_cycle, args.total_cycles):
        t_cycle = time.time()

        # ── Growth check ──
        if cycle in grow_kernel:
            idx = model.add_kernel_layer()
            opt.add_param_group({"params": list(model.kernel_layers[idx].parameters())})
        if cycle in grow_cortex:
            idx = model.add_cortex_layer()
            opt.add_param_group({"params": list(model.cortex_layers[idx].parameters())})

        # ── KERNEL PHASE ──
        if args.kernel_steps > 0:
            model.set_mode("kernel")
            cur_loader.refresh()

            for step in range(args.kernel_steps):
                global_step += 1
                tokens, sep_pos = cur_loader.get_batch(args.batch_size)
                logits = model(tokens)
                loss = curriculum_loss(logits, tokens, sep_pos)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                perp.project()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # Eval curriculum
            if cycle % args.eval_every == 0 or cycle == start_cycle:
                type_accs = evaluate_curriculum(model, teacher, device)
                # Fresh overall
                all_correct = sum(v * 50 for v in type_accs.values())
                all_total = len(type_accs) * 50
                fresh_acc = all_correct / max(all_total, 1)

                print(f"[Cycle {cycle}] KERNEL  loss={loss.item():.3f}  "
                      f"fresh={fresh_acc:.1%}  trainable={model.get_trainable_params():,}",
                      flush=True)
                for t, a in sorted(type_accs.items()):
                    if a > 0 or t in teacher.unlocked_tasks:
                        print(f"  {t}: {a:.0%}", flush=True)

                teacher.set_step(global_step)
                teacher.observe(type_accs)
                print(f"  teacher:\n{teacher.get_status()}", flush=True)
                if teacher.mastery_log:
                    print(teacher.get_learning_report(), flush=True)

                if fresh_acc > best_fresh:
                    best_fresh = fresh_acc

        # ── CORTEX PHASE ──
        if args.cortex_steps > 0 and lang_loader is not None:
            model.set_mode("cortex")
            # Add cortex layer if none exist yet
            if len(model.cortex_layers) == 0:
                idx = model.add_cortex_layer()
                opt.add_param_group({"params": list(model.cortex_layers[idx].parameters())})
                model.set_mode("cortex")

            for step in range(args.cortex_steps):
                global_step += 1
                x, y = lang_loader.get_batch(args.batch_size)
                logits = model(x)
                loss = language_loss(logits, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                perp.project()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # Eval language
            if cycle % args.eval_every == 0 or cycle == start_cycle:
                bpc = evaluate_language(model, lang_loader)
                if bpc < best_bpc:
                    best_bpc = bpc
                print(f"[Cycle {cycle}] CORTEX  loss={loss.item():.3f}  "
                      f"BPC={bpc:.2f}  trainable={model.get_trainable_params():,}",
                      flush=True)

        # ── Checkpoint ──
        cycle_time = time.time() - t_cycle
        if cycle % args.eval_every == 0 or cycle == args.total_cycles - 1:
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "teacher": teacher.to_dict(),
                "cycle": cycle + 1,
                "global_step": global_step,
                "n_kernel_layers": len(model.kernel_layers),
                "n_cortex_layers": len(model.cortex_layers),
                "d_model": args.d_model,
                "d_state": args.d_state,
                "headdim": args.headdim,
                "best_fresh": best_fresh,
                "best_bpc": best_bpc,
            }
            torch.save(ckpt_data, ckpt_dir / "progressive_latest.pt")
            if best_fresh == fresh_acc if args.kernel_steps > 0 else True:
                torch.save(ckpt_data, ckpt_dir / "progressive_best.pt")
            print(f"  [checkpoint saved, cycle {cycle+1}, {cycle_time:.1f}s/cycle]",
                  flush=True)

    print(f"\nDone. {args.total_cycles} cycles, {global_step} steps.", flush=True)
    print(model.summary(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument("--kernel-steps", type=int, default=500)
    parser.add_argument("--cortex-steps", type=int, default=0)
    parser.add_argument("--total-cycles", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW weight decay (0 = rely on PerpGrad alone)")
    parser.add_argument("--language-data", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Evaluate every N cycles")
    parser.add_argument("--grow-kernel-at", type=str, default="",
                        help="Comma-separated cycle numbers to add kernel layers")
    parser.add_argument("--grow-cortex-at", type=str, default="",
                        help="Comma-separated cycle numbers to add cortex layers")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args)
