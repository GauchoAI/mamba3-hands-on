"""
Head-to-head: Plain Mamba-3 vs Augmented Mamba-3 on Level 0 pattern recognition.

Same data, same training, same eval. The only difference is the architecture.
If registers + spikes help, the augmented model should:
  1. Reach higher fresh accuracy
  2. Show lower train-fresh gap (less memorization)
  3. Show non-zero spike rates (it learned to use the registers)
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from mamba_platform.mamba3_minimal import Mamba3Block, Mamba3Config
from mamba3_augmented import AugmentedMamba3
from train_bootstrap import (
    SPECIAL_TOKENS, VOCAB_SIZE, tokenize,
    BootstrapDataset, compute_loss, evaluate_on_examples, make_fresh_eval_set,
)


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


def train_and_eval(model, name, dataset, device, steps=3000, batch=64,
                   lr=3e-3, eval_every=250, data_refresh_every=750):
    """Train a model and return results."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}", flush=True)
    print(f"Training: {name} ({n_params:,} params)", flush=True)
    print(f"{'='*60}", flush=True)

    best_fresh = 0.0
    import generators.level0_patterns as gen0

    for step in range(1, steps + 1):
        # Refresh data periodically
        if step % data_refresh_every == 1 and step > 1:
            raw = gen0.generate_dataset(len(dataset.examples))
            dataset.examples = []
            for ex in raw:
                inp = [SPECIAL_TOKENS["<BOS>"]] + tokenize(ex["input"]) + [SPECIAL_TOKENS["<SEP>"]]
                out = tokenize(ex["output"]) + [SPECIAL_TOKENS["<EOS>"]]
                dataset.examples.append((inp, out, ex.get("type", "")))

        tokens, sep_pos = dataset.get_batch(batch)
        logits = model(tokens)
        loss = compute_loss(logits, tokens, sep_pos)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == 1:
            train_acc = evaluate_on_examples(model, dataset.examples, device, 300)
            fresh = make_fresh_eval_set(300)
            fresh_acc = evaluate_on_examples(model, fresh, device, 300)
            gap = train_acc - fresh_acc
            best_fresh = max(best_fresh, fresh_acc)

            extra = ""
            if hasattr(model, 'block') and hasattr(model.block, '_last_reg_spikes'):
                rs = model.block._last_reg_spikes
                ms = model.block._last_mem_spikes
                extra = f"  reg_spk={rs:.2f}  mem_spk={ms:.2f}"

            print(f"  [{name}] step {step:4d}  loss={loss.item():.3f}  "
                  f"train={train_acc:.1%}  fresh={fresh_acc:.1%}  "
                  f"gap={gap:+.1%}{extra}", flush=True)

    # Final detailed eval
    print(f"\n  [{name}] --- Final per-type (fresh) ---", flush=True)
    fresh_all = make_fresh_eval_set(800)
    by_type = {}
    for inp, out, t in fresh_all:
        if t not in by_type:
            by_type[t] = []
        if len(by_type[t]) < 100:
            by_type[t].append((inp, out, t))

    for task_type, examples in sorted(by_type.items()):
        acc = evaluate_on_examples(model, examples, device, len(examples))
        print(f"    {task_type}: {acc:.0%}", flush=True)

    return best_fresh


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    cfg = Mamba3Config(d_model=64, d_state=16, expand=2, headdim=16)
    steps = 3000

    # Load dataset
    dataset = BootstrapDataset("data/level0/patterns.jsonl", device)

    # Train plain
    torch.manual_seed(42)
    plain = PlainModel(cfg).to(device)
    plain_best = train_and_eval(plain, "PLAIN", dataset, device, steps=steps)

    # Reload dataset (fresh)
    dataset = BootstrapDataset("data/level0/patterns.jsonl", device)

    # Train augmented
    torch.manual_seed(42)
    aug = AugmentedModel(cfg, n_registers=8, n_memory=16).to(device)
    aug_best = train_and_eval(aug, "AUGMENTED", dataset, device, steps=steps)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    n_plain = sum(p.numel() for p in plain.parameters())
    n_aug = sum(p.numel() for p in aug.parameters())
    print(f"  Plain:     best fresh={plain_best:.1%}  ({n_plain:,} params)", flush=True)
    print(f"  Augmented: best fresh={aug_best:.1%}  ({n_aug:,} params)", flush=True)
    print(f"  Δ: {aug_best - plain_best:+.1%}", flush=True)
    print(flush=True)
