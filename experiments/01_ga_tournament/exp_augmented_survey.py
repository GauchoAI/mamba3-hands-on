"""
Survey: What does the augmented model do when it succeeds vs fails?

Trains both models, then runs detailed diagnostics on the augmented one:
  - Per-example spike rates (reg + mem) for correct vs incorrect predictions
  - Per-task-type breakdown
  - Register utilization: are all slots used or just a few?
  - Spike timing: when in the sequence do spikes fire? (input vs output portion)
  - Register norm evolution: does content accumulate or stay flat?

The goal: understand what correlates with success BEFORE changing anything.
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

from mamba_platform.mamba3_minimal import Mamba3Block, Mamba3Config
from mamba3_augmented import AugmentedMamba3
from train_bootstrap import (
    SPECIAL_TOKENS, VOCAB_SIZE, tokenize,
    BootstrapDataset, compute_loss, make_fresh_eval_set,
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


# ── Diagnostic forward pass ─────────────────────────────────────────

def diagnostic_forward(model, tokens, sep_positions):
    """
    Run augmented model with full internal logging.
    Returns dict of per-example diagnostics.
    """
    model.eval()
    B, L = tokens.shape
    block = model.block  # AugmentedMamba3

    with torch.no_grad():
        x = model.embed(tokens)
        D = x.shape[-1]

        # SSM pass
        ssm_out = block.ssm(x)

        # Manual register/memory loop with logging
        reg_state = x.new_zeros(B, block.registers.n_registers, D)
        mem_state = x.new_zeros(B, block.memory.n_slots, D)

        # Per-example, per-step logs
        reg_gates_log = []   # (L, B)
        mem_gates_log = []   # (L, B)
        reg_addr_log = []    # (L, B, n_reg)
        reg_norms_log = []   # (L, B) — total register bank norm
        mem_norms_log = []   # (L, B) — total memory bank norm

        outputs = []
        for t in range(L):
            h_t = ssm_out[:, t]

            # Register ops
            reg_read, reg_state, reg_gate = block.registers(h_t, reg_state)
            # Memory ops
            mem_read, mem_state, mem_gate = block.memory(h_t, mem_state)

            combined = block.combine(torch.cat([h_t, reg_read, mem_read], dim=-1))
            combined = block.norm(combined + h_t)
            outputs.append(combined)

            # Log
            reg_gates_log.append(reg_gate.cpu())          # (B,)
            mem_gates_log.append(mem_gate.cpu())           # (B,)
            # Capture write address distribution
            addr = F.softmax(block.registers.write_addr(h_t), dim=-1)
            reg_addr_log.append(addr.cpu())                # (B, n_reg)
            reg_norms_log.append(reg_state.norm(dim=-1).sum(dim=-1).cpu())  # (B,)
            mem_norms_log.append(mem_state.norm(dim=-1).sum(dim=-1).cpu())  # (B,)

        logits = model.head(torch.stack(outputs, dim=1))

    # Stack logs: (L, B) → (B, L)
    reg_gates = torch.stack(reg_gates_log, dim=0).T     # (B, L)
    mem_gates = torch.stack(mem_gates_log, dim=0).T
    reg_addrs = torch.stack(reg_addr_log, dim=0).permute(1, 0, 2)  # (B, L, n_reg)
    reg_norms = torch.stack(reg_norms_log, dim=0).T
    mem_norms = torch.stack(mem_norms_log, dim=0).T

    # Check correctness per example
    results = []
    for i in range(B):
        sep = sep_positions[i]
        # Find output tokens (after SEP, before PAD)
        correct = True
        n_output_tokens = 0
        for t in range(sep, L - 1):
            target = tokens[i, t + 1].item()
            if target == SPECIAL_TOKENS["<PAD>"]:
                break
            pred = logits[i, t].argmax().item()
            n_output_tokens += 1
            if pred != target:
                correct = False
                break

        # Compute diagnostics for this example
        # Split spike rates: input portion (0..sep) vs output portion (sep+1..end)
        input_reg_spikes = reg_gates[i, :sep+1].mean().item()
        output_reg_spikes = reg_gates[i, sep+1:].mean().item() if sep+1 < L else 0
        input_mem_spikes = mem_gates[i, :sep+1].mean().item()
        output_mem_spikes = mem_gates[i, sep+1:].mean().item() if sep+1 < L else 0

        # Register address entropy (higher = more spread across registers)
        avg_addr = reg_addrs[i].mean(dim=0)  # (n_reg,)
        addr_entropy = -(avg_addr * (avg_addr + 1e-8).log()).sum().item()

        # Peak register utilization (which register has most weight)
        peak_reg = avg_addr.argmax().item()
        peak_reg_weight = avg_addr.max().item()

        # Register/memory norm at end of sequence
        final_reg_norm = reg_norms[i, -1].item()
        final_mem_norm = mem_norms[i, -1].item()

        results.append({
            "correct": correct,
            "input_reg_spike_rate": input_reg_spikes,
            "output_reg_spike_rate": output_reg_spikes,
            "total_reg_spike_rate": reg_gates[i].mean().item(),
            "input_mem_spike_rate": input_mem_spikes,
            "output_mem_spike_rate": output_mem_spikes,
            "total_mem_spike_rate": mem_gates[i].mean().item(),
            "reg_addr_entropy": addr_entropy,
            "peak_register": peak_reg,
            "peak_reg_weight": peak_reg_weight,
            "final_reg_norm": final_reg_norm,
            "final_mem_norm": final_mem_norm,
            "n_output_tokens": n_output_tokens,
            "seq_len": L,
        })

    return results


# ── Training (same as exp_augmented_vs_plain) ────────────────────────

def train_model(model, name, dataset, device, steps=3000, batch=64, lr=3e-3,
                eval_every=500, data_refresh_every=750):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}", flush=True)
    print(f"Training: {name} ({n_params:,} params)", flush=True)
    print(f"{'='*60}", flush=True)

    best_fresh = 0.0
    import generators.level0_patterns as gen0

    for step in range(1, steps + 1):
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
            from train_bootstrap import evaluate_on_examples
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

    return best_fresh


# ── Survey ───────────────────────────────────────────────────────────

def run_survey(model, device, n_examples=1000):
    """Run diagnostic on fresh examples, grouped by task type."""
    print(f"\n{'='*60}", flush=True)
    print(f"DIAGNOSTIC SURVEY ({n_examples} fresh examples)", flush=True)
    print(f"{'='*60}", flush=True)

    fresh = make_fresh_eval_set(n_examples)

    # Group by type
    by_type = defaultdict(list)
    for ex in fresh:
        by_type[ex[2]].append(ex)

    all_results = []

    for task_type in sorted(by_type.keys()):
        examples = by_type[task_type]
        # Build batch
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

        # Run diagnostic in sub-batches
        results = []
        bs = 64
        for start in range(0, len(examples), bs):
            end = min(start + bs, len(examples))
            batch_tokens = tokens[start:end]
            batch_seps = sep_positions[start:end]
            batch_results = diagnostic_forward(model, batch_tokens, batch_seps)
            for j, r in enumerate(batch_results):
                r["task_type"] = task_type
            results.extend(batch_results)

        all_results.extend(results)

        # Summarize this type
        correct = [r for r in results if r["correct"]]
        wrong = [r for r in results if not r["correct"]]
        acc = len(correct) / len(results)

        def avg(lst, key):
            return sum(r[key] for r in lst) / max(len(lst), 1)

        print(f"\n  {task_type}: {acc:.0%} ({len(correct)}/{len(results)})", flush=True)

        if correct:
            print(f"    CORRECT examples:", flush=True)
            print(f"      reg_spike: input={avg(correct, 'input_reg_spike_rate'):.3f}  "
                  f"output={avg(correct, 'output_reg_spike_rate'):.3f}  "
                  f"total={avg(correct, 'total_reg_spike_rate'):.3f}", flush=True)
            print(f"      mem_spike: input={avg(correct, 'input_mem_spike_rate'):.3f}  "
                  f"output={avg(correct, 'output_mem_spike_rate'):.3f}  "
                  f"total={avg(correct, 'total_mem_spike_rate'):.3f}", flush=True)
            print(f"      reg_entropy={avg(correct, 'reg_addr_entropy'):.3f}  "
                  f"final_reg_norm={avg(correct, 'final_reg_norm'):.1f}  "
                  f"final_mem_norm={avg(correct, 'final_mem_norm'):.1f}", flush=True)

        if wrong:
            print(f"    WRONG examples:", flush=True)
            print(f"      reg_spike: input={avg(wrong, 'input_reg_spike_rate'):.3f}  "
                  f"output={avg(wrong, 'output_reg_spike_rate'):.3f}  "
                  f"total={avg(wrong, 'total_reg_spike_rate'):.3f}", flush=True)
            print(f"      mem_spike: input={avg(wrong, 'input_mem_spike_rate'):.3f}  "
                  f"output={avg(wrong, 'output_mem_spike_rate'):.3f}  "
                  f"total={avg(wrong, 'total_mem_spike_rate'):.3f}", flush=True)
            print(f"      reg_entropy={avg(wrong, 'reg_addr_entropy'):.3f}  "
                  f"final_reg_norm={avg(wrong, 'final_reg_norm'):.1f}  "
                  f"final_mem_norm={avg(wrong, 'final_mem_norm'):.1f}", flush=True)

        if correct and wrong:
            # Delta analysis
            d_reg = avg(correct, 'total_reg_spike_rate') - avg(wrong, 'total_reg_spike_rate')
            d_mem = avg(correct, 'total_mem_spike_rate') - avg(wrong, 'total_mem_spike_rate')
            d_ent = avg(correct, 'reg_addr_entropy') - avg(wrong, 'reg_addr_entropy')
            print(f"    Δ(correct-wrong): reg_spike={d_reg:+.3f}  mem_spike={d_mem:+.3f}  "
                  f"entropy={d_ent:+.3f}", flush=True)

    # Global summary
    print(f"\n{'='*60}", flush=True)
    print(f"GLOBAL SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    correct_all = [r for r in all_results if r["correct"]]
    wrong_all = [r for r in all_results if not r["correct"]]
    print(f"  Total: {len(correct_all)}/{len(all_results)} correct "
          f"({len(correct_all)/len(all_results):.1%})", flush=True)

    def avg(lst, key):
        return sum(r[key] for r in lst) / max(len(lst), 1)

    for label, subset in [("CORRECT", correct_all), ("WRONG", wrong_all)]:
        if not subset:
            continue
        print(f"\n  {label} ({len(subset)} examples):", flush=True)
        print(f"    reg_spike: input={avg(subset, 'input_reg_spike_rate'):.3f}  "
              f"output={avg(subset, 'output_reg_spike_rate'):.3f}", flush=True)
        print(f"    mem_spike: input={avg(subset, 'input_mem_spike_rate'):.3f}  "
              f"output={avg(subset, 'output_mem_spike_rate'):.3f}", flush=True)
        print(f"    reg_entropy={avg(subset, 'reg_addr_entropy'):.3f}  "
              f"peak_reg_weight={avg(subset, 'peak_reg_weight'):.3f}", flush=True)
        print(f"    final_reg_norm={avg(subset, 'final_reg_norm'):.1f}  "
              f"final_mem_norm={avg(subset, 'final_mem_norm'):.1f}", flush=True)

    # Register slot usage distribution
    print(f"\n  Register slot preference (peak register counts):", flush=True)
    from collections import Counter
    peak_counts = Counter(r["peak_register"] for r in all_results)
    for reg_id in range(8):
        count = peak_counts.get(reg_id, 0)
        bar = "█" * (count * 40 // len(all_results))
        print(f"    reg[{reg_id}]: {count:4d} ({count/len(all_results):.0%}) {bar}", flush=True)

    # Save raw results for further analysis
    out_path = Path("survey_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Raw results saved to {out_path}", flush=True)

    return all_results


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    cfg = Mamba3Config(d_model=64, d_state=16, expand=2, headdim=16)
    steps = 1500  # shorter — just enough to learn something

    # Train augmented only
    dataset = BootstrapDataset("data/level0/patterns.jsonl", device)
    torch.manual_seed(42)
    aug = AugmentedModel(cfg, n_registers=8, n_memory=16).to(device)
    aug_best = train_model(aug, "AUGMENTED", dataset, device, steps=steps,
                           eval_every=500)

    n_aug = sum(p.numel() for p in aug.parameters())
    print(f"\n  Augmented: best fresh={aug_best:.1%}  ({n_aug:,} params)", flush=True)

    # Survey on 200 fresh examples — enough to see patterns, not overwhelming
    survey_results = run_survey(aug, device, n_examples=200)
