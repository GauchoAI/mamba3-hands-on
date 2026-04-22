"""
Turbo Worker: tournament-based config search for one task.

Runs multiple model configs in parallel (single process, for-loop over models).
Every few cycles, kills the worst performers and mutates from the best.
When ANY config hits target accuracy → mastery. Reports winning config.

Uses Triton for GPU-native SSM scan when available.

Usage:
    python turbo_worker.py --task parity
    python turbo_worker.py --task parity --target-acc 0.95 --vram-budget 56000
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import random
import signal
import torch
from pathlib import Path
from collections import defaultdict

from progressive_model import ProgressiveModel, ByteTokenizer, VOCAB_SIZE, PAD
from specialist_trainer import (
    load_generators, get_loss_fn, write_metrics, precompute_teacher_cache,
)
from coordinator import smart_mutate_config, MutationHistory
from grokking import PerpGradOptimizer
from strategies import Lion, WarmRestartScheduler, inject_noise


# ── Memory estimation ──────────────────────────────────────────────

def estimate_vram_mb(cfg, seq_len=64):
    """Estimate VRAM usage for one config (model + optimizer + activations)."""
    d = cfg.get("d_model", 64)
    L = cfg.get("n_kernel_layers", 3)
    bs = cfg.get("batch_size", 256)
    d_inner = d * 2
    headdim = min(cfg.get("headdim", 16), d)
    nheads = d_inner // max(headdim, 1)
    d_state = cfg.get("d_state", 16)

    d_in_proj = 2 * d_inner + 2 * d_state + 3 * nheads + d_state // 2
    layer_params = d * d_in_proj + d_inner * d + d
    total_params = L * layer_params + 260 * d * 2

    param_mem = total_params * 4
    grad_mem = total_params * 4
    opt_mem = total_params * 8  # AdamW: m + v
    act_mem = bs * seq_len * d * L * 4 * 2

    return (param_mem + grad_mem + opt_mem + act_mem) / (1024 * 1024)


# ── Contestant: one model + optimizer + state ──────────────────────

class Contestant:
    """One config racing in the tournament."""

    def __init__(self, idx, cfg, device):
        self.idx = idx
        self.cfg = cfg
        self.device = device
        self.cycle = 0
        self.best_acc = 0.0
        self.best_at_cycle = 0
        self.last_acc = 0.0
        self.last_loss = 0.0
        self.alive = True
        self.parent_idx = cfg.pop("_parent_idx", None)

        # Create model
        self.model = ProgressiveModel(
            d_model=cfg.get("d_model", 64),
            d_state=cfg.get("d_state", 16),
            expand=2,
            headdim=min(cfg.get("headdim", 16), cfg.get("d_model", 64)),
        ).to(device)
        for _ in range(cfg.get("n_kernel_layers", 3)):
            self.model.add_kernel_layer()
        self.model.set_mode("kernel")

        self.n_params = sum(p.numel() for p in self.model.parameters())

        # Optimizer
        wd = cfg.get("weight_decay", 0.0)
        lr = cfg.get("lr", 1e-3)
        if cfg.get("optimizer") == "lion":
            self.opt = Lion(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        # Strategy flags
        use_perp = cfg.get("use_perp", wd == 0.0)
        self.perp = PerpGradOptimizer(self.model) if use_perp else None
        self.scheduler = WarmRestartScheduler(self.opt, T_0=100) if cfg.get("warm_restarts") else None
        self.noise = cfg.get("noise_scale", 0.0)
        self.loss_fn = get_loss_fn(cfg.get("loss_fn", "ce"))
        self.batch_size = cfg.get("batch_size", 256)
        self.steps_per_cycle = cfg.get("steps_per_cycle", 200)

    def tag(self):
        c = self.cfg
        method = "perp" if self.perp else f"wd={c.get('weight_decay', 0)}"
        return (f"d={c.get('d_model')} L={c.get('n_kernel_layers')} "
                f"lr={c.get('lr', 1e-3):.0e} {c.get('optimizer', 'adamw')} "
                f"{c.get('loss_fn', 'ce')} [{method}]")

    def train_cycle(self, examples, tok, device):
        """One training cycle on pre-generated examples."""
        self.model.train()
        cycle_loss = 0.0

        for step in range(self.steps_per_cycle):
            indices = torch.randint(0, len(examples), (self.batch_size,))
            max_len = 0
            batch = []
            for i in indices:
                tokens, sep = examples[i.item()]
                batch.append((tokens, sep))
                max_len = max(max_len, len(tokens))

            token_tensor = torch.full((self.batch_size, max_len), PAD,
                                     dtype=torch.long, device=device)
            sep_positions = []
            for i, (tokens, sep) in enumerate(batch):
                token_tensor[i, :len(tokens)] = torch.tensor(tokens)
                sep_positions.append(sep)

            logits = self.model(token_tensor)
            B, L, V = logits.shape
            pos = torch.arange(L, device=device).unsqueeze(0)
            sep_t = torch.tensor(sep_positions, device=device, dtype=torch.long).unsqueeze(1)
            mask = ((pos >= sep_t) & (pos < L - 1)).float()
            pad_mask = (token_tensor != PAD).float()
            pred_mask = mask[:, :L-1] * pad_mask[:, 1:]

            if pred_mask.sum() > 0:
                logits_flat = logits[:, :L-1].reshape(-1, V)
                targets_flat = token_tensor[:, 1:].reshape(-1)
                mask_flat = pred_mask.reshape(-1)
                loss_all = self.loss_fn(logits_flat, targets_flat, reduction='none')
                loss = (loss_all * mask_flat).sum() / (mask_flat.sum() + 1e-8)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.perp:
                    self.perp.project()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                if self.scheduler:
                    self.scheduler.step()
                cycle_loss += loss.item()

        if self.noise > 0:
            inject_noise(self.model, self.noise)

        self.last_loss = cycle_loss / max(self.steps_per_cycle, 1)
        self.cycle += 1

    def evaluate(self, gen_fn, tok, device, n_eval=100):
        """Evaluate on fresh examples."""
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for _ in range(n_eval):
                ex = gen_fn()
                tokens, sep = tok.encode_curriculum(ex)
                out_bytes = list(ex["output"].encode("utf-8"))
                t = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = self.model(t)
                ok = True
                for j, expected in enumerate(out_bytes):
                    p = sep + j
                    if p < logits.shape[1]:
                        if logits[0, p].argmax().item() != expected:
                            ok = False
                            break
                    else:
                        ok = False
                if ok:
                    correct += 1
                total += 1
        acc = correct / max(total, 1)
        self.last_acc = acc
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_at_cycle = self.cycle
        return acc


# ── Initial config pool ────────────────────────────────────────────

TURBO_CONFIGS = [
    # Grokking-style: small, fast, PerpGrad
    {"d_model": 32, "d_state": 16, "headdim": 16, "n_kernel_layers": 1,
     "lr": 3e-3, "weight_decay": 0.0, "use_perp": True,
     "optimizer": "adamw", "loss_fn": "stable_ce", "batch_size": 512,
     "steps_per_cycle": 200},

    # Proven default (graduated 5 tasks before)
    {"d_model": 64, "d_state": 16, "headdim": 16, "n_kernel_layers": 3,
     "lr": 1e-3, "weight_decay": 0.1, "use_perp": False,
     "optimizer": "adamw", "loss_fn": "ce", "batch_size": 256,
     "steps_per_cycle": 200},

    # Lion optimizer: sign-based, fast for small models
    {"d_model": 64, "d_state": 16, "headdim": 16, "n_kernel_layers": 2,
     "lr": 3e-4, "weight_decay": 0.1, "use_perp": False,
     "optimizer": "lion", "loss_fn": "ce", "batch_size": 256,
     "steps_per_cycle": 200},

    # Big model: more capacity for harder tasks
    {"d_model": 96, "d_state": 16, "headdim": 16, "n_kernel_layers": 4,
     "lr": 5e-4, "weight_decay": 0.1, "use_perp": False,
     "optimizer": "adamw", "loss_fn": "ce", "batch_size": 128,
     "steps_per_cycle": 200},

    # Focal loss: focus on hard examples
    {"d_model": 64, "d_state": 16, "headdim": 16, "n_kernel_layers": 3,
     "lr": 1e-3, "weight_decay": 0.1, "use_perp": False,
     "optimizer": "adamw", "loss_fn": "focal", "batch_size": 256,
     "steps_per_cycle": 200},

    # Small + aggressive LR
    {"d_model": 48, "d_state": 16, "headdim": 16, "n_kernel_layers": 1,
     "lr": 5e-3, "weight_decay": 0.05, "use_perp": False,
     "optimizer": "adamw", "loss_fn": "ce", "batch_size": 512,
     "steps_per_cycle": 200},

    # Label smoothing: anti-overconfidence
    {"d_model": 64, "d_state": 16, "headdim": 16, "n_kernel_layers": 3,
     "lr": 1e-3, "weight_decay": 0.1, "use_perp": False,
     "optimizer": "adamw", "loss_fn": "label_smooth", "batch_size": 256,
     "steps_per_cycle": 200},

    # Warm restarts: escape local minima
    {"d_model": 64, "d_state": 16, "headdim": 16, "n_kernel_layers": 2,
     "lr": 2e-3, "weight_decay": 0.1, "use_perp": False,
     "optimizer": "adamw", "loss_fn": "ce", "batch_size": 256,
     "steps_per_cycle": 200, "warm_restarts": True},
]


# ── Main tournament ────────────────────────────────────────────────

def run_tournament(args):
    task = args.task
    target_acc = args.target_acc
    vram_budget = args.vram_budget
    run_dir = Path(args.run_dir) if args.run_dir else None
    eval_interval = args.eval_interval

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load task generator
    load_generators()
    import specialist_trainer
    gen_fn = specialist_trainer.GENERATORS.get(task)
    if not gen_fn:
        print(f"Unknown task: {task}")
        return
    tok = ByteTokenizer()

    print(f"\n{'='*70}", flush=True)
    print(f"TURBO WORKER — Tournament for: {task}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Target: {target_acc:.0%}", flush=True)
    print(f"  VRAM budget: {vram_budget} MB", flush=True)
    print(f"  Eval every: {eval_interval} cycles", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Select configs that fit in VRAM
    configs = []
    total_vram = 0
    for cfg in TURBO_CONFIGS:
        mem = estimate_vram_mb(cfg)
        if total_vram + mem <= vram_budget:
            configs.append(cfg.copy())
            total_vram += mem
            print(f"  [{len(configs)-1}] {mem:.0f}MB — "
                  f"d={cfg['d_model']} L={cfg['n_kernel_layers']} "
                  f"lr={cfg['lr']:.0e} {cfg.get('optimizer','adamw')} "
                  f"{cfg.get('loss_fn','ce')}"
                  f"{' PerpGrad' if cfg.get('use_perp') else ''}"
                  f"{' WarmRestart' if cfg.get('warm_restarts') else ''}", flush=True)

    print(f"\n  {len(configs)} contestants, {total_vram:.0f}MB / {vram_budget}MB VRAM\n", flush=True)

    # Create contestants
    contestants = []
    for i, cfg in enumerate(configs):
        c = Contestant(i, cfg.copy(), device)
        contestants.append(c)
        print(f"  Created [{i}]: {c.tag()} ({c.n_params:,} params)", flush=True)

    # Mutation history (persists across tournament rounds)
    mut_history = MutationHistory()

    # Graceful shutdown
    should_stop = False
    def handle_signal(sig, frame):
        nonlocal should_stop
        should_stop = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Setup run_dir
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*70}", flush=True)
    print(f"  TOURNAMENT START", flush=True)
    print(f"{'─'*70}\n", flush=True)

    t_start = time.time()
    round_num = 0
    winner = None
    next_idx = len(contestants)

    while not should_stop:
        # ── Generate shared data (once per cycle, all contestants use it) ──
        examples = []
        for _ in range(5000):
            ex = gen_fn()
            tokens, sep = tok.encode_curriculum(ex)
            examples.append((tokens, sep))

        # ── Train each alive contestant for 1 cycle ──
        alive = [c for c in contestants if c.alive]
        for c in alive:
            t0 = time.time()
            c.train_cycle(examples, tok, device)
            elapsed = time.time() - t0

        # ── Evaluate every eval_interval cycles ──
        min_cycle = min(c.cycle for c in alive) if alive else 0
        if min_cycle > 0 and min_cycle % eval_interval == 0:
            round_num += 1
            print(f"\n  ── Round {round_num} (cycle {min_cycle}) "
                  f"── {time.time() - t_start:.0f}s elapsed ──", flush=True)

            # Evaluate all
            for c in alive:
                acc = c.evaluate(gen_fn, tok, device)

            # Sort by accuracy
            alive.sort(key=lambda c: -c.last_acc)

            # Print scoreboard
            for rank, c in enumerate(alive):
                marker = "★" if c.last_acc >= target_acc else " "
                plateau = c.cycle - c.best_at_cycle
                print(f"  {marker} [{c.idx:2d}] acc={c.last_acc:5.0%} "
                      f"best={c.best_acc:5.0%} loss={c.last_loss:.3f} "
                      f"plateau={plateau:3d} | {c.tag()}", flush=True)

            # Check for mastery
            for c in alive:
                if c.last_acc >= target_acc:
                    winner = c
                    break

            if winner:
                break

            # ── Tournament selection ──
            # Early: no killing (let configs explore)
            # Mid: kill bottom 25%
            # Late: kill bottom 50%
            if min_cycle >= 20 and len(alive) > 2:
                if min_cycle < 100:
                    kill_count = max(1, len(alive) // 4)
                else:
                    kill_count = max(1, len(alive) // 2)

                # Kill the worst
                to_kill = alive[-kill_count:]
                for c in to_kill:
                    c.alive = False
                    print(f"    ✗ Killed [{c.idx}] ({c.last_acc:.0%})", flush=True)

                # Determine mutation aggressiveness
                best = alive[0]
                plateau_cycles = best.cycle - best.best_at_cycle
                if plateau_cycles < 20:
                    severity = 0.0
                elif plateau_cycles < 50:
                    severity = 1.0
                else:
                    severity = 2.5

                # Spawn replacements from best
                for _ in range(kill_count):
                    child_cfg = smart_mutate_config(
                        best.cfg, mutation_history=mut_history,
                        plateau_severity=severity,
                    )
                    child_cfg["_parent_idx"] = best.idx

                    # Check VRAM
                    child_mem = estimate_vram_mb(child_cfg)
                    if child_mem > 500:  # sanity cap
                        child_cfg = best.cfg.copy()  # fallback to parent

                    child = Contestant(next_idx, child_cfg, device)
                    contestants.append(child)
                    next_idx += 1

                    changed = {k: child_cfg[k] for k in child_cfg
                              if k not in ("steps_per_cycle",) and
                              child_cfg.get(k) != best.cfg.get(k)}
                    print(f"    + [{child.idx}] from [{best.idx}]: {changed}", flush=True)

                # Record mutation outcomes for history
                for c in alive[1:]:
                    if c.parent_idx is not None:
                        parent = next((p for p in contestants if p.idx == c.parent_idx), None)
                        if parent:
                            mut_history.record(parent.cfg, c.cfg,
                                             parent.best_acc, c.best_acc)

            # Write metrics
            if run_dir:
                best = alive[0] if alive else None
                write_metrics(run_dir, {
                    "cycle": min_cycle,
                    "round": round_num,
                    "task": task,
                    "n_alive": len([c for c in contestants if c.alive]),
                    "best_acc": best.best_acc if best else 0,
                    "best_config": best.cfg if best else {},
                    "acc": best.last_acc if best else 0,
                    "loss": best.last_loss if best else 0,
                    "n_params": best.n_params if best else 0,
                    "elapsed": time.time() - t_start,
                })

    # ── Mastery achieved ──
    elapsed = time.time() - t_start
    if winner:
        print(f"\n{'='*70}", flush=True)
        print(f"  ★ MASTERY: {task} at {winner.last_acc:.0%} in {winner.cycle} cycles "
              f"({elapsed:.0f}s)", flush=True)
        print(f"  Winning config: {winner.tag()}", flush=True)
        print(f"  Config: {json.dumps(winner.cfg, indent=2)}", flush=True)
        print(f"{'='*70}\n", flush=True)

        # Save specialist checkpoint
        ckpt_dir = Path("checkpoints/specialists")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{task}.pt"
        torch.save({
            "model": winner.model.state_dict(),
            "task": task,
            "config": winner.cfg,
            "accuracy": winner.best_acc,
            "cycles": winner.cycle,
            "n_params": winner.n_params,
        }, ckpt_path)
        print(f"  Saved → {ckpt_path}", flush=True)

        # Precompute teacher cache
        print(f"  Precomputing teacher cache...", flush=True)
        teacher_data = precompute_teacher_cache(winner.model, gen_fn, tok, device)
        cache_path = ckpt_dir / f"{task}_cache.pt"
        torch.save(teacher_data, cache_path)
        print(f"  Cached {len(teacher_data)} examples → {cache_path}", flush=True)

        # Write final metrics
        if run_dir:
            write_metrics(run_dir, {
                "cycle": winner.cycle,
                "task": task,
                "mastered": True,
                "acc": winner.last_acc,
                "best_acc": winner.best_acc,
                "config": winner.cfg,
                "n_params": winner.n_params,
                "elapsed": elapsed,
                "rounds": round_num,
            })
    else:
        print(f"\n  Stopped without mastery. Best: "
              f"{max((c.best_acc for c in contestants), default=0):.0%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tournament-based config search")
    parser.add_argument("--task", required=True)
    parser.add_argument("--target-acc", type=float, default=0.95)
    parser.add_argument("--vram-budget", type=int, default=56000)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    run_tournament(args)
