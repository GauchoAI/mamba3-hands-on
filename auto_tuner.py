"""
Auto-tuner: dynamic model shopping with parallel experiments.

Detects available GPU/CPU resources, spawns parallel experiments with
different configurations, watches performance, prunes losers, keeps
winners. Persists results so future runs start from known-good configs.

Features:
  - Resource-aware: fits as many experiments as VRAM allows
  - Teacher cache: generate curriculum once, share across experiments
  - Pruning: kill underperformers after N cycles
  - Persistence: saves (task, config) → convergence_speed to JSON
  - Namespace by curriculum task: different tasks may prefer different configs

Usage:
    python auto_tuner.py                          # auto-detect resources
    python auto_tuner.py --max-experiments 8      # cap parallel runs
    python auto_tuner.py --resume                 # continue from saved results
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import copy
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from progressive_model import ProgressiveModel, ByteTokenizer, VOCAB_SIZE, PAD, SEP
from grokking import stable_cross_entropy, PerpGradOptimizer
from generators.teacher import AdaptiveTeacher


# ── Experiment config ───────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    d_model: int = 64
    d_state: int = 16
    headdim: int = 16
    n_kernel_layers: int = 1
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0

    def name(self):
        return f"d{self.d_model}_s{self.d_state}_L{self.n_kernel_layers}_b{self.batch_size}_lr{self.lr}"


# Default search space
SEARCH_CONFIGS = [
    ExperimentConfig(d_model=32,  d_state=16, headdim=16, n_kernel_layers=1, batch_size=4096),
    ExperimentConfig(d_model=64,  d_state=16, headdim=16, n_kernel_layers=1, batch_size=4096),
    ExperimentConfig(d_model=64,  d_state=16, headdim=16, n_kernel_layers=2, batch_size=4096),
    ExperimentConfig(d_model=128, d_state=16, headdim=16, n_kernel_layers=1, batch_size=2048),
    ExperimentConfig(d_model=32,  d_state=8,  headdim=8,  n_kernel_layers=2, batch_size=8192),
    ExperimentConfig(d_model=64,  d_state=8,  headdim=8,  n_kernel_layers=1, batch_size=8192),
    # Grokking experiment: high weight decay + no PerpGrad, classic grokking recipe
    # Smaller batch intentionally — more gradient steps per data point
    ExperimentConfig(d_model=64,  d_state=16, headdim=16, n_kernel_layers=1, batch_size=128, lr=1e-3, weight_decay=0.1),
    # Grokking + bigger model
    ExperimentConfig(d_model=128, d_state=16, headdim=16, n_kernel_layers=1, batch_size=128, lr=1e-3, weight_decay=0.1),
    # Grokking + deeper
    ExperimentConfig(d_model=64,  d_state=16, headdim=16, n_kernel_layers=2, batch_size=128, lr=1e-3, weight_decay=0.1),
]


# ── Resource detection ──────────────────────────────────────────────

def detect_resources():
    """Detect available compute and memory."""
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name()
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem = total_mem * 0.9  # leave 10% headroom
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        name = "Apple MPS"
        free_mem = 8.0  # conservative estimate
    else:
        device = "cpu"
        name = "CPU"
        free_mem = 4.0
    return device, name, free_mem


def estimate_memory(cfg: ExperimentConfig, seq_len=20) -> float:
    """Estimate peak memory for one experiment in GB."""
    d = cfg.d_model
    n_layers = cfg.n_kernel_layers
    bs = cfg.batch_size
    # Rough estimate: params + activations + gradients + optimizer
    n_params = VOCAB_SIZE * d + n_layers * (d * d * 6) + d * VOCAB_SIZE
    param_mem = n_params * 4 * 3  # weights + grads + optimizer state (Adam: 2x)
    act_mem = bs * seq_len * d * n_layers * 4 * 2  # activations + grad
    total = (param_mem + act_mem) / 1e9
    return max(total, 0.1)  # minimum 100MB


# ── Shared teacher cache ────────────────────────────────────────────

class TeacherCache:
    """Generate curriculum data once, share across experiments.
    Evicts and regenerates when teacher state changes (unlock, difficulty)."""

    def __init__(self, teacher, device, count=10000):
        self.teacher = teacher
        self.device = device
        self.count = count
        self.tok = ByteTokenizer()
        self._cache = None
        self._cache_hash = None

    def _teacher_hash(self):
        """Hash of teacher state to detect changes."""
        unlocked = tuple(sorted(self.teacher.unlocked_tasks))
        diffs = tuple(
            (t, round(c.difficulty, 2))
            for t, c in sorted(self.teacher.task_configs.items())
        )
        return hash((unlocked, diffs))

    def get_data(self):
        """Get cached curriculum data, regenerate if teacher changed."""
        h = self._teacher_hash()
        if self._cache is None or h != self._cache_hash:
            raw = self.teacher.generate(self.count)
            self._cache = []
            for ex in raw:
                tokens, sep_pos = self.tok.encode_curriculum(ex)
                self._cache.append((tokens, sep_pos, ex.get("type", "")))
            self._cache_hash = h
        return self._cache

    def get_batch(self, batch_size):
        """Sample a batch from cached data."""
        data = self.get_data()
        indices = torch.randint(0, len(data), (batch_size,))
        max_len = 0
        batch = []
        for idx in indices:
            tokens, sep_pos, _ = data[idx.item()]
            batch.append((tokens, sep_pos))
            max_len = max(max_len, len(tokens))

        token_tensor = torch.full((batch_size, max_len), PAD,
                                  dtype=torch.long, device=self.device)
        sep_positions = []
        for i, (tokens, sep_pos) in enumerate(batch):
            token_tensor[i, :len(tokens)] = torch.tensor(tokens)
            sep_positions.append(sep_pos)
        return token_tensor, sep_positions


# ── Single experiment ───────────────────────────────────────────────

class Experiment:
    """One running experiment with its own model and optimizer."""

    def __init__(self, cfg: ExperimentConfig, device):
        self.cfg = cfg
        self.device = device
        self.model = ProgressiveModel(
            d_model=cfg.d_model, d_state=cfg.d_state,
            expand=2, headdim=cfg.headdim,
        ).to(device)
        for _ in range(cfg.n_kernel_layers):
            self.model.add_kernel_layer()
        self.model.set_mode("kernel")

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        # Use PerpGrad only when weight_decay is 0 (paper approach)
        # Grokking experiment uses weight_decay instead
        self.use_perp = cfg.weight_decay == 0.0
        self.perp = PerpGradOptimizer(self.model) if self.use_perp else None
        self.n_params = self.model.total_params()

        # Tracking
        self.cycle = 0
        self.best_acc = 0.0
        self.history = []  # (cycle, fresh_acc, loss)
        self.alive = True

        # Per-experiment log file
        log_path = Path(f"logs/tuner_{cfg.name()}.log")
        log_path.parent.mkdir(exist_ok=True)
        self.log = open(log_path, "w")
        self.log.write(f"Experiment: {cfg.name()}\n")
        self.log.write(f"Params: {self.n_params:,}\n")
        self.log.write(f"Config: d_model={cfg.d_model} d_state={cfg.d_state} "
                       f"layers={cfg.n_kernel_layers} batch={cfg.batch_size} "
                       f"lr={cfg.lr} wd={cfg.weight_decay}\n")
        self.log.write(f"{'='*60}\n")
        self.log.flush()

    def _compile_train_step(self):
        """Compile the forward+loss into a single fused function."""
        @torch.compile(mode="reduce-overhead", disable=not torch.cuda.is_available())
        def _fwd_loss(model, tokens, sep_tensor, V):
            logits = model(tokens)
            B, L, _ = logits.shape
            # Vectorized mask: positions >= sep and < L-1
            pos = torch.arange(L, device=tokens.device).unsqueeze(0)  # (1, L)
            mask = (pos >= sep_tensor.unsqueeze(1)) & (pos < L - 1)   # (B, L)
            mask = mask.float()
            pad_mask = (tokens != PAD).float()
            target_valid = pad_mask[:, 1:]
            pred_mask = mask[:, :L-1] * target_valid
            logits_flat = logits[:, :L-1].reshape(-1, V)
            targets_flat = tokens[:, 1:].reshape(-1)
            mask_flat = pred_mask.reshape(-1)
            loss_all = stable_cross_entropy(logits_flat, targets_flat, reduction='none')
            loss = (loss_all * mask_flat).sum() / (mask_flat.sum() + 1e-8)
            return loss
        return _fwd_loss

    def train_cycle(self, cache: TeacherCache, steps=500):
        """Train for one cycle using shared cache."""
        self.model.train()
        last_loss = 0.0

        # Lazy compile on first call
        if not hasattr(self, '_train_fn'):
            self._train_fn = self._compile_train_step()

        for _ in range(steps):
            tokens, sep_pos = cache.get_batch(self.cfg.batch_size)
            sep_tensor = torch.tensor(sep_pos, device=self.device, dtype=torch.long)

            loss = self._train_fn(self.model, tokens, sep_tensor, VOCAB_SIZE)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.perp:
                self.perp.project()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            last_loss = loss.item()

        self.cycle += 1
        lr = self.opt.param_groups[0]["lr"]
        method = "PerpGrad" if self.use_perp else f"wd={self.cfg.weight_decay}"
        elapsed = (time.time() - t0) if 't0' in dir() else 0
        self.log.write(f"  step {self.cycle * steps:5d}  loss={last_loss:.4f}  "
                       f"lr={lr:.1e}  [{method}]  {self.n_params:,}p\n")
        self.log.flush()
        return last_loss

    def evaluate(self, cache: TeacherCache):
        """Quick eval on fresh data."""
        from generators.level0_patterns import generate_dataset
        tok = ByteTokenizer()
        raw = generate_dataset(200)
        by_type = defaultdict(list)
        for ex in raw:
            by_type[ex["type"]].append(ex)

        type_accs = {}
        self.model.eval()
        with torch.no_grad():
            for task_type, examples in by_type.items():
                correct = 0
                total = 0
                for ex in examples[:30]:
                    tokens, sep_pos = tok.encode_curriculum(ex)
                    out_bytes = list(ex["output"].encode("utf-8"))
                    t = torch.tensor([tokens], dtype=torch.long, device=self.device)
                    logits = self.model(t)
                    ok = True
                    for j, expected in enumerate(out_bytes):
                        pos = sep_pos + j
                        if pos >= logits.shape[1]:
                            ok = False
                            break
                        if logits[0, pos].argmax().item() != expected:
                            ok = False
                            break
                    if ok:
                        correct += 1
                    total += 1
                type_accs[task_type] = correct / max(total, 1)

        self.model.train()
        # Overall fresh
        fresh = sum(type_accs.values()) / max(len(type_accs), 1)
        self.best_acc = max(self.best_acc, fresh)
        self.history.append((self.cycle, fresh, type_accs))

        # Log in curriculum style
        self.log.write(f"  fresh={fresh:.1%}  best={self.best_acc:.1%}  "
                       f"gap={1.0 - fresh:+.1%}\n")
        for t, a in sorted(type_accs.items()):
            if a > 0:
                status = "✓" if a >= 0.90 else ("…" if a >= 0.40 else "✗")
                self.log.write(f"    {status} {t}: {a:.0%}\n")
        self.log.flush()

        return fresh, type_accs


# ── Tuning results persistence ──────────────────────────────────────

class TuningResults:
    """Persist (task, config) → performance mapping."""

    def __init__(self, path="tuning_results.json"):
        self.path = Path(path)
        self.results = {}
        if self.path.exists():
            with open(self.path) as f:
                self.results = json.load(f)

    def record(self, cfg: ExperimentConfig, task: str, cycles_to_master: int,
               final_acc: float):
        key = cfg.name()
        if key not in self.results:
            self.results[key] = {"config": asdict(cfg), "tasks": {}}
        self.results[key]["tasks"][task] = {
            "cycles_to_master": cycles_to_master,
            "final_acc": final_acc,
        }
        self.save()

    def get_best_config(self, task: str) -> dict | None:
        """Return the config that mastered a task fastest."""
        best = None
        best_cycles = float("inf")
        for key, data in self.results.items():
            if task in data.get("tasks", {}):
                c = data["tasks"][task]["cycles_to_master"]
                if c < best_cycles:
                    best_cycles = c
                    best = data["config"]
        return best

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.results, f, indent=2)

    def summary(self) -> str:
        lines = ["Tuning results:"]
        for key, data in self.results.items():
            tasks = data.get("tasks", {})
            task_str = ", ".join(f"{t}: {d['cycles_to_master']}cyc/{d['final_acc']:.0%}"
                                for t, d in tasks.items())
            lines.append(f"  {key}: {task_str}")
        return "\n".join(lines)


# ── Main auto-tuner ─────────────────────────────────────────────────

def run_tuner(args):
    device, device_name, free_mem = detect_resources()
    print(f"Device: {device_name} ({free_mem:.0f}GB available)", flush=True)

    # Select configs that fit in memory
    configs = SEARCH_CONFIGS
    if args.configs:
        # Custom configs from JSON file
        with open(args.configs) as f:
            configs = [ExperimentConfig(**c) for c in json.load(f)]

    # Estimate memory and select experiments
    experiments = []
    total_mem = 0
    for cfg in configs:
        mem = estimate_memory(cfg)
        if total_mem + mem <= free_mem and len(experiments) < args.max_experiments:
            experiments.append(cfg)
            total_mem += mem
            print(f"  {cfg.name()}: ~{mem:.1f}GB, {cfg.n_kernel_layers}L, "
                  f"{cfg.d_model}d", flush=True)
        else:
            print(f"  {cfg.name()}: skipped (would exceed memory)", flush=True)

    print(f"\nRunning {len(experiments)} experiments using ~{total_mem:.1f}GB", flush=True)

    # Shared teacher + cache
    teacher = AdaptiveTeacher(sequential_unlock=True)
    cache = TeacherCache(teacher, device)

    # Create experiments
    exps = [Experiment(cfg, device) for cfg in experiments]
    tuning = TuningResults()

    print(f"\nExperiments:", flush=True)
    for exp in exps:
        print(f"  {exp.cfg.name()}: {exp.n_params:,} params", flush=True)
    print(flush=True)

    # Training loop
    for cycle in range(args.total_cycles):
        t0 = time.time()

        # Refresh cache if needed (teacher state may change)
        cache.get_data()

        # Train all alive experiments
        # Sequential but with torch.cuda.synchronize() only at the end
        for exp in exps:
            if not exp.alive:
                continue
            exp.train_cycle(cache, steps=args.steps_per_cycle)

        # Evaluate all
        results = []
        for exp in exps:
            if not exp.alive:
                continue
            fresh, type_accs = exp.evaluate(cache)
            results.append((exp, fresh, type_accs))

        # Sort by performance
        results.sort(key=lambda x: -x[1])

        elapsed = time.time() - t0

        # Print status
        print(f"\n[Cycle {cycle+1}] ({elapsed:.1f}s)", flush=True)
        for exp, fresh, type_accs in results:
            parity_acc = type_accs.get("parity", 0)
            marker = "★" if exp == results[0][0] else " "
            status = "ALIVE" if exp.alive else "DEAD"
            lr = exp.opt.param_groups[0]["lr"]
            wd = exp.cfg.weight_decay
            perp_tag = "perp" if exp.use_perp else f"wd={wd}"
            print(f"  {marker} {exp.cfg.name()}: fresh={fresh:.1%}  "
                  f"parity={parity_acc:.0%}  best={exp.best_acc:.1%}  "
                  f"lr={lr:.0e}  [{perp_tag}]  {exp.n_params:,}p", flush=True)

            # Show per-type details for the leader
            if exp == results[0][0]:
                for t, a in sorted(type_accs.items()):
                    if a > 0:
                        print(f"      {t}: {a:.0%}", flush=True)

        # Update teacher with best experiment's results
        best_exp, best_fresh, best_type_accs = results[0]
        teacher.set_step(cycle * args.steps_per_cycle)
        teacher.observe(best_type_accs)
        if cycle % 5 == 0:
            print(f"  teacher:\n{teacher.get_status()}", flush=True)
            if teacher.mastery_log:
                print(teacher.get_learning_report(), flush=True)

        # Prune: kill bottom half after warmup
        if cycle >= args.prune_after and cycle % args.prune_every == 0:
            n_alive = sum(1 for e in exps if e.alive)
            if n_alive > 1:
                cutoff = max(1, n_alive // 2)
                alive_sorted = [(e, f) for e, f, _ in results if e.alive]
                for e, f in alive_sorted[cutoff:]:
                    e.alive = False
                    # Free memory
                    del e.model
                    del e.opt
                    torch.cuda.empty_cache() if device == "cuda" else None
                    print(f"  ✗ Pruned {e.cfg.name()} (fresh={f:.1%})", flush=True)

        # Update dashboard
        from dashboard import generate_dashboard
        generate_dashboard(exps, teacher, cycle + 1)

        # Record results for mastered tasks
        for exp, fresh, type_accs in results:
            if not exp.alive:
                continue
            for task, acc in type_accs.items():
                if acc >= 0.90:
                    tuning.record(exp.cfg, task, exp.cycle, acc)

        # Checkpoint best
        if cycle % 10 == 0:
            best = results[0][0]
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({
                "model": best.model.state_dict(),
                "config": asdict(best.cfg),
                "cycle": cycle,
                "best_fresh": best.best_acc,
                "teacher": teacher.to_dict(),
            }, ckpt_dir / "tuner_best.pt")
            tuning.save()

    # Final report
    print(f"\n{'='*60}", flush=True)
    print(f"TUNING COMPLETE — {args.total_cycles} cycles", flush=True)
    print(f"{'='*60}", flush=True)
    print(tuning.summary(), flush=True)

    alive = [(e, e.best_acc) for e in exps if e.alive]
    alive.sort(key=lambda x: -x[1])
    if alive:
        winner = alive[0][0]
        print(f"\nWinner: {winner.cfg.name()} — {winner.n_params:,} params, "
              f"best_fresh={winner.best_acc:.1%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-cycles", type=int, default=100)
    parser.add_argument("--steps-per-cycle", type=int, default=200)
    parser.add_argument("--max-experiments", type=int, default=8)
    parser.add_argument("--prune-after", type=int, default=10,
                        help="Start pruning after this many cycles")
    parser.add_argument("--prune-every", type=int, default=10,
                        help="Prune every N cycles")
    parser.add_argument("--configs", type=str, default=None,
                        help="Path to JSON file with custom configs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_tuner(args)
