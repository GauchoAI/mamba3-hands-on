"""
Worker: standalone training process. Writes metrics to filesystem.
Launched by coordinator. Knows nothing about other workers.

Usage:
    python worker.py --run-dir runs/exp_001 --config runs/exp_001/config.json
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import signal
import torch
from pathlib import Path
from collections import defaultdict

from progressive_model import ProgressiveModel, ByteTokenizer, VOCAB_SIZE, PAD
from grokking import stable_cross_entropy, PerpGradOptimizer
from generators.teacher import AdaptiveTeacher
from metrics_db import MetricsWriter


def load_config(path):
    with open(path) as f:
        return json.load(f)


def write_metrics(run_dir, metrics):
    """Atomically write metrics so coordinator never reads partial data."""
    tmp = run_dir / "metrics.tmp"
    final = run_dir / "metrics.json"
    with open(tmp, "w") as f:
        json.dump(metrics, f, indent=2)
    tmp.rename(final)


def write_status(run_dir, status):
    (run_dir / "status").write_text(status)


def train(args):
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Model
    model = ProgressiveModel(
        d_model=cfg["d_model"], d_state=cfg["d_state"],
        expand=2, headdim=cfg["headdim"],
    ).to(device)
    for _ in range(cfg.get("n_kernel_layers", 1)):
        model.add_kernel_layer()
    model.set_mode("kernel")

    # Optimizer
    wd = cfg.get("weight_decay", 0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=wd)
    use_perp = wd == 0.0
    perp = PerpGradOptimizer(model) if use_perp else None

    # Teacher
    teacher = AdaptiveTeacher(sequential_unlock=True)
    tok = ByteTokenizer()

    # Metrics DB
    db_path = str(run_dir.parent.parent / "metrics.db")
    metrics = MetricsWriter(db_path)
    exp_id = run_dir.name
    metrics.register_experiment(exp_id, cfg, model.total_params())

    # Resume if checkpoint exists
    ckpt_path = run_dir / "checkpoint.pt"
    cycle = 0
    best_fresh = 0.0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        cycle = ckpt.get("cycle", 0)
        best_fresh = ckpt.get("best_fresh", 0.0)
        if "teacher" in ckpt:
            teacher = AdaptiveTeacher.from_dict(ckpt["teacher"])
        if "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass

    # Log file
    log = open(run_dir / "train.log", "a")
    log.write(f"\n{'='*60}\n")
    log.write(f"Worker started: {cfg}\n")
    log.write(f"Params: {model.total_params():,}  device={device}\n")
    log.write(f"Resuming from cycle {cycle}\n" if cycle > 0 else "Starting fresh\n")
    log.flush()

    write_status(run_dir, "running")

    # Graceful shutdown
    should_stop = False
    def handle_signal(sig, frame):
        nonlocal should_stop
        should_stop = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    steps_per_cycle = cfg.get("steps_per_cycle", 200)

    # Training loop
    while not should_stop:
        cycle += 1
        t0 = time.time()

        # Generate data
        raw = teacher.generate(5000)
        examples = []
        for ex in raw:
            tokens, sep_pos = tok.encode_curriculum(ex)
            examples.append((tokens, sep_pos))

        # Train
        model.train()
        cycle_loss = 0.0
        for step in range(steps_per_cycle):
            idx = torch.randint(0, len(examples), (cfg["batch_size"],))
            max_len = 0
            batch = []
            for i in idx:
                tokens, sep_pos = examples[i.item()]
                batch.append((tokens, sep_pos))
                max_len = max(max_len, len(tokens))

            token_tensor = torch.full((cfg["batch_size"], max_len), PAD,
                                     dtype=torch.long, device=device)
            sep_positions = []
            for i, (tokens, sep_pos) in enumerate(batch):
                token_tensor[i, :len(tokens)] = torch.tensor(tokens)
                sep_positions.append(sep_pos)

            logits = model(token_tensor)
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
                loss_all = stable_cross_entropy(logits_flat, targets_flat, reduction='none')
                loss = (loss_all * mask_flat).sum() / (mask_flat.sum() + 1e-8)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if perp:
                    perp.project()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                cycle_loss += loss.item()

        cycle_loss /= max(steps_per_cycle, 1)
        elapsed = time.time() - t0

        # Evaluate
        from generators.level0_patterns import generate_dataset
        eval_raw = generate_dataset(200)
        by_type = defaultdict(list)
        for ex in eval_raw:
            by_type[ex["type"]].append(ex)

        type_accs = {}
        model.eval()
        with torch.no_grad():
            for task_type, task_examples in by_type.items():
                correct = 0
                total = 0
                for ex in task_examples[:30]:
                    tokens, sep_pos = tok.encode_curriculum(ex)
                    out_bytes = list(ex["output"].encode("utf-8"))
                    t = torch.tensor([tokens], dtype=torch.long, device=device)
                    logits = model(t)
                    ok = True
                    for j, expected in enumerate(out_bytes):
                        p = sep_pos + j
                        if p >= logits.shape[1] or logits[0, p].argmax().item() != expected:
                            ok = False
                            break
                    if ok:
                        correct += 1
                    total += 1
                type_accs[task_type] = correct / max(total, 1)
        model.train()

        fresh = sum(type_accs.values()) / max(len(type_accs), 1)
        best_fresh = max(best_fresh, fresh)

        # Update teacher
        prev_mastery_count = len(teacher.mastery_log)
        teacher.set_step(cycle * steps_per_cycle)
        teacher.observe(type_accs)

        # Log mastery events
        if len(teacher.mastery_log) > prev_mastery_count:
            for entry in teacher.mastery_log[prev_mastery_count:]:
                metrics.log_event("mastery", exp_id,
                    f"{entry['task']} mastered in {entry['steps_to_master']} steps")

        # Log unlock events
        for t in teacher.unlocked_tasks:
            if t not in type_accs:  # newly unlocked
                metrics.log_event("unlock", exp_id, f"{t} unlocked")

        # Log
        parity = type_accs.get("parity", 0)
        same_diff = type_accs.get("same_different", 0)
        method = "PerpGrad" if use_perp else f"wd={wd}"
        log.write(f"  cycle {cycle:4d}  loss={cycle_loss:.4f}  fresh={fresh:.1%}  "
                  f"parity={parity:.0%}  same_diff={same_diff:.0%}  "
                  f"{elapsed:.1f}s  [{method}]\n")
        for t, a in sorted(type_accs.items()):
            if a > 0:
                status = "✓" if a >= 0.90 else ("…" if a >= 0.40 else "✗")
                log.write(f"    {status} {t}: {a:.0%}\n")
        if teacher.mastery_log:
            log.write(teacher.get_learning_report() + "\n")
        log.flush()

        # Write to SQLite timeseries
        metrics.log_cycle(exp_id, cycle, cycle_loss, fresh, best_fresh,
                         elapsed_s=elapsed)
        # Include difficulty levels from teacher
        difficulties = {}
        for t, cfg_t in teacher.task_configs.items():
            difficulties[t] = cfg_t.difficulty
        metrics.log_tasks(exp_id, cycle, type_accs, difficulties=difficulties)
        if cycle % 10 == 0:
            metrics.log_teacher(exp_id, cycle, teacher)

        # Write metrics JSON for coordinator (legacy)
        write_metrics(run_dir, {
            "cycle": cycle,
            "loss": cycle_loss,
            "fresh": fresh,
            "best_fresh": best_fresh,
            "type_accs": type_accs,
            "elapsed": elapsed,
            "config": cfg,
            "params": model.total_params(),
            "teacher_status": teacher.get_status(),
            "mastery_log": teacher.mastery_log,
        })

        # Checkpoint every 10 cycles
        if cycle % 10 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "teacher": teacher.to_dict(),
                "cycle": cycle,
                "best_fresh": best_fresh,
                "config": cfg,
            }, ckpt_path)

    # Shutdown
    write_status(run_dir, "paused")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "teacher": teacher.to_dict(),
        "cycle": cycle,
        "best_fresh": best_fresh,
        "config": cfg,
    }, ckpt_path)
    log.write(f"Worker stopped at cycle {cycle}\n")
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args)
