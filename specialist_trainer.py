"""
Specialist trainer: one model, one task, GA-optimized.

The genetic algorithm still explores architecture/hyperparameters,
but each worker only trains on ONE task. No multi-task interference.

Once a specialist reaches mastery (90%+), it freezes and becomes
a teacher for the distillation phase.

Usage:
    python specialist_trainer.py --task parity
    python specialist_trainer.py --task logic_gate --d-model 64 --layers 3
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import torch
from pathlib import Path
from collections import defaultdict

from progressive_model import ProgressiveModel, ByteTokenizer, VOCAB_SIZE, PAD
from grokking import stable_cross_entropy, PerpGradOptimizer
from strategies import Lion, WarmRestartScheduler, inject_noise, label_smoothed_cross_entropy
from metrics_db import MetricsWriter
import torch.nn.functional as F


GENERATORS = {}  # task_name → generator function

def load_generators():
    """Load all task generators."""
    global GENERATORS
    from generators.level0_patterns import (
        gen_parity, gen_binary_pattern_next, gen_same_different,
        gen_odd_one_out, gen_sequence_completion, gen_pattern_period,
        gen_run_length_next, gen_mirror_detection, gen_repeat_count,
        gen_arithmetic_next, gen_geometric_next, gen_alternating_next,
        gen_logic_gate, gen_logic_chain, gen_modus_ponens,
    )
    GENERATORS = {
        "parity": gen_parity,
        "binary_pattern_next": gen_binary_pattern_next,
        "same_different": gen_same_different,
        "odd_one_out": gen_odd_one_out,
        "sequence_completion": gen_sequence_completion,
        "pattern_period": gen_pattern_period,
        "run_length_next": gen_run_length_next,
        "mirror_detection": gen_mirror_detection,
        "repeat_count": gen_repeat_count,
        "arithmetic_next": gen_arithmetic_next,
        "geometric_next": gen_geometric_next,
        "alternating_next": gen_alternating_next,
        "logic_gate": gen_logic_gate,
        "logic_chain": gen_logic_chain,
        "modus_ponens": gen_modus_ponens,
    }


def get_loss_fn(name):
    if name == "ce":
        return lambda l, t, reduction='none': F.cross_entropy(l, t, reduction=reduction)
    elif name == "focal":
        def focal(l, t, reduction='none', gamma=2.0):
            ce = F.cross_entropy(l, t, reduction='none')
            pt = torch.exp(-ce)
            loss = ((1 - pt) ** gamma) * ce
            return loss.mean() if reduction == 'mean' else loss
        return focal
    elif name == "label_smooth":
        return lambda l, t, reduction='none': label_smoothed_cross_entropy(l, t, smoothing=0.1, reduction=reduction)
    else:
        return stable_cross_entropy


def train_specialist(task, config, device, max_cycles=500, target_acc=0.95,
                     on_cycle=None, teachers=None):
    """Train one specialist on one task. Returns when mastered or max cycles.

    Resumes from checkpoints/specialists/{task}.pt if it exists (same task only).

    If teachers is provided (list of {model, weight}), blends distillation
    loss from each teacher with the task loss each cycle.
    """
    load_generators()
    gen_fn = GENERATORS.get(task)
    if not gen_fn:
        print(f"Unknown task: {task}")
        return None

    tok = ByteTokenizer()

    # Model
    model = ProgressiveModel(
        d_model=config.get("d_model", 64),
        d_state=config.get("d_state", 16),
        expand=2,
        headdim=config.get("headdim", 16),
    ).to(device)
    for _ in range(config.get("n_kernel_layers", 3)):
        model.add_kernel_layer()
    model.set_mode("kernel")

    n_params = sum(p.numel() for p in model.parameters())

    # Optimizer
    wd = config.get("weight_decay", 0.0)
    if config.get("optimizer") == "lion":
        opt = Lion(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3), weight_decay=wd)

    use_perp = config.get("use_perp", wd == 0.0)
    perp = PerpGradOptimizer(model) if use_perp else None
    scheduler = WarmRestartScheduler(opt, T_0=100) if config.get("warm_restarts") else None
    noise = config.get("noise_scale", 0.0)
    loss_fn = get_loss_fn(config.get("loss_fn", "stable_ce"))

    batch_size = config.get("batch_size", 256)
    steps_per_cycle = config.get("steps_per_cycle", 200)
    best_acc = 0.0
    cycle_start = 0

    # Resume from checkpoint if exists (same task)
    ckpt_dir = Path("checkpoints/specialists")
    ckpt_path = ckpt_dir / f"{task}.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ckpt.get("task") == task:
                model.load_state_dict(ckpt["model"])
                cycle_start = ckpt.get("cycles", 0)
                best_acc = ckpt.get("accuracy", 0.0)
                if "optimizer" in ckpt:
                    try:
                        opt.load_state_dict(ckpt["optimizer"])
                    except Exception:
                        pass
                print(f"  Resumed {task} from cycle {cycle_start}, best={best_acc:.0%}", flush=True)
        except Exception as e:
            print(f"  Checkpoint load failed: {e}", flush=True)

    print(f"\n[{task}] d={config.get('d_model')}, L={config.get('n_kernel_layers')}, "
          f"{n_params:,} params", flush=True)

    # Load teacher models for distillation
    teacher_models = []
    if teachers:
        try:
            from external_teacher import ExternalTeacher
            for t_info in teachers:
                t_name = t_info.get("model", "")
                t_weight = t_info.get("weight", 1.0)
                if t_weight < 0.1:
                    continue  # too weak, skip
                try:
                    ext = ExternalTeacher(t_name, device=device)
                    teacher_models.append({"ext": ext, "weight": t_weight, "name": t_name})
                    print(f"  Teacher: {t_name} (weight={t_weight:.2f})", flush=True)
                except Exception as e:
                    print(f"  Teacher {t_name} failed: {e}", flush=True)
        except ImportError:
            pass
    if teacher_models:
        print(f"  Distilling from {len(teacher_models)} teachers", flush=True)

    for cycle in range(cycle_start + 1, cycle_start + max_cycles + 1):
        t0 = time.time()
        model.train()

        # Generate task-specific data
        examples = []
        for _ in range(5000):
            ex = gen_fn()
            tokens, sep = tok.encode_curriculum(ex)
            examples.append((tokens, sep))

        cycle_loss = 0.0
        for step in range(steps_per_cycle):
            # Build batch
            indices = torch.randint(0, len(examples), (batch_size,))
            max_len = 0
            batch = []
            for idx in indices:
                tokens, sep = examples[idx.item()]
                batch.append((tokens, sep))
                max_len = max(max_len, len(tokens))

            token_tensor = torch.full((batch_size, max_len), PAD,
                                     dtype=torch.long, device=device)
            sep_positions = []
            for i, (tokens, sep) in enumerate(batch):
                token_tensor[i, :len(tokens)] = torch.tensor(tokens)
                sep_positions.append(sep)

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
                loss_all = loss_fn(logits_flat, targets_flat, reduction='none')
                loss = (loss_all * mask_flat).sum() / (mask_flat.sum() + 1e-8)

                # Blend distillation loss from teachers
                if teacher_models and step % 10 == 0:  # every 10th step (amortize cost)
                    for t_info in teacher_models:
                        try:
                            ext = t_info["ext"]
                            w = t_info["weight"]
                            if ext.specialist_model is not None:
                                with torch.no_grad():
                                    t_logits = ext.specialist_model(token_tensor)
                                # KL divergence on output positions
                                t_flat = t_logits[:, :L-1].reshape(-1, t_logits.shape[-1])
                                soft_teacher = torch.nn.functional.softmax(t_flat / 3.0, dim=-1)
                                soft_student = torch.nn.functional.log_softmax(logits_flat / 3.0, dim=-1)
                                kl = torch.nn.functional.kl_div(
                                    soft_student, soft_teacher, reduction='none'
                                ).sum(-1)
                                distill_loss = (kl * mask_flat).sum() / (mask_flat.sum() + 1e-8) * 9.0
                                loss = loss + w * 0.3 * distill_loss
                        except Exception:
                            pass

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if perp:
                    perp.project()
                _grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if scheduler:
                    scheduler.step()
                cycle_loss += loss.item()

        if noise > 0:
            inject_noise(model, noise)

        cycle_loss /= max(steps_per_cycle, 1)
        elapsed = time.time() - t0

        # Evaluate on this task only
        model.eval()
        correct = 0
        total = 0
        _errors_by_len = {}   # input_len → [correct_count, total_count]
        _errors_by_out = {}   # output_char → [correct_count, total_count]
        _conf_correct = []
        _conf_wrong = []
        with torch.no_grad():
            for _ in range(100):
                ex = gen_fn()
                tokens, sep = tok.encode_curriculum(ex)
                out_bytes = list(ex["output"].encode("utf-8"))
                inp_len = len(ex.get("input", "").split(","))
                out_char = ex["output"][0] if ex["output"] else "?"
                t = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = model(t)
                ok = True
                conf = 0.0
                for j, expected in enumerate(out_bytes):
                    p = sep + j
                    if p < logits.shape[1]:
                        probs = torch.softmax(logits[0, p], dim=-1)
                        conf = probs[expected].item()
                        if logits[0, p].argmax().item() != expected:
                            ok = False
                            break
                    else:
                        ok = False
                if ok:
                    correct += 1
                    _conf_correct.append(conf)
                else:
                    _conf_wrong.append(conf)
                total += 1
                # Track by length
                if inp_len not in _errors_by_len:
                    _errors_by_len[inp_len] = [0, 0]
                _errors_by_len[inp_len][1] += 1
                if ok:
                    _errors_by_len[inp_len][0] += 1
                # Track by output
                if out_char not in _errors_by_out:
                    _errors_by_out[out_char] = [0, 0]
                _errors_by_out[out_char][1] += 1
                if ok:
                    _errors_by_out[out_char][0] += 1
        acc = correct / max(total, 1)
        best_acc = max(best_acc, acc)
        model.train()

        # Compute error analysis
        _ebl = {str(k): round(v[0]/max(v[1],1), 3) for k, v in sorted(_errors_by_len.items())}
        _ebo = {k: round(v[0]/max(v[1],1), 3) for k, v in _errors_by_out.items()}
        _acc_correct = sum(_conf_correct) / max(len(_conf_correct), 1)
        _acc_wrong = sum(_conf_wrong) / max(len(_conf_wrong), 1)
        # Length correlation: negative means fails on longer inputs
        if len(_ebl) >= 3:
            lens = sorted(_ebl.keys(), key=int)
            accs_by_len = [_ebl[l] for l in lens]
            n = len(accs_by_len)
            xs = list(range(n))
            mx, my = sum(xs)/n, sum(accs_by_len)/n
            cov = sum((x-mx)*(y-my) for x,y in zip(xs, accs_by_len))
            vx = sum((x-mx)**2 for x in xs)
            vy = sum((y-my)**2 for y in accs_by_len)
            _len_corr = cov / max((vx*vy)**0.5, 1e-8)
        else:
            _len_corr = 0.0
        # Output bias: how skewed predictions are
        if _ebo:
            vals = list(_ebo.values())
            _out_bias = max(vals) - min(vals) if len(vals) > 1 else 0.0
        else:
            _out_bias = 0.0
        _overconf = _acc_wrong  # high confidence when wrong = bad

        # Compute param norm for health monitoring
        _param_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5
        _gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        _current_lr = opt.param_groups[0]["lr"]
        _eval_time = (time.time() - t0 - elapsed) if elapsed < (time.time() - t0) else 0

        print(f"  [{task}] cycle {cycle:3d}  loss={cycle_loss:.3f}  "
              f"acc={acc:.0%}  best={best_acc:.0%}  {elapsed:.1f}s", flush=True)

        # Log to state DB (real-time + per-cycle history)
        try:
            from state_db import StateDB
            _db = StateDB("three_pop/training.db")
            _db.update_active_run(task, cycle, accuracy=acc, best_accuracy=best_acc,
                                  loss=cycle_loss, config=config)
            _db.log_cycle(task, cycle, accuracy=acc, loss=cycle_loss,
                         grad_norm=float(_grad_norm) if isinstance(_grad_norm, torch.Tensor) else _grad_norm,
                         lr=_current_lr, forward_ms=elapsed * 1000 / max(steps_per_cycle, 1),
                         eval_ms=_eval_time * 1000, gpu_mem_mb=_gpu_mem,
                         param_norm=_param_norm)
            _db.log_error_analysis(task, cycle, correct, total, acc,
                                   errors_by_length=_ebl, errors_by_output=_ebo,
                                   avg_confidence_correct=round(_acc_correct, 4),
                                   avg_confidence_wrong=round(_overconf, 4),
                                   length_correlation=round(_len_corr, 4),
                                   output_bias=round(_out_bias, 4),
                                   overconfidence=round(_overconf, 4))
            # Run diagnostician on self — store signals for orchestrator
            try:
                from diagnostician import Diagnostician
                _diag = Diagnostician(_db)
                _signals = _diag.diagnose(task)
                if _signals:
                    _db.update_task_status(task, diagnostic_signals=_signals)
            except Exception:
                pass
            _db.close()
        except Exception:
            pass

        # Callback — push to Firebase / UI
        if on_cycle:
            on_cycle(task, cycle, acc, best_acc, cycle_loss)

        # Direct Firebase push (for subprocess workers without on_cycle)
        if not on_cycle:
            try:
                import firebase_push as fb
                fb._put(f"mamba3/task_series/{task}/{cycle}", {
                    "acc": round(acc, 3), "diff": 0,
                })
                fb._put(f"mamba3/experiments/{task}/best_fresh", round(best_acc, 4))
                fb._put(f"mamba3/experiments/{task}/cycle", cycle)
                fb._put(f"mamba3/experiments/{task}/cycles/{cycle}", {
                    "fresh": round(acc, 4), "loss": round(cycle_loss, 4), "t": time.time(),
                })
                # Push real-time GPU stats (during training, not between batches)
                from coordinator import get_gpu_usage
                gpu_pct, mem_pct = get_gpu_usage()
                fb._put("mamba3/snapshot/gpu_pct", round(gpu_pct, 1))
                fb._put("mamba3/snapshot/mem_pct", round(mem_pct, 1))
                fb._put("mamba3/snapshot/timestamp", time.time())
            except Exception:
                pass

        # Mastered!
        if acc >= target_acc:
            print(f"  ★ [{task}] MASTERED at {acc:.0%} in {cycle} cycles!", flush=True)
            break

    # Log lineage + update task_status + clear active run
    # Worker writes its own state so nothing is lost if orchestrator dies
    try:
        from state_db import StateDB
        _db = StateDB("three_pop/training.db")
        ckpt_str = str(Path("checkpoints/specialists") / f"{task}.pt")
        _db.log_lineage(
            task=task, round_num=cycle,
            accuracy=best_acc, best_accuracy=best_acc,
            config=config, role="worker",
            checkpoint_path=ckpt_str,
        )
        if best_acc >= target_acc:
            _db.update_task_status(task, "mastered", config, best_acc,
                                   total_cycles=cycle)
            _db.register_teacher(task, best_acc, cycle, config,
                                checkpoint_path=ckpt_str)
            print(f"  Worker registered teacher: {task} ({best_acc:.0%})", flush=True)
        else:
            _db.update_task_status(task, "training", config, best_acc,
                                   total_cycles=cycle)
        _db.clear_active_run(task)
        _db.close()
    except Exception:
        pass

    # Save specialist
    ckpt_dir = Path("checkpoints/specialists")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{task}.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "task": task,
        "config": config,
        "accuracy": best_acc,
        "cycles": cycle,
        "n_params": n_params,
    }, ckpt_path)
    print(f"  Saved specialist → {ckpt_path} ({best_acc:.0%})", flush=True)

    # Precompute teacher outputs for distillation (only if mastered)
    if best_acc < target_acc:
        return best_acc

    print(f"  Precomputing teacher outputs...", flush=True)
    model.eval()
    teacher_data = []
    with torch.no_grad():
        for _ in range(10000):
            ex = gen_fn()
            tokens, sep = tok.encode_curriculum(ex)
            t = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(t)
            # Save the full distribution at output positions
            out_bytes = list(ex["output"].encode("utf-8"))
            distributions = []
            for j in range(len(out_bytes)):
                p = sep + j
                if p < logits.shape[1]:
                    distributions.append(logits[0, p].cpu())
            if distributions:
                teacher_data.append({
                    "tokens": tokens,
                    "sep": sep,
                    "target_bytes": out_bytes,
                    "teacher_logits": distributions,
                })

    cache_path = ckpt_dir / f"{task}_cache.pt"
    torch.save(teacher_data, cache_path)
    print(f"  Cached {len(teacher_data)} teacher outputs → {cache_path}", flush=True)

    return best_acc


def train_all_specialists(config, device, tasks=None):
    """Train one specialist per task."""
    load_generators()
    if tasks is None:
        tasks = list(GENERATORS.keys())

    results = {}
    for task in tasks:
        print(f"\n{'='*60}", flush=True)
        print(f"Training specialist: {task}", flush=True)
        print(f"{'='*60}", flush=True)
        acc = train_specialist(task, config, device)
        results[task] = acc

        # Push to Firebase
        try:
            import firebase_push as fb
            if acc and acc >= 0.9:
                fb.evt_mastery(f"specialist_{task}", task, 0, 0)
        except Exception:
            pass

    print(f"\n{'='*60}", flush=True)
    print(f"ALL SPECIALISTS TRAINED", flush=True)
    print(f"{'='*60}", flush=True)
    for task, acc in sorted(results.items()):
        status = "✅" if acc and acc >= 0.9 else "🔄" if acc and acc > 0 else "❌"
        print(f"  {status} {task}: {acc:.0%}" if acc else f"  ❌ {task}: failed", flush=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, help="Single task, or None for all")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--loss-fn", type=str, default="stable_ce")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps-per-cycle", type=int, default=200)
    parser.add_argument("--max-cycles", type=int, default=500)
    parser.add_argument("--target-acc", type=float, default=0.95)
    parser.add_argument("--mode", type=str, default="champion",
                       choices=["champion", "challenger"],
                       help="champion: normal training. challenger: compare against champion best.")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    config = {
        "d_model": args.d_model, "d_state": args.d_state,
        "headdim": args.headdim, "n_kernel_layers": args.layers,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "optimizer": args.optimizer, "loss_fn": args.loss_fn,
        "batch_size": args.batch_size, "steps_per_cycle": args.steps_per_cycle,
        "use_perp": args.weight_decay == 0.0,
    }

    if args.task:
        acc = train_specialist(args.task, config, device, max_cycles=args.max_cycles,
                              target_acc=args.target_acc)

        # Challenger mode: compare against champion and restore if lost
        if args.mode == "challenger" and acc is not None:
            try:
                from state_db import StateDB
                import shutil
                _db = StateDB("three_pop/training.db")
                status = _db.get_task_status(args.task)
                champion_best = status["best_accuracy"] if status else 0

                ckpt_path = Path("checkpoints/specialists") / f"{args.task}.pt"
                champion_ckpt = Path("checkpoints/specialists") / f"{args.task}_champion.pt"

                if acc > champion_best:
                    print(f"  ✓ Challenger wins: {acc:.0%} > {champion_best:.0%}", flush=True)
                    _db.update_task_status(args.task, "training", config, acc)
                    _db.log_diagnostic(args.task, 0, "challenger_result",
                                       "win", config, acc, champion_best, True)
                else:
                    print(f"  ✗ Champion holds: {champion_best:.0%} >= {acc:.0%}", flush=True)
                    if champion_ckpt.exists():
                        shutil.copy2(champion_ckpt, ckpt_path)
                    _db.log_diagnostic(args.task, 0, "challenger_result",
                                       "loss", config, acc, champion_best, False)
                _db.close()
            except Exception as e:
                print(f"  Challenger comparison error: {e}", flush=True)
    else:
        train_all_specialists(config, device)
