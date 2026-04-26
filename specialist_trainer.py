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
_REGISTRY = None  # ProblemRegistry singleton
_DB_PATH = "three_pop/training.db"  # configurable via --db-path CLI arg

def _set_db_path(path):
    global _DB_PATH
    _DB_PATH = path

def load_generators(problems_dir="problems"):
    """Load task generators via ProblemRegistry (YAML-driven discovery)."""
    global GENERATORS, _REGISTRY
    from registry.problem_registry import ProblemRegistry
    _REGISTRY = ProblemRegistry()
    _REGISTRY.discover([problems_dir])
    # Populate GENERATORS dict for backward compatibility
    for name in _REGISTRY.list_problems():
        GENERATORS[name] = _REGISTRY.get_generator(name)


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
                     on_cycle=None, teachers=None, problems_dir="problems"):
    """Train one specialist on one task. Returns when mastered or max cycles.

    Resumes from checkpoints/specialists/{task}.pt if it exists (same task only).

    If teachers is provided (list of {model, weight}), blends distillation
    loss from each teacher with the task loss each cycle.
    """
    # Override device from config (cpu for precision, cuda for speed)
    config_device = config.get("device")
    if config_device:
        device = config_device
        print(f"  Device override: {device}", flush=True)

    # Set scan backend from config (jit vs triton)
    scan_backend = config.get("scan_backend")
    if scan_backend:
        import ssm_triton
        ssm_triton.FORCE_BACKEND = scan_backend
        print(f"  Scan backend: {scan_backend}", flush=True)

    load_generators(problems_dir)
    gen_fn = GENERATORS.get(task)
    if not gen_fn:
        print(f"Unknown task: {task}")
        return None

    # Curriculum: get current stage and bind generator to stage params
    _current_stage = 1
    _problem_spec = _REGISTRY.problems.get(task) if _REGISTRY else None
    try:
        from state_db import StateDB as _StageDB
        _sdb = _StageDB(_DB_PATH)
        _current_stage = _sdb.get_current_stage(task)
        _sdb.close()
    except Exception:
        pass
    if _problem_spec and _problem_spec.curriculum:
        gen_fn = _REGISTRY.get_generator(task, stage=_current_stage)
        stage_info = _problem_spec.get_stage(_current_stage)
        if stage_info:
            print(f"  Curriculum stage {_current_stage}: {stage_info.params} (advance at {stage_info.advance_at:.0%})", flush=True)

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

    # Load teacher for distillation — transparently from any node via ModelRegistry
    teacher_model_for_distill = None
    teacher_ckpt_path = None
    try:
        from registry.model_registry import ModelRegistry
        _model_reg = ModelRegistry()
        if _model_reg.has_teacher(task):
            result = _model_reg.get_teacher(task, device=device)
            if result:
                teacher_model_for_distill, _t_cfg, _t_acc = result
                teacher_ckpt_path = _model_reg.local_dir / f"{task}.pt"
                print(f"  Distilling from teacher ({_t_acc:.0%}, d={_t_cfg.get('d_model')} L={_t_cfg.get('n_kernel_layers')})", flush=True)
    except Exception as e:
        print(f"  Teacher discovery: {e}", flush=True)

    _hit_target = False
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

                # Distillation: blend KL loss from teacher's soft targets
                if teacher_model_for_distill is not None and step % 5 == 0:
                    try:
                        with torch.no_grad():
                            t_logits = teacher_model_for_distill(token_tensor)
                        t_flat = t_logits[:, :L-1].reshape(-1, t_logits.shape[-1])
                        soft_teacher = torch.nn.functional.softmax(t_flat / 3.0, dim=-1)
                        soft_student = torch.nn.functional.log_softmax(logits_flat / 3.0, dim=-1)
                        kl = torch.nn.functional.kl_div(
                            soft_student, soft_teacher, reduction='none'
                        ).sum(-1)
                        distill_loss = (kl * mask_flat).sum() / (mask_flat.sum() + 1e-8) * 9.0
                        loss = loss + 0.3 * distill_loss
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
            _db = StateDB(_DB_PATH)
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

        # Curriculum ratchet: advance stage if accuracy sustains
        if _problem_spec and _problem_spec.curriculum:
            stage_info = _problem_spec.get_stage(_current_stage)
            if stage_info and acc >= stage_info.advance_at and _current_stage < _problem_spec.max_stage:
                _current_stage += 1
                try:
                    from state_db import StateDB as _StageDB2
                    _sdb2 = _StageDB2(_DB_PATH)
                    _sdb2.advance_stage(task, _current_stage)
                    _sdb2.close()
                except Exception:
                    pass
                gen_fn = _REGISTRY.get_generator(task, stage=_current_stage)
                new_stage = _problem_spec.get_stage(_current_stage)
                if new_stage:
                    print(f"  ★ [{task}] Stage {_current_stage}: {new_stage.params} (advance at {new_stage.advance_at:.0%})", flush=True)

        # Note: do NOT break early on target_acc hit.
        # The champion must run ALL cycles to build confidence history.
        # Early break meant only 3 good cycles + 7 old bad ones = low confidence.
        if acc >= target_acc and not _hit_target:
            _hit_target = True
            print(f"  ★ [{task}] Hit target at {acc:.0%} in {cycle} cycles! (continuing for confidence)", flush=True)

    # Log lineage + update task_status + clear active run
    # Worker writes its own state so nothing is lost if orchestrator dies
    try:
        from state_db import StateDB
        _db = StateDB(_DB_PATH)
        ckpt_str = str(Path("checkpoints/specialists") / f"{task}.pt")
        # Track distillation in config for provenance
        _logged_config = dict(config)
        if teacher_model_for_distill is not None:
            _logged_config["distilled_from"] = str(teacher_ckpt_path)
        _db.log_lineage(
            task=task, round_num=cycle,
            accuracy=best_acc, best_accuracy=best_acc,
            config=_logged_config, role="worker",
            checkpoint_path=ckpt_str,
        )
        # Use confidence-based scoring: mean - k*std, not raw best
        # IMPORTANT: use last_n=cycle (THIS run's cycles only) to avoid
        # contamination from challenger cycles mixed into cycle_history
        existing = _db.get_task_status(task)
        existing_best = existing["best_accuracy"] if existing else 0
        # Use exactly the number of cycles THIS run completed
        cycles_this_run = cycle - cycle_start
        conf_score, conf_mean, conf_std, conf_n = _db.get_confidence_score(
            task, last_n=max(cycles_this_run, 5), k=1.0)
        # Mastery requires BOTH a spike above target AND reliable score above 90%
        if best_acc >= target_acc and conf_score >= 0.90:
            _db.update_task_status(task, "mastered", config, best_acc,
                                   total_cycles=cycle,
                                   confidence_score=conf_score,
                                   confidence_mean=conf_mean,
                                   confidence_std=conf_std)
            _db.register_teacher(task, best_acc, cycle, config,
                                checkpoint_path=ckpt_str)
            print(f"  Worker registered teacher: {task} ({best_acc:.0%})", flush=True)
        elif best_acc > existing_best:
            _db.update_task_status(task, "training", config, best_acc,
                                   total_cycles=cycle,
                                   confidence_score=conf_score,
                                   confidence_mean=conf_mean,
                                   confidence_std=conf_std)
        else:
            # Even if we didn't beat the best, update confidence (it may have improved)
            _db.update_task_status(task, confidence_score=conf_score,
                                   confidence_mean=conf_mean,
                                   confidence_std=conf_std)
        # else: don't overwrite — DB has a better result than us
        _db.clear_active_run(task)
        _db.close()

        # Push RICH data to Firebase (worker does it, not orchestrator)
        import firebase_push as fb
        _db2 = StateDB(_DB_PATH)

        # 1. Full per-round lineage entry (what UI needs for genetic tree)
        node_id = f"{task}_c{cycle}_w"
        cfg_full = {
            "d_model": config.get("d_model", 64),
            "n_kernel_layers": config.get("n_kernel_layers", 3),
            "d_state": config.get("d_state", 16),
            "lr": config.get("lr", 1e-3),
            "weight_decay": config.get("weight_decay", 0.1),
            "optimizer": config.get("optimizer", "adamw"),
            "loss_fn": config.get("loss_fn", "ce"),
            "batch_size": config.get("batch_size", 256),
        }
        if config.get("use_perp"): cfg_full["use_perp"] = True
        if config.get("warm_restarts"): cfg_full["warm_restarts"] = True
        if config.get("teacher_model"): cfg_full["teacher_model"] = config["teacher_model"]
        if config.get("noise_scale"): cfg_full["noise_scale"] = config["noise_scale"]

        # Get previous round's config for diff
        lineage = _db2.get_lineage(task)
        prev_config = None
        if len(lineage) >= 2:
            prev_config = lineage[-2].get("config", {})

        # Compute mutation diff
        mutation_diff = {}
        if prev_config:
            for k in set(list(cfg_full.keys()) + list(prev_config.keys())):
                old = prev_config.get(k)
                new = cfg_full.get(k)
                if old != new:
                    mutation_diff[k] = {"from": old, "to": new}

        # Compute generation (how many rounds this task has been through)
        generation = len([e for e in lineage if e.get("role") in ("champion", "worker")])

        fb._put(f"mamba3/lineage/{node_id}", {
            "task": task,
            "round": cycle,
            "acc": round(best_acc, 3),
            "best": round(max(best_acc, existing_best), 3),
            "role": "worker",
            "status": "champion",  # workers are always champion runs
            "won": best_acc > existing_best,
            "config": cfg_full,
            "parent_id": f"{task}_c{lineage[-2]['round']}_{lineage[-2].get('role','w')[0]}" if len(lineage) >= 2 else "seed",
            "mutation_diff": mutation_diff,
            "teachers": config.get("teachers", []),
            "n_params": n_params,
            "generation": generation,
            "timestamp": time.time(),
        })

        # 2. Task status + plateau state + diagnostic signals
        best_ever_cfg, best_ever_acc = _db2.get_best_config(task)
        status_data = {
            "status": "mastered" if best_acc >= target_acc else "training",
            "best": round(max(best_acc, existing_best), 3),
            "cycles": cycle,
            "config": cfg_full,
            "best_config": cfg_full if best_acc >= existing_best else None,
            "champion_config": cfg_full,
            "best_ever_config": best_ever_cfg,
            "best_ever_acc": round(best_ever_acc, 3),
            "generation": generation,
            "n_params": n_params,
        }
        # Diagnostic signals + plateau state
        try:
            from diagnostician import Diagnostician
            diag = Diagnostician(_db2)
            signals = diag.diagnose(task)
            if signals:
                status_data["diagnostic_signals"] = [
                    {"signal": s["signal"], "evidence": s["evidence"]}
                    for s in signals
                ]
            # Plateau state
            recent = _db2.get_cycle_history(task, last_n=20)
            if recent:
                recent.sort(key=lambda r: r.get("cycle", 0))
                accs = [r["accuracy"] for r in recent if r.get("accuracy") is not None]
                if len(accs) >= 5:
                    improving = accs[-1] > accs[0] + 0.02
                    stuck_cycles = 0
                    for i in range(len(accs)-1, -1, -1):
                        if accs[i] >= max(accs) - 0.01:
                            stuck_cycles = len(accs) - i
                            break
                    fb._put(f"mamba3/state/plateau/{task}", {
                        "active": not improving and stuck_cycles > 10,
                        "stuck_cycles": stuck_cycles,
                        "severity": round(min(stuck_cycles / 30, 1.0), 2),
                        "best_ever": round(max(best_acc, existing_best), 3),
                        "since": recent[-stuck_cycles]["timestamp"] if stuck_cycles < len(recent) else time.time(),
                    })
        except Exception:
            pass
        fb._put(f"mamba3/state/task_status/{task}", status_data)

        # 3. Per-task lineage summary (for mutation timeline)
        model_card = _db2.build_model_card(task)
        challengers = [e for e in lineage if e.get("role") == "challenger"]
        champions = [e for e in lineage if e.get("role") == "champion"]
        improvements = sum(1 for i, e in enumerate(lineage)
                          if i > 0 and e["accuracy"] > lineage[i-1].get("best_accuracy", 0))
        fb._put(f"mamba3/state/lineage/{task}", {
            "rounds": len(lineage),
            "best": round(max(best_acc, existing_best), 3),
            "latest_config": cfg_full,
            "latest_round": cycle,
            "n_champions": len(champions),
            "n_challengers": len(challengers),
            "n_improvements": improvements,
            "teachers": model_card.get("teachers", []),
            "champion_config": cfg_full,
            "best_ever_config": best_ever_cfg,
            "best_ever_acc": round(best_ever_acc, 3),
            "challengers": [{
                "round": c["round"],
                "acc": round(c["accuracy"], 3),
                "config": c.get("config", {}),
                "mutation": c.get("mutation", ""),
                "won": c["accuracy"] > c.get("best_accuracy", 0),
                "status": "retired",
                "timestamp": c.get("timestamp", 0),
            } for c in challengers[-20:]],  # last 20 challengers (retention)
        })

        # 4. Diagnostic history with timestamps
        diag_stats = _db2.get_diagnostic_stats(task)
        if diag_stats:
            # Add last_tried timestamp from DB
            for ds in diag_stats:
                cur = _db2.conn.execute(
                    "SELECT MAX(timestamp) FROM diagnostic_history "
                    "WHERE task=? AND signal=? AND prescription_type=?",
                    (task, ds["signal"], ds["prescription"])
                )
                row = cur.fetchone()
                ds["last_tried"] = row[0] if row and row[0] else None
                # Is this the currently active prescription?
                ds["active"] = ds["tries"] > 0 and ds["wins"] == 0 and ds["tries"] < 3
            fb._put(f"mamba3/state/diagnostics/{task}", diag_stats)

        # 5. Teacher matrix
        teacher_scores = _db2.get_best_teachers_for_task(task)
        if teacher_scores:
            for t_name, t_score in teacher_scores:
                fb._put(f"mamba3/state/teacher_matrix/{t_name}/{task}", round(t_score, 3))

        # 6. Rich events (full vocabulary the UI wants)
        ts = time.time()

        # challenger_spawned — every run is a new config being tried
        fb._put(f"mamba3/events/{node_id}_spawned", {
            "type": "challenger_spawned",
            "task": task,
            "exp_id": node_id,
            "parent_id": f"{task}_c{lineage[-2]['round']}_{lineage[-2].get('role','w')[0]}" if len(lineage) >= 2 else "seed",
            "config": cfg_full,
            "mutation_diff": mutation_diff,
            "generation": generation,
            "timestamp": ts,
        })

        if best_acc > existing_best:
            # improvement — any accuracy increase over previous best
            fb._put(f"mamba3/events/{node_id}_improved", {
                "type": "improvement",
                "task": task,
                "exp_id": node_id,
                "from_acc": round(existing_best, 3),
                "to_acc": round(best_acc, 3),
                "config": cfg_full,
                "timestamp": ts,
            })
            # new_best — only if this is the all-time high for this task
            if best_acc >= best_ever_acc:
                fb._put(f"mamba3/events/{node_id}_new_best", {
                    "type": "new_best",
                    "task": task,
                    "exp_id": node_id,
                    "acc": round(best_acc, 3),
                    "previous_best": round(best_ever_acc, 3),
                    "config": cfg_full,
                    "timestamp": ts,
                })

        if best_acc >= target_acc:
            # mastery — task graduated, with timestamp
            fb._put(f"mamba3/events/{node_id}_mastery", {
                "type": "mastery",
                "task": task,
                "exp_id": node_id,
                "acc": round(best_acc, 3),
                "cycles": cycle,
                "config": cfg_full,
                "timestamp": ts,
            })

        # 7. Teachers with graduation timestamps
        teachers = _db2.get_teachers()
        import socket as _sock
        _node_hostname = _sock.gethostname()
        fb._put("mamba3/three_pop/teachers", {
            t: {"exp_id": info.get("exp_id", t),
                "accuracy": round(info["accuracy"], 3),
                "graduated_at": info.get("graduated_at", 0),
                "node": _node_hostname,
                "checkpoint": info.get("checkpoint_path", ""),
                "config": info.get("config", {})}
            for t, info in teachers.items()
        })

        # 8. Meta (run info for reproducibility)
        fb._put("mamba3/meta", {
            "run_name": "three_populations_v2",
            "start_timestamp": _db2.conn.execute(
                "SELECT MIN(timestamp) FROM lineage").fetchone()[0],
            "tasks": 15,
            "base_config": {
                "d_model": 64, "n_kernel_layers": 3, "d_state": 16,
                "lr": 1e-3, "weight_decay": 0.1,
            },
        })

        _db2.close()
    except Exception:
        pass

    # Sync full state to Firebase (models catalog, task progress, teachers)
    try:
        from server.push_state import push_state
        push_state(_DB_PATH)
    except Exception:
        pass

    # Save specialist — with regression guard. Refuse to overwrite a
    # prior good checkpoint with a worse run. Without this, a 0% run
    # could clobber a 100% checkpoint and then poison subsequent
    # re-trains via the load path (which loads the existing .pt as
    # initial weights when present).
    ckpt_dir = Path("checkpoints/specialists")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{task}.pt"
    prior_acc = None
    if ckpt_path.exists():
        try:
            prior_ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            prior_acc = float(prior_ck.get("accuracy", 0.0))
        except Exception:
            prior_acc = None
    if prior_acc is not None and best_acc < prior_acc - 1e-6:
        print(f"  REGRESSION GUARD: not saving — best_acc={best_acc:.0%} < prior_acc={prior_acc:.0%}; "
              f"keeping {ckpt_path}", flush=True)
    else:
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

    # Run register inspection and push to Firebase
    try:
        from register_inspector import inspect_model, save_and_push
        report = inspect_model(task, n_examples=5, device=device)
        if report:
            save_and_push(task, report, push_firebase=True)
    except Exception as e:
        print(f"  Inspection error: {e}", flush=True)

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
    parser.add_argument("--scan-backend", type=str, default=None,
                       choices=["native", "compiled", "jit", "triton"],
                       help="SSM scan backend: jit (precise) or triton (fast)")
    parser.add_argument("--device", type=str, default=None,
                       choices=["cuda", "cpu", "mps"],
                       help="Training device: cpu (precise) or cuda (fast)")
    parser.add_argument("--problems-dir", type=str, default="problems",
                       help="Directory containing problem YAML manifests")
    parser.add_argument("--db-path", type=str, default="three_pop/training.db",
                       help="Path to StateDB SQLite file")
    args = parser.parse_args()

    # Set module-level DB path before any usage
    _set_db_path(args.db_path)

    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    config = {
        "d_model": args.d_model, "d_state": args.d_state,
        "headdim": args.headdim, "n_kernel_layers": args.layers,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "optimizer": args.optimizer, "loss_fn": args.loss_fn,
        "batch_size": args.batch_size, "steps_per_cycle": args.steps_per_cycle,
        "use_perp": args.weight_decay == 0.0,
    }
    if args.scan_backend:
        config["scan_backend"] = args.scan_backend
    if args.device:
        config["device"] = args.device

    if args.task:
        acc = train_specialist(args.task, config, device, max_cycles=args.max_cycles,
                              target_acc=args.target_acc,
                              problems_dir=args.problems_dir)

        # Challenger mode: compare against champion and restore if lost
        if args.mode == "challenger" and acc is not None:
            try:
                from state_db import StateDB
                import shutil
                _db = StateDB(_DB_PATH)
                status = _db.get_task_status(args.task)
                champion_best = status["best_accuracy"] if status else 0

                # Champion confidence from STORED value (set before challenger ran)
                # This avoids mixing champion + challenger cycles in the computation
                champ_conf = status.get("confidence_score", 0) if status else 0
                champ_mean = status.get("confidence_mean", 0) if status else 0
                champ_std = status.get("confidence_std", 0) if status else 0

                # Challenger confidence from its own recent cycles (last N = max_cycles)
                chall_conf, chall_mean, chall_std, chall_n = _db.get_confidence_score(
                    args.task, last_n=args.max_cycles, k=1.0)

                ckpt_path = Path("checkpoints/specialists") / f"{args.task}.pt"
                champion_ckpt = Path("checkpoints/specialists") / f"{args.task}_champion.pt"

                # Challenger wins if its confidence score beats the champion's stored score
                # For early runs or zero champion confidence, fall back to raw accuracy
                if chall_n >= 3 and champ_conf > 0:
                    wins = chall_conf > champ_conf
                    print(f"  Confidence comparison: challenger {chall_conf:.2%} (μ={chall_mean:.0%} σ={chall_std:.2f} n={chall_n}) "
                          f"vs champion {champ_conf:.2%} (μ={champ_mean:.0%} σ={champ_std:.2f})", flush=True)
                else:
                    wins = acc > champion_best
                    print(f"  Raw comparison (insufficient history): challenger {acc:.0%} vs champion {champion_best:.0%}", flush=True)

                if wins:
                    print(f"  ✓ Challenger wins!", flush=True)
                    _db.update_task_status(args.task, "training", config, acc)
                    _db.log_diagnostic(args.task, 0, "challenger_result",
                                       "win", config, acc, champion_best, True)
                else:
                    print(f"  ✗ Champion holds!", flush=True)
                    if champion_ckpt.exists():
                        shutil.copy2(champion_ckpt, ckpt_path)
                    _db.log_diagnostic(args.task, 0, "challenger_result",
                                       "loss", config, acc, champion_best, False)
                _db.close()
            except Exception as e:
                print(f"  Challenger comparison error: {e}", flush=True)
    else:
        train_all_specialists(config, device)
