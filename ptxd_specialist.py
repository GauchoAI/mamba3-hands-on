#!/usr/bin/env python3
"""
Drop-in replacement for specialist_trainer.py that proxies to ptxd.

Accepts the same CLI surface that three_populations.py.spawn_worker passes,
ships a single JSON job to a long-running ptxd subprocess, parses the
result, and writes the same MetricsWriter rows that specialist_trainer.py
writes — so three_populations.py doesn't need to change anything except
which script it spawns.

Usage (same as specialist_trainer.py):
    python3 ptxd_specialist.py --task parity --d-model 32 --d-state 16 \
            --headdim 16 --layers 1 --lr 1e-3 --weight-decay 0.1 \
            --batch-size 16 --steps-per-cycle 200 --max-cycles 10 \
            --target-acc 0.95 --mode champion

Environment:
    PTXD_BIN  — path to the ptxd binary (default:
                engine/ptx/target/release/ptxd relative to this file)
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PTXD = REPO_ROOT / "engine" / "ptx" / "target" / "release" / "ptxd"


def parse_args():
    p = argparse.ArgumentParser()
    # Subset of specialist_trainer's flags that ptxd actually consumes.
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--headdim", type=int, default=16)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--vocab-size", type=int, default=260)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--optimizer", type=str, default="adamw")
    p.add_argument("--loss-fn", type=str, default="ce")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps-per-cycle", type=int, default=200)
    p.add_argument("--max-cycles", type=int, default=10)
    p.add_argument("--target-acc", type=float, default=0.95)
    p.add_argument("--n-bits", type=int, default=4,
                   help="Fixed bit length for parity task (ptxd doesn't currently do curriculum).")
    p.add_argument("--mode", type=str, default="champion")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--scan-backend", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--problems-dir", type=str, default="problems")
    p.add_argument("--db-path", type=str, default=None)
    p.add_argument("--run-dir", type=str, default=None)
    # Distillation: if --kd-weight > 0 AND a teacher exists for this task in
    # ModelRegistry, ptxd_specialist runs the teacher forward on every batch
    # and ships teacher logits via batch v2. ptxd's kd_apply kernel blends
    # them in. If no teacher is found, falls back to plain CE silently.
    p.add_argument("--kd-weight", type=float, default=0.0,
                   help="KD blend weight (default 0 = no distillation, matches "
                        "specialist_trainer's 0.3 when distilling)")
    p.add_argument("--kd-temperature", type=float, default=3.0,
                   help="KD temperature; matches specialist_trainer's T=3")
    p.add_argument("--teacher-pt", type=str, default=None,
                   help="Override teacher discovery — load this .pt path "
                        "directly instead of querying ModelRegistry. Useful "
                        "for testing or running offline.")
    return p.parse_args()


def main():
    args = parse_args()
    # ptxd is now task-agnostic. Any task in `problems/` works through the
    # streaming protocol: Python generates batches via the existing
    # generators (registry → encode_curriculum → batch_writer), Rust trains
    # on whatever (tokens, targets) lands in the batch file. If the task is
    # unknown to the registry, task_runner raises and we surface a clear
    # error rather than silently falling back to PyTorch.
    ptxd_bin = os.environ.get("PTXD_BIN", str(DEFAULT_PTXD))
    if not Path(ptxd_bin).exists():
        sys.stderr.write(
            f"[ptxd_specialist] ptxd binary not found at {ptxd_bin}.\n"
            f"  Build it first:  cd engine/ptx && cargo build --release --bin ptxd\n"
        )
        sys.exit(2)

    # Total steps = max_cycles × steps_per_cycle. ptxd does its own eval every
    # 200 steps and early-stops at target_acc. So passing the budget as
    # `steps` is right.
    total_steps = max(1, args.max_cycles * args.steps_per_cycle)

    # Resume from checkpoint if one exists for this task — same convention
    # specialist_trainer.py uses.
    ckpt_dir  = REPO_ROOT / "checkpoints" / "specialists"
    ckpt_path = ckpt_dir / f"{args.task}.pt"
    init_from_bin = None
    resume_meta = None
    if ckpt_path.exists():
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from ckpt_bridge import pt_to_bin
            tmp_bin = REPO_ROOT / f"/tmp/ptxd_resume_{args.task}.bin"
            os.makedirs(tmp_bin.parent, exist_ok=True)
            resume_meta = pt_to_bin(str(ckpt_path), str(tmp_bin))
            init_from_bin = str(tmp_bin)
            sys.stderr.write(
                f"[ptxd_specialist] resuming {args.task} from {ckpt_path} "
                f"(prev acc={resume_meta.get('accuracy', 0):.0%}, "
                f"cycles={resume_meta.get('cycles', 0)})\n"
            )
        except Exception as e:
            sys.stderr.write(
                f"[ptxd_specialist] checkpoint resume FAILED ({e}); "
                f"starting from random init\n"
            )

    # save_bin: ptxd writes weights to this path on completion. We then
    # convert it back to .pt and overwrite the canonical checkpoint.
    save_bin_path = f"/tmp/ptxd_save_{args.task}.bin"

    # Generate training and eval batches in Python via the existing task
    # generators (Phase 1 streaming protocol — see batch_writer.py and
    # task_runner.py). ptxd reads them through batch_format::BatchReader.
    # This is the seam that lets ptxd stay task-agnostic: any task in
    # `generators/` works without Rust changes.
    sys.path.insert(0, str(REPO_ROOT))
    from task_runner import write_task_batches
    train_path = f"/tmp/ptxd_train_{args.task}.bin"
    eval_path  = f"/tmp/ptxd_eval_{args.task}.bin"
    # Curriculum: read the task's current stage from StateDB. This is the
    # ratchet specialist_trainer used (state_db.get_current_stage) — only
    # advances upward, never goes back. Each ptxd_specialist invocation
    # covers ONE stage; advancement happens at the end based on best_acc.
    # three_populations.py respawns the worker per round, so the next
    # round picks up the new stage automatically (matches the hot-deploy
    # pattern that already works).
    sys.path.insert(0, str(REPO_ROOT))
    from state_db import StateDB as _StageDB
    try:
        _stage_db = _StageDB(args.db_path or "three_pop/training.db")
        current_stage = _stage_db.get_current_stage(args.task)
        _stage_db.close()
    except Exception as e:
        sys.stderr.write(f"[ptxd_specialist] stage lookup failed ({e}); using stage=0\n")
        current_stage = 0
    try:
        from task_runner import make_examples_for_task
        from batch_writer import write_examples
        n_train = max(20000, args.batch_size * 64)
        train_examples = make_examples_for_task(args.task, n_train,
                                                stage=current_stage, seed=args.seed)
        eval_examples  = make_examples_for_task(args.task, 200,
                                                stage=current_stage, seed=args.seed + 1)
        sys.stderr.write(f"[ptxd_specialist] curriculum stage={current_stage}\n")
    except Exception as e:
        sys.stderr.write(f"[ptxd_specialist] batch generation FAILED: {e}\n")
        sys.exit(4)

    # Distillation: load teacher and run forward to get per-example logits
    # at supervised positions. If the task has no teacher in ModelRegistry,
    # `find_teacher_for_task` returns None and we silently fall back to CE.
    teacher_train_logits = None
    teacher_eval_logits  = None
    teacher_loaded = False
    if args.kd_weight > 0.0:
        try:
            from teacher import (find_teacher_for_task, load_teacher_model,
                                 compute_teacher_logits_for_examples)
            # --teacher-pt overrides discovery. Useful when offline / testing.
            if args.teacher_pt and Path(args.teacher_pt).exists():
                import torch as _t
                _ck = _t.load(args.teacher_pt, map_location="cpu", weights_only=False)
                t_cfg = _ck.get("config", {})
                t_acc = _ck.get("accuracy", 0.0)
                found = (args.teacher_pt, t_cfg, t_acc)
            else:
                found = find_teacher_for_task(args.task)
            if found is None:
                sys.stderr.write(f"[ptxd_specialist] kd_weight={args.kd_weight} "
                                 f"but no teacher registered for {args.task!r}; "
                                 f"falling back to plain CE\n")
            else:
                teacher_pt, t_cfg, t_acc = found
                sys.stderr.write(f"[ptxd_specialist] distilling from teacher "
                                 f"{teacher_pt} (acc={t_acc:.0%}, "
                                 f"d={t_cfg.get('d_model')} L={t_cfg.get('n_kernel_layers')})\n")
                teacher_model, t_device = load_teacher_model(teacher_pt, device="cuda")
                teacher_train_logits = compute_teacher_logits_for_examples(
                    teacher_model, train_examples, args.vocab_size,
                    batch_size=64, device=t_device,
                )
                teacher_eval_logits = compute_teacher_logits_for_examples(
                    teacher_model, eval_examples, args.vocab_size,
                    batch_size=64, device=t_device,
                )
                teacher_loaded = True
                sys.stderr.write(f"[ptxd_specialist] teacher logits computed on {t_device}\n")
        except Exception as e:
            sys.stderr.write(f"[ptxd_specialist] teacher logits computation "
                             f"failed ({e}); falling back to plain CE\n")
            teacher_train_logits = None
            teacher_eval_logits  = None

    # Write batches (v1 if no teacher, v2 if teacher loaded).
    try:
        write_examples(train_path, train_examples,
                       teacher_logits=teacher_train_logits,
                       vocab_size=args.vocab_size if teacher_loaded else None)
        write_examples(eval_path, eval_examples,
                       teacher_logits=teacher_eval_logits,
                       vocab_size=args.vocab_size if teacher_loaded else None)
    except Exception as e:
        sys.stderr.write(f"[ptxd_specialist] batch write FAILED: {e}\n")
        sys.exit(4)

    # Map specialist_trainer's flag names → ptxd's tagged-enum job spec.
    # Variants ptxd doesn't fully implement yet (lion, focal, label_smooth,
    # warm_restarts) flow through and trigger a stderr warning + fallback
    # in scheduler.rs::JobRunner::new — they don't crash the job, just
    # fall back to the implemented behaviour. This keeps the GA's mutation
    # surface alive end-to-end so we can audit which knobs actually do
    # something today.
    if args.optimizer == "adamw":
        optimizer_cfg = {"type": "adamw", "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
    elif args.optimizer == "lion":
        optimizer_cfg = {"type": "lion", "beta1": 0.9, "beta2": 0.99}
    else:
        sys.stderr.write(f"[ptxd_specialist] unknown optimizer={args.optimizer!r}, defaulting to adamw\n")
        optimizer_cfg = {"type": "adamw", "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}

    if args.loss_fn in ("ce", "stable_ce"):
        loss_cfg = {"type": "ce"}
    elif args.loss_fn == "focal":
        loss_cfg = {"type": "focal", "gamma": 2.0}
    elif args.loss_fn == "label_smooth":
        loss_cfg = {"type": "label_smooth", "smoothing": 0.1}
    else:
        sys.stderr.write(f"[ptxd_specialist] unknown loss_fn={args.loss_fn!r}, defaulting to ce\n")
        loss_cfg = {"type": "ce"}

    # If a teacher was successfully loaded, override the loss to ce_kd —
    # this signals to ptxd's JobRunner to wire kd_apply into the training
    # path. Without this, the teacher logits would sit unused in the v2
    # batch file. kd_weight + temperature mirror specialist_trainer's
    # production values (0.3, 3.0).
    if teacher_loaded:
        loss_cfg = {
            "type": "ce_kd",
            "kd_weight": args.kd_weight,
            "temperature": args.kd_temperature,
        }

    job = {
        "id": str(uuid.uuid4())[:8],
        "task": args.task,
        "n_bits": args.n_bits,
        "d_model": args.d_model,
        "d_state": args.d_state,
        "headdim": args.headdim,
        "n_layers": args.layers,
        "vocab_size": args.vocab_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "steps": total_steps,
        "batch_size": args.batch_size,
        "target_acc": args.target_acc,
        "seed": args.seed,
        "save_bin": save_bin_path,
        "batches_path": train_path,
        "eval_batches_path": eval_path,
        "optimizer": optimizer_cfg,
        "loss": loss_cfg,
    }
    if init_from_bin:
        job["init_from_bin"] = init_from_bin

    # StateDB only — specialist_trainer.py also uses StateDB exclusively for
    # this DB file (it imports MetricsWriter but never calls it, because
    # state_db.py and metrics_db.py both define an `experiments` table with
    # incompatible schemas — using both on the same file raises errors).
    sys.path.insert(0, str(REPO_ROOT))
    from state_db import StateDB

    db_path = args.db_path or "three_pop/training.db"
    sdb = StateDB(db_path)
    exp_id = job["id"]
    config = {
        "task": args.task,
        "d_model": args.d_model,
        "d_state": args.d_state,
        "headdim": args.headdim,
        "n_kernel_layers": args.layers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "steps_per_cycle": args.steps_per_cycle,
        "engine": "ptxd",
    }

    sys.stderr.write(f"[ptxd_specialist] launching ptxd: {ptxd_bin}\n")
    sys.stderr.write(f"[ptxd_specialist] job: {json.dumps(job)}\n")

    proc = subprocess.Popen(
        [ptxd_bin],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    t0 = time.time()
    proc.stdin.write(json.dumps(job) + "\n")
    proc.stdin.flush()
    proc.stdin.close()

    # Stream rows. ptxd emits one JSON object per line:
    #   {"type":"cycle", "cycle":N, "step":S, "loss":..., "fresh_acc":..., ...}
    #     — every 200 steps; orchestrator can monitor in real time.
    #   {"type":"final", "best_acc":..., "status":..., "wall_ms":..., ...}
    #     — once at the end.
    final_result = None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(f"[ptxd_specialist] non-JSON line: {line!r}\n")
            continue
        rtype = row.get("type")
        if rtype == "cycle":
            # StateDB cycle_history is what get_confidence_score queries;
            # without it confidence is 0 and the task can never be promoted.
            try:
                sdb.log_cycle(args.task, int(row["cycle"]),
                              accuracy=float(row["fresh_acc"]),
                              loss=float(row["loss"]))
            except Exception as e:
                sys.stderr.write(f"[ptxd_specialist] StateDB log_cycle warn: {e}\n")
            sys.stderr.write(
                f"[ptxd_specialist] cycle {row['cycle']}  "
                f"loss={row['loss']:.4f}  acc={row['fresh_acc']*100:.0f}%  "
                f"best={row['best_fresh']*100:.0f}%  stage={row['stage']}\n"
            )
        elif rtype == "final":
            final_result = row
        elif rtype == "tick":
            # Scheduler heartbeat; silently ignore (firebase_push handles it).
            pass
        else:
            sys.stderr.write(f"[ptxd_specialist] unknown row type: {rtype}\n")

    proc.wait(timeout=10)
    if final_result is None:
        sys.stderr.write(
            f"[ptxd_specialist] ptxd produced no final row; stderr:\n"
            f"{proc.stderr.read()}\n"
        )
        sys.exit(3)

    elapsed = time.time() - t0
    final_loss = float(final_result.get("final_loss", 0.0))
    best_acc = float(final_result.get("best_acc", 0.0))
    final_status = ("mastered" if best_acc >= args.target_acc
                    else final_result.get("status", "needs_tuning"))
    result = final_result

    # ---- StateDB final integration (lineage + task_status promotion) ----
    # cycle_history rows were already written above as each cycle arrived;
    # here we close out with a lineage row and the right task_status.
    try:
        # log_lineage: one row per training round.  cycles_completed proxies for
        # round_num; for ptxd that's the total cycles run.
        cycles_completed = max(1, int(result.get("steps_executed", 0)) // max(1, args.steps_per_cycle))
        ckpt_str = ""  # ptxd does not yet emit a checkpoint file; downstream
                       # three_populations only uses the path for backup/restore,
                       # so an empty string just disables that feature for ptxd
                       # specialists. They'll respawn fresh if the GA replays them.
        logged_config = dict(config)
        sdb.log_lineage(
            task=args.task,
            round_num=cycles_completed,
            accuracy=best_acc,
            best_accuracy=best_acc,
            config=logged_config,
            role=args.mode,
            checkpoint_path=ckpt_str,
        )

        existing = sdb.get_task_status(args.task)
        existing_best = existing["best_accuracy"] if existing else 0.0
        # Confidence score uses recent cycles; ptxd_specialist's cycles count is
        # available via MetricsWriter, so the StateDB accessor will see them.
        try:
            conf_score, conf_mean, conf_std, _ = sdb.get_confidence_score(
                args.task, last_n=max(cycles_completed, 5), k=1.0)
        except Exception:
            conf_score, conf_mean, conf_std = best_acc, best_acc, 0.0

        if best_acc >= args.target_acc and conf_score >= 0.90:
            sdb.update_task_status(args.task, "mastered", logged_config, best_acc,
                                   total_cycles=cycles_completed,
                                   confidence_score=conf_score,
                                   confidence_mean=conf_mean,
                                   confidence_std=conf_std)
            try:
                sdb.register_teacher(args.task, best_acc, cycles_completed,
                                     logged_config, checkpoint_path=ckpt_str)
            except Exception:
                pass
            sys.stderr.write(f"[ptxd_specialist] {args.task} mastered (best={best_acc:.0%})\n")
        elif best_acc > existing_best:
            sdb.update_task_status(args.task, "training", logged_config, best_acc,
                                   total_cycles=cycles_completed,
                                   confidence_score=conf_score,
                                   confidence_mean=conf_mean,
                                   confidence_std=conf_std)
        else:
            sdb.update_task_status(args.task,
                                   confidence_score=conf_score,
                                   confidence_mean=conf_mean,
                                   confidence_std=conf_std)

        # Curriculum advancement. Look up the current task's spec to find
        # this stage's advance_at threshold; if best_acc cleared it,
        # ratchet to the next stage. State is stored in StateDB so the
        # next ptxd_specialist invocation (next GA round) picks up the
        # new stage automatically. Mirrors specialist_trainer's behaviour.
        try:
            from registry.problem_registry import ProblemRegistry
            _reg = ProblemRegistry()
            _reg.discover([str(REPO_ROOT / "problems")])
            spec = _reg.problems.get(args.task)
            if spec and spec.curriculum:
                # Stages are 1-indexed in YAML; current_stage from StateDB
                # is 1-indexed too (default 1). curriculum is a list, so
                # the entry for stage N is curriculum[N-1].
                stage_idx = max(0, current_stage - 1)
                if stage_idx < len(spec.curriculum):
                    threshold = spec.curriculum[stage_idx].advance_at
                    if best_acc >= threshold and current_stage < len(spec.curriculum):
                        sdb.advance_stage(args.task, current_stage + 1)
        except Exception as e:
            sys.stderr.write(f"[ptxd_specialist] stage advancement check failed: {e}\n")

        try:
            sdb.clear_active_run(args.task)
        except Exception:
            pass
        sdb.close()
    except Exception as e:
        sys.stderr.write(f"[ptxd_specialist] StateDB integration warning: {e}\n")

    # ---- Save checkpoint back to canonical .pt path ----
    # If ptxd wrote a save_bin file, convert it to PyTorch format and
    # overwrite checkpoints/specialists/{task}.pt — matches what
    # specialist_trainer.py would do at end of training.
    try:
        if os.path.exists(save_bin_path):
            from ckpt_bridge import bin_to_pt
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            prior_cycles = (resume_meta.get("cycles", 0) if resume_meta else 0)
            bin_to_pt(
                save_bin_path, str(ckpt_path),
                task=args.task, config=config,
                accuracy=best_acc,
                cycles=prior_cycles + (cycles_completed if 'cycles_completed' in dir() else int(result.get("steps_executed", 0)) // max(1, args.steps_per_cycle)),
            )
            sys.stderr.write(
                f"[ptxd_specialist] saved {args.task} → {ckpt_path} "
                f"(acc={best_acc:.0%})\n"
            )
            # Cleanup tmp bin
            try: os.unlink(save_bin_path)
            except Exception: pass
        else:
            sys.stderr.write(
                f"[ptxd_specialist] WARNING: ptxd did not write {save_bin_path}; "
                f"checkpoint NOT updated\n"
            )
    except Exception as e:
        sys.stderr.write(f"[ptxd_specialist] checkpoint save FAILED: {e}\n")

    # Same human-readable summary to stdout that three_populations might tail.
    print(f"[ptxd_specialist] {args.task}  best_acc={best_acc*100:.1f}%  "
          f"loss={final_loss:.4f}  ms/step={result.get('ms_per_step', 0):.2f}  "
          f"({elapsed:.1f}s wall, status={final_status})")
    sys.exit(0 if best_acc >= args.target_acc else 1)


if __name__ == "__main__":
    main()
