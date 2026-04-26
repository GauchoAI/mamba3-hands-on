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


def run_curriculum_mode(args, ptxd_bin, ckpt_dir, ckpt_path, opt_state_path,
                        save_bin_path, sdb, exp_id, config,
                        teacher_model=None, t_device="cpu",
                        kd_weight=0.0, kd_temperature=3.0):
    """From-scratch training via curriculum advancement. Submits one ptxd job
    per stage; chains weights via /tmp/ptxd_curriculum_chain_{task}.{bin,opt.bin}.

    R-3 proved this flow trains parity from scratch in 35 seconds when the
    legacy parity-data path is used. This makes the same curriculum work
    for ANY task in problems/ via the streaming protocol — generate per-
    stage batches, submit each as a separate ptxd job, advance stage on
    convergence, save final .pt at the end.

    Returns (final_dict, cycles_completed, status_string) or (None, 0, "error").
    """
    sys.path.insert(0, str(REPO_ROOT))
    from registry.problem_registry import ProblemRegistry
    from task_runner import make_examples_for_task
    from batch_writer import write_examples
    from ckpt_bridge import bin_to_pt

    reg = ProblemRegistry()
    reg.discover([str(REPO_ROOT / "problems")])
    spec = reg.problems.get(args.task)
    if not spec or not spec.curriculum:
        sys.stderr.write(f"[ptxd_specialist] task {args.task!r} has no curriculum; "
                         f"caller should fall back to single-stage path\n")
        return None, 0, "no_curriculum"

    n_stages = len(spec.curriculum)
    chain_bin = f"/tmp/ptxd_curriculum_chain_{args.task}.bin"
    chain_opt = f"/tmp/ptxd_curriculum_chain_{args.task}.opt.bin"
    # Clean chain at start of run (curriculum is fresh-init by definition).
    for p in [chain_bin, chain_opt]:
        if Path(p).exists():
            try: os.unlink(p)
            except Exception: pass

    # Each stage gets the FULL per-call budget. This sounds wasteful
    # (n_stages × max_cycles × steps_per_cycle worst case) but in
    # practice each stage `target_acc = advance_at` triggers ptxd's
    # early-exit on convergence: stage 1 typically uses ~30-35 cycles
    # finding the rule, stages 2/3 inherit good weights and finish in
    # 1-3 cycles each. Splitting the budget per-stage was the bug that
    # killed the integration test — stage 1 needed ~30 cycles but only
    # got 16. The right policy is "give each stage as much as it needs;
    # convergence triggers early exit; the budget is shared via early
    # exit, not pre-divided."
    total_steps = max(1, args.max_cycles * args.steps_per_cycle)
    steps_per_stage = total_steps

    sys.stderr.write(f"[ptxd_specialist] curriculum mode: {n_stages} stages, "
                     f"{steps_per_stage} steps/stage budget\n")

    cumulative_cycles = 0
    last_best_acc = 0.0
    final_status = "needs_tuning"
    overall_final = None

    for stage_idx, stage in enumerate(spec.curriculum):
        stage_num = stage_idx + 1
        sys.stderr.write(f"[ptxd_specialist] === stage {stage_num}/{n_stages}: "
                         f"params={stage.params}, advance_at={stage.advance_at} ===\n")

        # Generate batches at this stage's params.
        n_train = max(20000, args.batch_size * 64)
        try:
            train_examples = make_examples_for_task(args.task, n_train,
                                                    stage=stage_num, seed=args.seed + stage_num)
            eval_examples  = make_examples_for_task(args.task, 200,
                                                    stage=stage_num, seed=args.seed + 1000 + stage_num)
        except Exception as e:
            sys.stderr.write(f"[ptxd_specialist] stage {stage_num} batch gen FAILED: {e}\n")
            break

        train_path = f"/tmp/ptxd_curr_train_{args.task}_s{stage_num}.bin"
        eval_path  = f"/tmp/ptxd_curr_eval_{args.task}_s{stage_num}.bin"

        # Per-stage teacher logits when distillation is requested. The
        # teacher_model has weights tuned for the FULL task distribution
        # so it can score each curriculum stage's examples coherently.
        # Without this, distillation in curriculum mode would silently
        # fall back to plain CE (the bug R-4 didn't catch).
        per_stage_teacher_train = None
        per_stage_teacher_eval = None
        if teacher_model is not None and kd_weight > 0:
            try:
                from teacher import compute_teacher_logits_for_examples
                per_stage_teacher_train = compute_teacher_logits_for_examples(
                    teacher_model, train_examples, args.vocab_size,
                    batch_size=64, device=t_device,
                )
                per_stage_teacher_eval = compute_teacher_logits_for_examples(
                    teacher_model, eval_examples, args.vocab_size,
                    batch_size=64, device=t_device,
                )
                sys.stderr.write(f"[ptxd_specialist] stage{stage_num} teacher logits computed\n")
            except Exception as e:
                sys.stderr.write(f"[ptxd_specialist] stage{stage_num} teacher logits FAILED ({e}); falling back to CE for this stage\n")
                per_stage_teacher_train = None
                per_stage_teacher_eval = None

        try:
            has_teacher = per_stage_teacher_train is not None
            write_examples(train_path, train_examples,
                           teacher_logits=per_stage_teacher_train,
                           vocab_size=args.vocab_size if has_teacher else None)
            write_examples(eval_path, eval_examples,
                           teacher_logits=per_stage_teacher_eval,
                           vocab_size=args.vocab_size if has_teacher else None)
        except Exception as e:
            sys.stderr.write(f"[ptxd_specialist] stage {stage_num} batch write FAILED: {e}\n")
            break

        # Build job: single ptxd run, target_acc = stage's advance_at so
        # ptxd early-exits on convergence. Init from chain.bin if it
        # exists (i.e., not the first stage). Save back to chain.bin
        # so the next stage picks up our weights.
        # Loss config: ce_kd when teacher logits were successfully
        # computed for THIS stage (per-stage; if teacher forward failed
        # for some stages but not others, those stages run plain CE).
        if has_teacher:
            stage_loss = {"type": "ce_kd",
                          "kd_weight": kd_weight,
                          "temperature": kd_temperature}
        else:
            stage_loss = {"type": "ce"}

        job = {
            "id": f"{exp_id}_s{stage_num}",
            "task": args.task,
            "n_bits": args.n_bits,
            "d_model": args.d_model, "d_state": args.d_state,
            "headdim": args.headdim, "n_layers": args.layers,
            "vocab_size": args.vocab_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "steps": steps_per_stage,
            "batch_size": args.batch_size,
            "target_acc": stage.advance_at,  # this stage's clearance threshold
            "seed": args.seed,
            "save_bin": chain_bin,
            "optimizer_state_out": chain_opt,
            "batches_path": train_path,
            "eval_batches_path": eval_path,
            "loss": stage_loss,
            # The auto-tuner's stagnation rule (8 cycles flat → bail)
            # fires too early for fresh-init parity, where the model
            # plateaus at ~50% for ~27 cycles before the breakthrough.
            # During curriculum sub-jobs, the OUTER curriculum loop is
            # the supervisor: each stage's `target_acc` (= advance_at)
            # triggers ptxd's own early-exit on convergence. Disable
            # auto_tune here so it doesn't preempt the breakthrough.
            "auto_tune": False,
        }
        if Path(chain_bin).exists():
            job["init_from_bin"] = chain_bin
        if Path(chain_opt).exists():
            job["optimizer_state_in"] = chain_opt

        # Submit and read events.
        proc = subprocess.Popen([ptxd_bin], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
        proc.stdin.write(json.dumps(job) + "\n")
        proc.stdin.flush()
        proc.stdin.close()

        stage_final = None
        for line in proc.stdout:
            line = line.strip()
            if not line: continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = row.get("type")
            if t == "cycle":
                sys.stderr.write(
                    f"[ptxd_specialist] stage{stage_num}.cycle{row['cycle']:3d}  "
                    f"loss={row['loss']:.4f}  acc={row['fresh_acc']*100:5.1f}%  "
                    f"best={row['best_fresh']*100:5.1f}%\n"
                )
                try:
                    sdb.log_cycle(args.task, cumulative_cycles + int(row["cycle"]),
                                  accuracy=float(row["fresh_acc"]),
                                  loss=float(row["loss"]))
                except Exception as e:
                    sys.stderr.write(f"[ptxd_specialist] StateDB log_cycle warn: {e}\n")
            elif t == "final":
                stage_final = row
            elif t == "auto_tune":
                sys.stderr.write(f"[ptxd_specialist] stage{stage_num} ★ auto_tune {row['action']} (trigger={row['trigger']})\n")
            # ignore tick/lr_change/etc. for brevity in curriculum mode

        proc.wait(timeout=10)

        if stage_final is None:
            sys.stderr.write(f"[ptxd_specialist] stage {stage_num} produced no "
                             f"final event; aborting curriculum\n")
            sys.stderr.write(proc.stderr.read()[-600:] + "\n")
            break

        stage_acc = float(stage_final.get("best_acc", 0.0))
        stage_status = stage_final.get("status", "needs_tuning")
        cumulative_cycles += int(stage_final.get("steps_executed", 0)) // max(1, args.steps_per_cycle)
        last_best_acc = max(last_best_acc, stage_acc)
        overall_final = stage_final
        sys.stderr.write(f"[ptxd_specialist] stage{stage_num} done: "
                         f"best_acc={stage_acc:.3f}, status={stage_status}\n")

        # Advancement decision: did this stage clear its threshold?
        if stage_acc >= stage.advance_at:
            sys.stderr.write(f"[ptxd_specialist] ✓ stage{stage_num} cleared "
                             f"advance_at={stage.advance_at}; continuing\n")
            try:
                sdb.advance_stage(args.task, stage_num + 1)
            except Exception:
                pass
            final_status = "converged" if stage_num == n_stages else "learning"
        else:
            sys.stderr.write(f"[ptxd_specialist] ✗ stage{stage_num} stuck below "
                             f"advance_at={stage.advance_at} (best={stage_acc:.3f}); "
                             f"stopping curriculum\n")
            final_status = "needs_tuning"
            break

    # Convert chain → canonical .pt at the very end. This gets the
    # accumulated weights from all completed stages. Regression guard
    # in main() will compare against any prior canonical .pt.
    if Path(chain_bin).exists() and overall_final is not None:
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            # Write to save_bin_path so main()'s save logic picks it up
            # and runs through the regression guard.
            import shutil
            shutil.copy(chain_bin, save_bin_path)
            if Path(chain_opt).exists():
                # Place opt state at canonical path so the next ptxd_specialist
                # invocation can resume from it.
                shutil.copy(chain_opt, str(opt_state_path))
        except Exception as e:
            sys.stderr.write(f"[ptxd_specialist] chain → save_bin copy failed: {e}\n")

    # Synthesize a final result the rest of main() can consume.
    if overall_final is None:
        return None, 0, "error"
    overall_final["best_acc"] = last_best_acc
    overall_final["status"] = final_status
    return overall_final, cumulative_cycles, final_status


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

    # Phase 5: optimizer state round-trip. The .opt.bin file lives next to
    # the canonical .pt and carries the AdamW m/v moments + step counter
    # across rounds. With this loaded, the warmup-on-resume hack stays out
    # of the way — moments are already settled. Each round writes the
    # final state so the next round can pick up where this left off.
    opt_state_path = ckpt_dir / f"{args.task}.opt.bin"
    optimizer_state_in = str(opt_state_path) if opt_state_path.exists() else None

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
    # teacher_model + t_device are kept in scope so run_curriculum_mode can
    # recompute teacher logits per stage (R-4 fix).
    teacher_train_logits = None
    teacher_eval_logits  = None
    teacher_loaded = False
    teacher_model = None
    t_device = "cpu"
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
        "optimizer_state_out": str(opt_state_path),
    }
    if init_from_bin:
        job["init_from_bin"] = init_from_bin
    if optimizer_state_in:
        job["optimizer_state_in"] = optimizer_state_in

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

    # ---- Curriculum mode (R-3 unlock) ----
    # When training from scratch on a task that has a curriculum, run the
    # per-stage loop in run_curriculum_mode instead of the single-stage
    # path below. Test_parity_train showed parity converges in 16s with
    # curriculum; ptxd_specialist's single-stage path got stuck at 56%.
    # This flips the default for from-scratch + curriculum tasks.
    use_curriculum = (init_from_bin is None)  # fresh-init only
    final_result = None
    if use_curriculum:
        # run_curriculum_mode handles its own ptxd invocations and event
        # logging. Returns the equivalent of `final_result` so downstream
        # save / StateDB code works unchanged.
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from registry.problem_registry import ProblemRegistry as _PR
            _r = _PR()
            _r.discover([str(REPO_ROOT / "problems")])
            _spec = _r.problems.get(args.task)
            if _spec and _spec.curriculum:
                t0 = time.time()
                # Pass the teacher model (if loaded) so curriculum mode can
                # recompute teacher logits per-stage. Without this, the
                # single-stage teacher_train_logits computed on the
                # ORIGINAL batches would be discarded — and ptxd would
                # silently fall back to plain CE for the curriculum run.
                _t_model = teacher_model if teacher_loaded else None
                _t_dev = t_device if teacher_loaded else "cpu"
                cur_final, cur_cycles, cur_status = run_curriculum_mode(
                    args, ptxd_bin, ckpt_dir, ckpt_path, opt_state_path,
                    save_bin_path, sdb, exp_id, config,
                    teacher_model=_t_model, t_device=_t_dev,
                    kd_weight=args.kd_weight, kd_temperature=args.kd_temperature,
                )
                if cur_final is not None:
                    final_result = cur_final
                    elapsed = time.time() - t0
                    final_loss = float(final_result.get("final_loss", 0.0))
                    best_acc = float(final_result.get("best_acc", 0.0))
                    final_status = cur_status
                    cycles_completed = cur_cycles
                    result = final_result
                    # Skip the legacy single-stage path entirely; jump to
                    # the StateDB-and-save block below.
                else:
                    sys.stderr.write(f"[ptxd_specialist] curriculum mode failed; "
                                     f"falling back to single-stage path\n")
            else:
                use_curriculum = False  # task has no curriculum → single-stage
        except Exception as e:
            sys.stderr.write(f"[ptxd_specialist] curriculum check failed ({e}); "
                             f"single-stage path\n")
            use_curriculum = False

    # Track whether curriculum mode produced a final_result so we skip
    # the legacy single-stage event-reading block entirely.
    did_curriculum = (final_result is not None)
    if not did_curriculum:  # single-stage path
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

    # Per-job event log + event-streaming loop. SKIPPED when curriculum
    # mode already ran (its sub-jobs each have their own event flow
    # logged via stderr; per-stage event files are a future enhancement).
    events_log = None
    diagnostic_counts = {}
    if not did_curriculum:
        events_dir = REPO_ROOT / "runs"
        events_dir.mkdir(exist_ok=True)
        events_path = events_dir / f"{job['id']}_{args.task}.events.jsonl"
        events_log = open(events_path, "w")
        events_log.write(json.dumps({"type":"job_start","job":job,"t":time.time()}) + "\n")
        events_log.flush()
        sys.stderr.write(f"[ptxd_specialist] event log → {events_path}\n")

        # Stream rows. ptxd emits one JSON object per line; every line goes to
        # the events log, and we additionally interpret cycle/final/tick/etc.
        # for live console output and StateDB writes.
        diagnostic_counts = {"loss_jump": 0, "grad_norm_alert": 0,
                             "nan_detected": 0, "mode_collapse_suspected": 0,
                             "lr_change": 0, "auto_tune": 0}
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            # Tee to events log unconditionally (still readable even if parsing fails).
            events_log.write(line + "\n")
            events_log.flush()
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                sys.stderr.write(f"[ptxd_specialist] non-JSON line: {line!r}\n")
                continue
            rtype = row.get("type")
            if rtype == "cycle":
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
                pass  # scheduler heartbeat — silently ignore on console
            elif rtype in diagnostic_counts:
                diagnostic_counts[rtype] += 1
                # Surface diagnostics live so a human watching can see them.
                if rtype == "lr_change":
                    sys.stderr.write(f"[ptxd_specialist]   lr_change @ step {row['step']} reason={row['reason']} lr_eff={row['lr_eff']:.2e}\n")
                elif rtype == "grad_norm_alert":
                    sys.stderr.write(f"[ptxd_specialist]   ⚠ grad_norm_alert @ step {row['step']} norm={row['norm']:.2f} (recent_mean={row['recent_mean']:.2f})\n")
                elif rtype == "loss_jump":
                    sys.stderr.write(f"[ptxd_specialist]   ⚠ loss_jump cycle {row['cycle']} {row['prev_loss']:.3f} → {row['new_loss']:.3f} (×{row['ratio']:.1f})\n")
                elif rtype == "nan_detected":
                    sys.stderr.write(f"[ptxd_specialist]   ⚠ NaN in {row['source']} @ step {row['step']}\n")
                elif rtype == "mode_collapse_suspected":
                    sys.stderr.write(f"[ptxd_specialist]   ⚠ mode-collapse suspected @ cycle {row['cycle']} (flat for {row['flat_for_cycles']} cycles)\n")
                elif rtype == "auto_tune":
                    sys.stderr.write(f"[ptxd_specialist]   ★ auto_tune {row['action']} (trigger={row['trigger']}, value={row['value']})\n")
            else:
                sys.stderr.write(f"[ptxd_specialist] unknown row type: {rtype}\n")

        events_log.write(json.dumps({"type":"job_end","t":time.time()}) + "\n")
        events_log.close()
        if any(diagnostic_counts.values()):
            summary = ", ".join(f"{k}={v}" for k, v in diagnostic_counts.items() if v)
            sys.stderr.write(f"[ptxd_specialist] diagnostic summary: {summary}\n")

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

    # ---- Save checkpoint back, with regression guard ----
    # NEVER overwrite the canonical {task}.pt with a worse model. The 82
    # checkpoints that exist took a lot of compute to produce; a tighter-
    # budget ptxd run that doesn't reach the prior accuracy must NOT
    # clobber them. Mirrors specialist_trainer's _champion convention.
    #
    # Decision tree:
    #   best_acc >= prior_acc        → save canonical (training continues)
    #   prior_acc - best_acc <= 5%   → save to {task}_ptxd_candidate.pt
    #                                   (regression within noise; let GA decide)
    #   prior_acc - best_acc > 5%    → save to {task}_ptxd_candidate.pt
    #                                   AND log a warning (clear regression)
    #   no prior .pt                 → save canonical (fresh training run)
    #
    # The optimizer state file is gated by the same check — a regressed
    # run's m/v shouldn't seed the next round either.
    try:
        if os.path.exists(save_bin_path):
            from ckpt_bridge import bin_to_pt
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            prior_cycles = (resume_meta.get("cycles", 0) if resume_meta else 0)
            prior_acc    = (resume_meta.get("accuracy", 0.0) if resume_meta else 0.0)

            total_cycles = prior_cycles + (cycles_completed if 'cycles_completed' in dir()
                                           else int(result.get("steps_executed", 0)) // max(1, args.steps_per_cycle))

            REGRESSION_TOL = 0.05  # 5pp slack for noise / eval-distribution shift
            is_fresh   = (resume_meta is None)
            no_loss    = (best_acc + 1e-6 >= prior_acc)
            small_drop = (prior_acc - best_acc <= REGRESSION_TOL)

            if is_fresh or no_loss:
                # Safe to overwrite canonical.
                bin_to_pt(save_bin_path, str(ckpt_path),
                          task=args.task, config=config,
                          accuracy=best_acc, cycles=total_cycles)
                sys.stderr.write(
                    f"[ptxd_specialist] saved {args.task} → {ckpt_path} "
                    f"(acc={best_acc:.0%}; prior={prior_acc:.0%})\n"
                )
                if opt_state_path.exists() or args.kd_weight > 0:
                    pass  # opt state was already written by ptxd to the canonical path
            else:
                # Regression — DO NOT overwrite canonical. Save candidate
                # for the GA / human to inspect. Also avoid tainting the
                # opt state: if ptxd wrote one to the canonical path,
                # rename it out of the way so the next round doesn't
                # resume from a regressed state.
                candidate_pt = ckpt_dir / f"{args.task}_ptxd_candidate.pt"
                bin_to_pt(save_bin_path, str(candidate_pt),
                          task=args.task, config=config,
                          accuracy=best_acc, cycles=total_cycles)
                level = "WARNING" if not small_drop else "info"
                sys.stderr.write(
                    f"[ptxd_specialist] {level}: {args.task} regressed "
                    f"(prior={prior_acc:.0%}, this run={best_acc:.0%}, "
                    f"Δ={(best_acc - prior_acc)*100:+.1f}pp). "
                    f"Canonical {ckpt_path.name} preserved; "
                    f"candidate written to {candidate_pt.name}\n"
                )
                # Move the regressed opt state out of the canonical path
                # so the next round resumes from the prior good state.
                if opt_state_path.exists():
                    candidate_opt = ckpt_dir / f"{args.task}_ptxd_candidate.opt.bin"
                    try:
                        os.replace(opt_state_path, candidate_opt)
                        sys.stderr.write(f"[ptxd_specialist]   opt state moved to {candidate_opt.name}\n")
                    except Exception as e:
                        sys.stderr.write(f"[ptxd_specialist]   couldn't move opt state ({e})\n")

            # Cleanup tmp bin in either path.
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
