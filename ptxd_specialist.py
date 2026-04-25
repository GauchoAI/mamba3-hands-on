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
    return p.parse_args()


def main():
    args = parse_args()
    if args.task != "parity":
        # ptxd only knows parity for now. Fall back to PyTorch trainer for
        # other tasks so the orchestrator keeps working.
        sys.stderr.write(
            f"[ptxd_specialist] task={args.task} not supported by ptxd, "
            f"falling back to specialist_trainer.py\n"
        )
        os.execvp(sys.executable, [sys.executable,
            str(REPO_ROOT / "specialist_trainer.py"), *sys.argv[1:]])

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

    job = {
        "id": str(uuid.uuid4())[:8],
        "task": "parity",
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
    }

    # MetricsWriter init — we want the same DB rows specialist_trainer.py emits.
    sys.path.insert(0, str(REPO_ROOT))
    from metrics_db import MetricsWriter

    db_path = args.db_path or "three_pop/training.db"
    mw = MetricsWriter(db_path)
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
    # rough param count for the registry — d_model × vocab + per-layer terms
    n_params_estimate = args.vocab_size * args.d_model * 2  # close enough
    mw.register_experiment(exp_id, config, n_params_estimate)
    mw.log_event("ptxd_start", exp_id, json.dumps({"binary": ptxd_bin}))

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
            mw.log_cycle(
                exp_id,
                int(row["cycle"]),
                float(row["loss"]),
                float(row["fresh_acc"]),
                float(row["best_fresh"]),
                train_acc=None,
                elapsed_s=float(row.get("elapsed_s", 0.0)),
            )
            mw.log_tasks(exp_id, int(row["cycle"]), {args.task: float(row["fresh_acc"])})
            sys.stderr.write(
                f"[ptxd_specialist] cycle {row['cycle']}  "
                f"loss={row['loss']:.4f}  acc={row['fresh_acc']*100:.0f}%  "
                f"best={row['best_fresh']*100:.0f}%  stage={row['stage']}\n"
            )
        elif rtype == "final":
            final_result = row
        else:
            sys.stderr.write(f"[ptxd_specialist] unknown row type: {rtype}\n")

    proc.wait(timeout=10)
    if final_result is None:
        sys.stderr.write(
            f"[ptxd_specialist] ptxd produced no final row; stderr:\n"
            f"{proc.stderr.read()}\n"
        )
        mw.update_status(exp_id, "error")
        sys.exit(3)

    elapsed = time.time() - t0
    final_loss = float(final_result.get("final_loss", 0.0))
    best_acc = float(final_result.get("best_acc", 0.0))
    final_status = ("mastered" if best_acc >= args.target_acc
                    else final_result.get("status", "needs_tuning"))
    mw.update_status(exp_id, final_status)
    result = final_result
    mw.log_event("ptxd_done", exp_id, json.dumps({
        "best_acc": best_acc, "final_loss": final_loss,
        "ms_per_step": result.get("ms_per_step"), "wall_ms": result.get("wall_ms"),
    }))

    # Same human-readable summary to stdout that three_populations might tail.
    print(f"[ptxd_specialist] {args.task}  best_acc={best_acc*100:.1f}%  "
          f"loss={final_loss:.4f}  ms/step={result.get('ms_per_step', 0):.2f}  "
          f"({elapsed:.1f}s wall, status={final_status})")
    sys.exit(0 if best_acc >= args.target_acc else 1)


if __name__ == "__main__":
    main()
