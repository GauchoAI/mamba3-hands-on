"""Phase 2/3 verification: ptxd is task-agnostic via streaming protocol.

Trains TWO tasks through the same code path:
- parity (single-byte answer "S"/"D") — regression check
- cumulative_sum (multi-byte answer like "12") — proves multi-position
  supervision works end-to-end

Both should drive loss DOWN over the cycles. Convergence to mastery
isn't expected in 600 steps; the point is to show that:
1. ptxd accepts non-parity batches and trains on them
2. multi-position eval works (no crash, sensible accuracy)
3. parity still works after the eval refactor (no regression)
"""
import json, subprocess, sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from task_runner import write_task_batches

PTXD = "/root/mamba3-hands-on/engine/ptx/target/release/ptxd"
TMP  = "/tmp/phase2_test"
os.makedirs(TMP, exist_ok=True)


def run_job(task, train_path, eval_path, steps=600, batch=128):
    job = {
        "id": task,
        "task": task,
        "n_bits": 4,  # ignored when batches_path is set
        "d_model": 64, "d_state": 8, "headdim": 16, "n_layers": 4,
        "vocab_size": 260,
        "lr": 1e-3, "weight_decay": 0.1,
        "steps": steps,
        "batch_size": batch,
        "target_acc": 0.95,
        "seed": 42,
        "batches_path": train_path,
        "eval_batches_path": eval_path,
    }
    proc = subprocess.run([PTXD], input=json.dumps(job) + "\n",
                          capture_output=True, text=True, timeout=600)
    cycles, final = [], None
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "cycle": cycles.append(ev)
        elif t == "final": final = ev
    return cycles, final, proc.stderr


def test_one(task):
    print(f"\n=== {task} ===")
    train_path = f"{TMP}/{task}_train.bin"
    eval_path  = f"{TMP}/{task}_eval.bin"
    # Generate enough train examples to avoid wraparound: 600 steps × 128
    # batch = 76,800 reads. 80K examples covers it.
    write_task_batches(train_path, task, n_examples=80000, stage=0, seed=42)
    write_task_batches(eval_path,  task, n_examples=200,   stage=0, seed=99)
    cycles, final, stderr = run_job(task, train_path, eval_path)
    for c in cycles:
        print(f"  cycle {c['cycle']:2d}  loss={c['loss']:.3f}  acc={c['fresh_acc']*100:5.1f}%")
    if final is None:
        print(f"  ERROR — no final event. stderr tail:\n{stderr[-800:]}")
        return False
    print(f"  final: best_acc={final['best_acc']:.3f}  status={final['status']}")
    # Pass if (a) the job completed without crashing, and (b) the loss
    # decreased over time (training is making progress on the new path).
    if len(cycles) < 2:
        return final is not None
    # Check loss trend: late-cycle min should be lower than cycle-1 loss.
    cycle_1_loss = cycles[0]["loss"]
    late_min_loss = min(c["loss"] for c in cycles[len(cycles)//2:])
    making_progress = late_min_loss < cycle_1_loss
    print(f"  loss progress: cycle1={cycle_1_loss:.3f} → late_min={late_min_loss:.3f}  {'✓' if making_progress else '✗'}")
    return making_progress


def main():
    results = {}
    for task in ["parity", "cumulative_sum"]:
        results[task] = test_one(task)

    print("\n=== verdict ===")
    for task, ok in results.items():
        print(f"  {task}: {'PASS' if ok else 'FAIL'}")
    if all(results.values()):
        print("PASS — ptxd is task-agnostic through streaming protocol")
        sys.exit(0)
    else:
        print("FAIL — at least one task didn't train")
        sys.exit(1)


if __name__ == "__main__":
    main()
