"""Per-task verification sweep — runs ptxd training on a handful of tasks
through the streaming protocol and checks that loss decreases. Not a
convergence test (600 steps is far too few for most of these); just
proves "the path doesn't break for any of these."

Tasks chosen to span the difficulty/length spectrum:
- parity (single byte answer, smallest)
- cumulative_sum (multi-byte numeric answer)
- max_element (multi-byte; pick from sequence)
- alternating_next (single byte; pattern continuation)
- duplicate_detect (single byte 'D'/'U')
- binary_pattern_next (single byte continuation)
"""
import json, subprocess, sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from task_runner import write_task_batches

PTXD = "/root/mamba3-hands-on/engine/ptx/target/release/ptxd"
TMP  = "/tmp/per_task_sweep"
os.makedirs(TMP, exist_ok=True)


def run_task(task, steps=400, batch=64):
    train_path = f"{TMP}/{task}_train.bin"
    eval_path  = f"{TMP}/{task}_eval.bin"
    n_train = max(20000, batch * 64)
    try:
        write_task_batches(train_path, task, n_examples=n_train, stage=0, seed=42)
        write_task_batches(eval_path,  task, n_examples=200,    stage=0, seed=99)
    except Exception as e:
        return None, None, f"batch generation failed: {e}"

    job = {
        "id": task[:12],
        "task": task,
        "n_bits": 4,
        "d_model": 64, "d_state": 16, "headdim": 16, "n_layers": 2,
        "vocab_size": 260,
        "lr": 1e-3, "weight_decay": 0.1,
        "steps": steps, "batch_size": batch,
        "target_acc": 0.95,
        "seed": 42,
        "batches_path": train_path,
        "eval_batches_path": eval_path,
    }
    proc = subprocess.run([PTXD], input=json.dumps(job) + "\n",
                          capture_output=True, text=True, timeout=300)
    cycles, final = [], None
    for line in proc.stdout.splitlines():
        if not line.startswith("{"): continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError: continue
        t = ev.get("type")
        if t == "cycle": cycles.append(ev)
        elif t == "final": final = ev
    return cycles, final, proc.stderr


TASKS = [
    "parity",
    "cumulative_sum",
    "max_element",
    "alternating_next",
    "duplicate_detect",
    "binary_pattern_next",
]


def main():
    results = {}
    for task in TASKS:
        print(f"\n=== {task} ===", flush=True)
        cycles, final, stderr = run_task(task)
        if final is None:
            print(f"  FAIL — no final event")
            print(f"  stderr tail: {stderr[-400:]}" if stderr else "")
            results[task] = ("ERROR", None, None)
            continue
        for c in cycles:
            print(f"  cycle {c['cycle']:2d}  loss={c['loss']:.3f}  acc={c['fresh_acc']*100:5.1f}%")
        c1_loss = cycles[0]["loss"] if cycles else float("nan")
        late_min = min((c["loss"] for c in cycles[len(cycles)//2:]), default=float("inf"))
        making_progress = late_min < c1_loss if cycles else False
        verdict = "TRAIN" if making_progress else "STUCK"
        print(f"  final acc={final['best_acc']:.3f}  loss progress: {c1_loss:.3f} → {late_min:.3f}  [{verdict}]")
        results[task] = (verdict, c1_loss, late_min)

    print("\n=== summary ===")
    n_train = sum(1 for v, _, _ in results.values() if v == "TRAIN")
    n_stuck = sum(1 for v, _, _ in results.values() if v == "STUCK")
    n_error = sum(1 for v, _, _ in results.values() if v == "ERROR")
    for task, (verdict, c1, c2) in results.items():
        msg = f"  {task:24s} {verdict}"
        if c1 is not None:
            msg += f"   loss {c1:.2f} → {c2:.2f}"
        print(msg)
    print(f"\n  {n_train}/{len(TASKS)} tasks training, {n_stuck} stuck, {n_error} errored")
    if n_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
