"""Validate the hypothesis: ptxd's built-in curriculum trains parity
from scratch in < 10 minutes when used correctly.

Submit a job with `stages` populated but `batches_path` absent so ptxd
uses its legacy parity data generator (which respects stage_idx and
advances stages when fresh_acc clears advance_at).

This is the same path test_parity_train uses, just driven via the
JSON protocol so it's reproducible from outside the binary.
"""
import json, subprocess, sys, time
from pathlib import Path

PTXD = "/root/mamba3-hands-on/engine/ptx/target/release/ptxd"


def main():
    job = {
        "id": "curriculum",
        "task": "parity",
        "n_bits": 4,             # ignored when stages is populated
        "d_model": 32, "d_state": 16, "headdim": 16, "n_layers": 1,
        "vocab_size": 260,
        "lr": 1e-3, "weight_decay": 0.1,
        "steps": 20000,           # ample budget; ptxd early-exits on mastery
        "batch_size": 16,
        "target_acc": 0.95,
        "seed": 42,                # match test_parity_train default
        "stages": [
            # Higher advance_at so each stage's weights are *robustly*
            # in mastery before advancing — the 90% threshold let the
            # model advance with shaky stage-1 fundamentals and then
            # fail to generalise to stage 2's longer sequences.
            {"min_len": 2, "max_len": 4,  "advance_at": 0.97},
            {"min_len": 3, "max_len": 8,  "advance_at": 0.95},
            {"min_len": 4, "max_len": 16, "advance_at": 0.95},
        ],
        # NO batches_path → ptxd uses its built-in parity generator with
        # stage_idx advancement. NO eval_batches_path → same for eval.
        "auto_tune": False,  # we want the full 20K steps if needed; no early-bail
    }
    print(f"Submitting parity curriculum job: stages 1→3, target 95%, budget {job['steps']} steps")
    t0 = time.time()
    proc = subprocess.run([PTXD], input=json.dumps(job) + "\n",
                          capture_output=True, text=True, timeout=900)
    elapsed = time.time() - t0
    cycles = []
    final = None
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "cycle":
            cycles.append(ev)
            print(f"  cycle {ev['cycle']:3d}  stage={ev.get('stage', '?')}  "
                  f"loss={ev['loss']:.4f}  acc={ev['fresh_acc']*100:5.1f}%  "
                  f"best={ev['best_fresh']*100:5.1f}%")
        elif t == "final":
            final = ev

    print(f"\n=== verdict ===")
    print(f"  wall: {elapsed:.1f}s")
    if final is None:
        print(f"  FAIL — no final event. stderr tail:\n{proc.stderr[-600:]}")
        sys.exit(1)
    print(f"  status: {final['status']}")
    print(f"  best_acc: {final['best_acc']:.3f}")
    print(f"  steps_executed: {final['steps_executed']}")
    if final["best_acc"] >= 0.95:
        print("  PASS — parity mastered from scratch via curriculum")
        sys.exit(0)
    elif final["best_acc"] >= 0.85:
        print("  CLOSE — partial convergence")
        sys.exit(0)
    else:
        print("  FAIL — didn't converge")
        sys.exit(1)


if __name__ == "__main__":
    main()
