"""Phase 4 smoke test — Loss::CeKd kernel runs end-to-end.

Generates parity batches WITH synthetic teacher logits and ships them
to ptxd as a Loss::CeKd job. The teacher is fake (Gaussian noise) so
this isn't a convergence test — it just exercises the kd_apply kernel
path: BatchReader parses v2, scheduler launches kd_apply between CE and
the rest of backward, no crashes.

Run:
    python3 test_phase4_distillation.py
"""
import json, random, subprocess, sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from batch_writer import write_examples, IGNORE

PTXD = "/root/mamba3-hands-on/engine/ptx/target/release/ptxd"
TMP  = "/tmp/phase4_test"
os.makedirs(TMP, exist_ok=True)
V = 260


def gen_examples_with_synthetic_teacher(n_examples, n_bits, seed):
    """Generate parity examples + synthetic teacher logits at the SEP
    position. Teacher is deliberately fake (Gaussian) so we're testing
    the kernel pipeline, not convergence."""
    rng = random.Random(seed)
    examples = []
    teacher_logits = []
    for _ in range(n_examples):
        bits = [rng.randint(0, 1) for _ in range(n_bits)]
        parity = sum(bits) & 1
        toks = [256]
        for i, b in enumerate(bits):
            if i > 0: toks.append(32)
            toks.append(48 + b)
        toks.append(258)
        ans = 83 if parity == 0 else 68
        toks.append(ans)
        toks.append(257)
        # Single supervised position at SEP (matches legacy parity).
        sup_pos = len(toks) - 3
        tgts = [IGNORE] * len(toks)
        tgts[sup_pos] = ans
        examples.append((toks, tgts))
        # Synthetic teacher: Gaussian noise centred near zero. Real Phase 4
        # work loads a teacher from .pt and runs its forward at sup_pos.
        logits = [rng.gauss(0.0, 1.0) for _ in range(V)]
        # Make the correct answer slightly favoured so KD has SOME signal,
        # otherwise the test devolves to "teacher is pure noise."
        logits[ans] += 2.0
        teacher_logits.append([(sup_pos, logits)])
    return examples, teacher_logits


def run_job(name, batches_path, eval_batches_path, kd_weight=0.0, temperature=1.0):
    job = {
        "id": name,
        "task": "parity",
        "n_bits": 4,
        "d_model": 32, "d_state": 16, "headdim": 16, "n_layers": 1,
        "vocab_size": V,
        "lr": 1e-3, "weight_decay": 0.1,
        "steps": 100, "batch_size": 16,
        "target_acc": 0.95,
        "seed": 7,
        "batches_path": batches_path,
        "eval_batches_path": eval_batches_path,
        "loss": ({"type": "ce_kd", "kd_weight": kd_weight, "temperature": temperature}
                 if kd_weight > 0 else {"type": "ce"}),
    }
    proc = subprocess.run([PTXD], input=json.dumps(job) + "\n",
                          capture_output=True, text=True, timeout=180)
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


def main():
    # Generate batches with teacher logits (v2 format).
    examples, teachers = gen_examples_with_synthetic_teacher(
        n_examples=2000, n_bits=4, seed=42)
    eval_examples, eval_teachers = gen_examples_with_synthetic_teacher(
        n_examples=200, n_bits=4, seed=99)

    train_v2  = f"{TMP}/parity_kd_train.bin"
    eval_v2   = f"{TMP}/parity_kd_eval.bin"
    write_examples(train_v2, examples, teacher_logits=teachers, vocab_size=V)
    write_examples(eval_v2,  eval_examples, teacher_logits=eval_teachers, vocab_size=V)
    print(f"wrote {train_v2} and {eval_v2} (v2 with synthetic teacher logits)")

    # Run a Loss::CeKd job. Just verify it completes without crashing.
    print("\n=== running Loss::CeKd (kd_weight=0.3, temperature=2.0) ===")
    cycles, final, stderr = run_job("kd", train_v2, eval_v2,
                                     kd_weight=0.3, temperature=2.0)
    if final is None:
        print(f"FAIL — no final event. stderr tail:\n{stderr[-1200:]}")
        sys.exit(1)
    print(f"  ms_per_step={final.get('ms_per_step', 0):.2f}  "
          f"loss={final.get('final_loss', 0):.3f}  "
          f"status={final.get('status')}")
    # Should not see "kernel not yet implemented" — distillation is now real.
    if "ce_kd" in stderr and "not yet implemented" in stderr:
        print("FAIL — Loss::CeKd is still warning about unimplemented kernel")
        print(f"stderr: {stderr[:600]}")
        sys.exit(1)
    print("PASS — Loss::CeKd ran end-to-end (kd_apply kernel exercised)")


if __name__ == "__main__":
    main()
