"""End-to-end test for Phase 1 streaming protocol — minimal smoke test.

Generates fixed-4-bit parity examples in Python (mirroring the legacy
hardcoded path's distribution), ships them through the streaming
protocol, and verifies training behaves equivalently to the legacy path.

If accuracy curves match, the protocol is correctly wired. If streaming
diverges while legacy doesn't, there's a bug in the protocol path.

Run on the H100:
    python3 test_streaming_parity.py
"""
import json, random, struct, subprocess, sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from batch_writer import write_examples, IGNORE

PTXD = "/root/mamba3-hands-on/engine/ptx/target/release/ptxd"
TMP  = "/tmp/streaming_test"
os.makedirs(TMP, exist_ok=True)


def gen_parity_fixed(n_bits, n_examples, seed):
    """Generate `n_examples` parity examples with FIXED bit length, mirroring
    the legacy scheduler.rs hardcoded path (which uses min_len=max_len=n_bits
    when no curriculum is supplied). Single-position supervision at SEP."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_examples):
        bits = [rng.randint(0, 1) for _ in range(n_bits)]
        parity = sum(bits) & 1
        # Tokens: BOS + bytes("0" or "1") interleaved with " " + SEP + answer + EOS
        # Matches scheduler.rs encoding exactly.
        tokens = [256]  # BOS
        for i, b in enumerate(bits):
            if i > 0:
                tokens.append(32)   # ' '
            tokens.append(48 + b)   # '0' or '1'
        tokens.append(258)          # SEP
        answer = 83 if parity == 0 else 68  # 'S' or 'D'
        tokens.append(answer)
        tokens.append(257)          # EOS
        # Single-position supervision at SEP position. Matches legacy
        # scheduler.rs behaviour: target only at len-3 (SEP position).
        targets = [IGNORE] * len(tokens)
        targets[len(tokens) - 3] = answer
        out.append((tokens, targets))
    return out


def run_job(name, extra):
    job = {
        "id": name,
        "task": "parity",
        "n_bits": 4,
        "d_model": 64, "d_state": 8, "headdim": 16, "n_layers": 4,
        "vocab_size": 260,
        "lr": 1e-3, "weight_decay": 0.1,
        "steps": 1200,
        "batch_size": 256,
        "target_acc": 0.95,
        "seed": 42,
    }
    job.update(extra)
    proc = subprocess.run([PTXD], input=json.dumps(job) + "\n",
                          capture_output=True, text=True, timeout=600)
    final = None
    cycles = []
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
        elif t == "final":
            final = ev
    return final, cycles, proc.stderr


def main():
    # --- Path A: legacy hardcoded parity ---
    print("\n=== A: legacy hardcoded path (no batches_path) ===")
    final_a, cycles_a, _ = run_job("legacy", {})
    for c in cycles_a:
        print(f"    cycle {c['cycle']:2d}  loss={c['loss']:.3f}  acc={c['fresh_acc']*100:5.1f}%")
    print(f"  final: best_acc={final_a['best_acc']:.3f}  status={final_a['status']}")

    # --- Path B: streaming via fixed-4-bit batches (matches legacy distribution) ---
    print("\n=== B: streaming path (fixed 4-bit, matching legacy distribution) ===")
    train_path = f"{TMP}/parity_train.bin"
    eval_path  = f"{TMP}/parity_eval.bin"
    # Generate enough examples to avoid wraparound: 1200 steps × 256 batch =
    # 307K reads. Provide a touch more so the reader never wraps.
    write_examples(train_path, gen_parity_fixed(n_bits=4, n_examples=320000, seed=42))
    write_examples(eval_path,  gen_parity_fixed(n_bits=4, n_examples=200,    seed=99))
    print(f"  wrote {train_path} + {eval_path}")
    final_b, cycles_b, _ = run_job("streaming", {
        "batches_path": train_path,
        "eval_batches_path": eval_path,
    })
    for c in cycles_b:
        print(f"    cycle {c['cycle']:2d}  loss={c['loss']:.3f}  acc={c['fresh_acc']*100:5.1f}%")
    print(f"  final: best_acc={final_b['best_acc']:.3f}  status={final_b['status']}")

    a, b = final_a["best_acc"], final_b["best_acc"]
    print(f"\n=== verdict ===")
    print(f"  legacy   best_acc={a:.3f}")
    print(f"  streaming best_acc={b:.3f}")
    print(f"  difference: {abs(a-b)*100:.1f}%")
    # Equivalence: streaming should reach within 5% of legacy. If both fail
    # to converge in 1200 steps but at the same level, the protocol is fine.
    if abs(a - b) <= 0.05:
        print("  PASS — streaming path is equivalent to legacy")
        sys.exit(0)
    else:
        print("  FAIL — streaming path diverges from legacy")
        sys.exit(1)


if __name__ == "__main__":
    main()
