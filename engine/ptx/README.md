# PTX Mamba-3 training engine

Hand-written PTX kernels for Mamba-3 training on NVIDIA H100. Bit-clean against PyTorch autograd; ~14× faster than PyTorch CPU on the same training task.

> Don't read this if you want narrative — see `findings.md` Entries 26–37 in the repo root for the full session arc, methodology, and design decisions.

## Quick start (on the H100)

```bash
cd engine/ptx
cargo build --release --bin fd-check --bin forward-parity \
                       --bin single-step-check --bin parity-replay \
                       --bin test-parity-train --bin ptxd

# 1. Gradient correctness (15s)
./target/release/fd-check
# expect: most tensors PASS, a few "noise" near the FD floor

# 2. Forward parity vs PyTorch reference (5s, requires Python+torch + dump_pytorch_run.py)
python3 /tmp/forward_parity_export.py   # writes /tmp/forward_parity*.bin
./target/release/forward-parity
# expect: BIT-PARITY ✓  (max_abs ≤ 1e-5)

# 3. One training step bit-equality vs PyTorch autograd (5s)
python3 /tmp/single_step_compare.py
./target/release/single-step-check
# expect: every tensor max_abs ≤ 1e-4

# 4. Multi-step training trajectory match (3.5s on H100, vs 51s for PyTorch CPU)
python3 /tmp/dump_pytorch_run.py
./target/release/parity-replay
# expect: PTX matches PyTorch's per-cycle accuracy column, hits 100% in cycle 2
```

## Four correctness gates

The training engine is verified against PyTorch in four layers, each catching a different class of bug:

| Gate | What it checks | Catches |
|---|---|---|
| `fd-check` | Analytical gradients match finite-difference of *our* loss | Arithmetic kernel bugs in single-sample backward |
| `forward-parity` | Forward output equals `mamba3_minimal.Mamba3Block` to FP32 noise | Forward kernel divergence from reference architecture |
| `single-step-check` | Post-AdamW weights match PyTorch autograd after ONE step from identical init | Missing gradient paths, optimizer-config mismatches |
| `parity-replay` | Multi-step trajectory matches PyTorch on identical data + init | Batched-accumulation bugs, training-loop bookkeeping |

Run all four after every kernel change. Each gate is fast enough to be a daily check.

## Daemon: `ptxd`

Single-process, sequential JSON job runner. Reads jobs from stdin (one JSON object per line), writes streaming results to stdout.

### Job schema

```json
{
  "id": "string — for correlating output rows",
  "task": "parity",
  "d_model": 32,        "d_state": 16,    "headdim": 16,
  "n_layers": 1,        "vocab_size": 260,
  "lr": 0.001,          "weight_decay": 0.1,
  "batch_size": 16,     "steps": 5000,
  "n_bits": 4,          "target_acc": 0.95,
  "seed": 12345,

  // optional curriculum (mirrors problems/parity/problem.yaml)
  "stages": [
    {"min_len": 2, "max_len": 4, "advance_at": 0.90},
    {"min_len": 3, "max_len": 8, "advance_at": 0.90},
    {"min_len": 4, "max_len": 16, "advance_at": 0.95}
  ]
}
```

### Output stream

```jsonl
{"type":"cycle", "id":"j1", "cycle":1, "step":200, "loss":..., "fresh_acc":..., "best_fresh":..., "stage":0, "elapsed_s":...}
{"type":"cycle", ...}
{"type":"final", "id":"j1", "status":"converged", "final_loss":..., "best_acc":..., "ms_per_step":..., "wall_ms":...}
```

`type:cycle` rows arrive every 200 training steps. `type:final` arrives once when the job ends (target accuracy hit, or `steps` exhausted).

## Drop-in replacement for `specialist_trainer.py`

`../../ptxd_specialist.py` (in repo root) is a Python shim that accepts the same CLI surface `three_populations.py.spawn_worker` uses, ships a single ptxd job, and writes the same `MetricsWriter` rows. To switch the GA from PyTorch to PTX training:

```diff
-    cmd = [sys.executable, "-u", "specialist_trainer.py",
+    cmd = [sys.executable, "-u", "ptxd_specialist.py",
```

Set `PTXD_BIN=/path/to/ptxd` if the binary lives outside the default location. Falls back to `specialist_trainer.py` for non-parity tasks.

## Layout

```
engine/ptx/
├── src/
│   ├── ptx/kernels.cu        # all CUDA C kernels (forward + backward + AdamW + reductions)
│   ├── runtime.rs            # PtxContext: NVRTC compile, kernel handles, CUDA stream
│   ├── model.rs              # PtxModel: forward + forward_cached
│   ├── trainer.rs            # PtxTrainer: backward, AdamW, gradient accumulation, AdamW scaling
│   ├── train_scratch.rs      # Per-layer activation cache + gradient buffers
│   ├── scratch.rs            # Inference scratch (smaller, no backward state)
│   └── bin/
│       ├── ptxd.rs                   # JSON daemon (above)
│       ├── fd-check.rs               # gate 1
│       ├── forward_parity.rs         # gate 2
│       ├── single_step_check.rs      # gate 3
│       ├── parity_replay.rs          # gate 4
│       ├── test_parity_train.rs      # standalone parity training (dev/exploration)
│       ├── test_train_kernels.rs     # individual-kernel tests
│       └── ...
└── README.md (this file)
```

## Performance

H100, default parity config (d=32 L=1 dS=16 batch=16, 4 cycles × 200 steps):

- PTX (this engine):           **3.5s**, 100% accuracy
- PyTorch CPU (mamba3_minimal): 50.8s, 100% accuracy
- 14× speedup, same convergence trajectory

Forward inference (cached scratch, no backward), 7 tokens, d=32: **~0.27ms/token**. CUDA Graph capture available for further latency reduction (see `findings.md` Entry 24).

## What's NOT done yet

1. **Tasks beyond parity in ptxd.** `specialist_trainer.py` reads YAML from `problems/<task>/problem.yaml`; ptxd hardcodes parity. Adding a task = one `run_<task>` function + token/eval contract.
2. **Slot scheduler.** ptxd is single-process sequential. The Tetris-like packing for concurrent jobs on one GPU (memory + SM budgeting) is the real upgrade — needs multi-stream CUDA contexts.
3. **Inference-only fast path** for the GA's eval loop. Currently `PtxModel::forward` allocates per call; a pre-bound persistent eval mode would shave latency.

Each is small-to-medium scope. The kernel/correctness work — the hard part — is done.
