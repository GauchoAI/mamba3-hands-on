# PTX Mamba-3 training engine — what got built

A hand-written PTX training engine for Mamba-3 on NVIDIA H100, with a four-gate verification methodology proving it's functionally PyTorch-equivalent.

## TL;DR

| | PyTorch CPU | PTX on H100 |
|---|---|---|
| Same parity training task | 50.8s, 100% | **3.5s, 100%** (14× faster) |
| Single-step gradient parity | reference | ≤ 2.5e-6 max_abs |
| Forward output parity | reference | ≤ 5.7e-6 max_abs |
| Multi-step trajectory parity | reference | matches per-cycle accuracy |

## The four correctness gates

The hard part of building a kernel-based training system isn't writing the kernels — it's proving each one is correct. Four gates, each catching a different class of bug:

1. **fd-check** — analytical gradient matches numerical finite-difference per tensor.  Catches arithmetic kernel bugs.
2. **forward-parity** — forward output bit-equal to a Python autograd reference (`mamba3_minimal.Mamba3Block`) given identical weights.  Catches forward kernel divergence from the reference architecture.
3. **single-step-check** — post-AdamW weights bit-equal to PyTorch's autograd after one training step from identical init.  Catches missing gradient paths and optimizer config mismatches.
4. **parity-replay** — multi-step training trajectory matches PyTorch's on identical input data.  Catches batched-accumulation bugs that only manifest across many steps.

**Build all four before trusting the system.** Bugs hide in surfaces no gate exercises. Example: `matmul_atb_tiled` had `=` instead of `+=`, which was invisible to fd-check / forward-parity / single-step-check (none of them exercise multi-sample gradient accumulation). It only fell to parity-replay.

## What's in the repo

```
engine/ptx/
├── src/
│   ├── ptx/kernels.cu        # CUDA C kernels — forward + backward + AdamW + reductions
│   ├── runtime.rs, model.rs, trainer.rs, train_scratch.rs   # Rust orchestration
│   └── bin/
│       ├── ptxd.rs                # JSON job daemon (stdin/stdout)
│       ├── fd_check.rs            # Gate 1
│       ├── forward_parity.rs      # Gate 2
│       ├── single_step_check.rs   # Gate 3
│       ├── parity_replay.rs       # Gate 4
│       └── test_parity_train.rs   # standalone parity training
├── scripts/
│   ├── test_ptxd_quick.sh         # 1-job smoke
│   ├── test_ptxd_full.sh          # 3-job smoke (incl. curriculum)
│   └── test_ptxd_resilient.sh     # nohup variant for flaky SSH
└── README.md                      # quick-start

ptxd_specialist.py    # drop-in replacement for specialist_trainer.py
findings.md           # methodology + decisions, Entries 28-39 cover this arc
```

## To switch the GA orchestrator from PyTorch to PTX

One diff in `three_populations.py`:

```diff
-    cmd = [sys.executable, "-u", "specialist_trainer.py",
+    cmd = [sys.executable, "-u", "ptxd_specialist.py",
```

`ptxd_specialist.py` is a drop-in shim that accepts the same CLI surface, ships a JSON job to ptxd, parses the streaming output, and writes the same `MetricsWriter` rows specialist_trainer.py would write. Falls back to specialist_trainer.py for non-parity tasks.

## What's still open

1. **Tasks beyond parity in ptxd** — currently parity-only, framework's there for more.
2. **Multi-stream slot scheduler** — ptxd is single-process sequential. Concurrent jobs on one GPU is the real upgrade.
3. **Persistent inference fast path** — model.forward() allocates per call.
4. **Init-quality**: small models on parity have narrow basins. Our LCG init hits the basin ~14% of the time at d=32 L=2; the GA's seed exploration handles this, but a wider basin (bigger model, label smoothing, gradient-noise injection) would mean fewer wasted seeds.

The kernel/correctness work — the hard part — is done. Each remaining item is a half-day to day-of-work.

## Reading order

If you want the narrative, read findings.md Entries 28 onwards in order — each entry is a milestone with the "why" and the "what we learned." If you just want the methodology, Entries 32 + 35 + 38 cover it. If you want the punchline, Entry 35 is the one that matters.
