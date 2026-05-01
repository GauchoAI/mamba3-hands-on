# Mac branch layout

This branch is the **Apple Silicon daily driver**. The PTX/CUDA/ptxd engine
that targeted vast.ai H100 pods has moved to the `pod-archive` branch.

## What runs here

| Layer | File / dir | Notes |
|---|---|---|
| Trainer | `specialist_trainer.py` | PyTorch + MPS (auto-detects on Mac), distillation built in, regression guard |
| GA orchestrator | `three_populations.py` | Spawns `specialist_trainer.py` workers; mutation surface in `registry/mutations.yaml` |
| Sweep harness | `mac_sweep.py` | Sequential one-task-at-a-time sweep over the 37-task suite |
| Tasks | `problems/`, `generators/` | Per-task data generators; YAML curricula |
| Inference engine | `engine/wgpu/` | WebGPU/WGSL — runs on Metal natively. Inference-only |
| CPU reference | `engine/mamba3-engine/` | Rust CPU implementation (parity reference) |
| Dashboard | `serve_local.sh`, `server/` | Static HTML on :9090. `serve_dashboard.sh` polls H100 — not used here |
| State | `checkpoints/specialists/*.pt`, `metrics_db.py` | Specialist checkpoints + metrics SQLite |

## Quick-start

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy pyyaml
.venv/bin/python mac_sweep.py --quick                  # 6 representative tasks
.venv/bin/python mac_sweep.py                           # all 37 tasks (~1h)
.venv/bin/python specialist_trainer.py --task addition  # single task
.venv/bin/python three_populations.py                   # full GA loop
```

## What's in `pod-archive` and not here

| Removed | Reason |
|---|---|
| `engine/ptx/`, `engine/ptx-bench/` | NVIDIA PTX engine — needs CUDA |
| `ptxd_specialist.py`, `ptxd_tail.py` | PTX-engine wrappers |
| `test_parallel_sweep.py`, `test_multi_task_sweep.py`, `sweep_and_diff.py` | Parallel sweep harnesses for ptxd |
| `test_phase{2,4,5}*.py`, `test_streaming_parity.py`, `test_regression_guard.py`, `test_kd_correctness.py`, `test_per_task_sweep.py`, `test_from_scratch.py`, `test_parity_curriculum.py` | PTX-specific phase tests |
| `test_cpu_vs_cuda.py`, `test_backend_forced.py`, `test_scan_parity.py` | CUDA/Triton-only tests |
| `ssm_triton.py` | Triton (CUDA-only) SSM scan |
| `ckpt_bridge.py`, `batch_writer.py`, `task_runner.py`, `teacher.py` | BTCH binary protocol + ptxd glue |
| `PTX_ENGINE.md` | PTX engine docs |

If you need any of those, `git checkout pod-archive -- <path>` cherry-picks
without switching branches. If you actually need a stable CUDA host again,
`git switch pod-archive` brings the whole engine back.

## Cluster plan

User is buying 2 more M4 Mac minis ~2026-05-06 → 3-node Apple-silicon cluster.
The natural shape is one task per Mac at a time; `three_populations.py`'s
`spawn_worker()` would just need an SSH variant to fan jobs out across nodes.
The wgpu engine (Metal) becomes more interesting once we want fast cross-node
teacher-distill passes.

### Cluster topology — each node thinks it's solo

Nodes coordinate exclusively through Firebase Realtime DB. **No SSH or
peer-to-peer rsync between training nodes.** This means:

- A node never has to know other nodes exist by name.
- No `Remote Login` flag has to be enabled in macOS Sharing.
- Adding a fourth Mac is "rsync the repo, install torch, run node_agent" — done.

The flow:

```
┌─ specialist_trainer.py on Node A ────────────────────────────┐
│  1. trains task X to mastery                                  │
│  2. saves checkpoints/specialists/X.pt locally                │
│  3. firebase_push.upload_teacher_blob(X, X.pt)                │
│       → mamba3/teacher_blobs/X = base64(X.pt)                 │
│       → mamba3/teacher_blobs_meta/X = {size, sha256, ts}      │
└───────────────────────────────────────────────────────────────┘

┌─ specialist_trainer.py on Node B (independent run) ───────────┐
│  1. starts training task X                                    │
│  2. ModelRegistry.has_teacher(X) hits Firebase, sees X listed │
│  3. ensure_local(X) → download_teacher_blob fetches the .pt   │
│  4. Distillation kicks in: 9× KL on teacher logits            │
│      Node B has no idea Node A exists.                        │
└───────────────────────────────────────────────────────────────┘
```

**Files involved:**

| File | Role |
|---|---|
| `firebase_push.py` | `upload_teacher_blob` / `download_teacher_blob` (with sha256 verification) |
| `registry/model_registry.py` | `ensure_local` — Firebase blob path is primary; SSH/rsync is a legacy fallback |
| `server/node_agent.py` | per-node heartbeat + capability registration; run with `--register` |
| `specialist_trainer.py` | auto-uploads after Saved (skipped at acc=0) |

**Firebase paths used:**

```
mamba3/nodes/<node_id>           — capability + heartbeat
mamba3/three_pop/teachers/<task> — teacher index (acc, source_node)
mamba3/teacher_blobs/<task>      — base64 .pt (the data plane)
mamba3/teacher_blobs_meta/<task> — size + sha256 + uploaded_at
```

**Cost / latency:** ~900KB per .pt → ~1.2MB base64 → 3s upload over home Wi-Fi,
3s download. 37 tasks ≈ 45MB total in Firebase, well inside the 1GB free tier.

**Bringing up a new Mac (any model with PyTorch+MPS):**

```bash
# On the new Mac, Wi-Fi + Mac on the LAN:
git clone <repo>             # or rsync from another node
cd mamba3-hands-on
python3.13 -m venv .venv
.venv/bin/pip install torch numpy pyyaml
.venv/bin/python server/node_agent.py --register &     # heartbeats every 30s
.venv/bin/python three_populations.py                  # joins the cluster
```

That's it. The node will:
- pull teachers from Firebase as it needs them,
- publish its own mastered teachers back,
- never directly touch any other node.
