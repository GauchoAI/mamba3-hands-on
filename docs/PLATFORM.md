# Mamba Platform — Multi-Node Training System

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Firebase RTDB                     │
│              (signaling database)                    │
│                                                      │
│  /mamba3/nodes/{node_id}     ← node manifests       │
│  /mamba3/three_pop/          ← training telemetry    │
│  /mamba3/jobs/{job_id}       ← job tracking          │
│  /mamba3/events/             ← mastery events        │
└──────────┬──────────┬──────────┬────────────────────┘
           │          │          │
     ┌─────┴───┐ ┌────┴────┐ ┌──┴──────┐
     │ H100    │ │Mac Mini │ │Mac Air  │
     │ (cuda)  │ │ (mps)   │ │ (mps)   │
     │ node    │ │ node    │ │ node    │
     │ agent   │ │ agent   │ │ agent   │
     └─────────┘ └─────────┘ └─────────┘
           ▲          ▲          ▲
           │          │          │
     ┌─────┴──────────┴──────────┴────┐
     │          CLI / MCP              │
     │   mamba nodes|status|submit     │
     │   (runs on any machine)        │
     └────────────────────────────────┘
```

## Communication Protocol

### 1. Node Registration

Each training node runs `server/node_agent.py` which:
1. **Probes** local hardware (GPU, CPU, backends, VRAM)
2. **Registers** to Firebase at `/mamba3/nodes/{node_id}`
3. **Heartbeats** every 30s to `/mamba3/nodes/{node_id}/last_heartbeat`

#### Node Manifest (Firebase schema)

```json
{
  "node_id": "h100-vast-001",
  "backends": ["cuda", "triton", "jit", "cpu"],
  "vram_mb": 81079,
  "cpu_cores": 256,
  "max_workers": 8,
  "hostname": "21406a041cb8",
  "platform": "linux",
  "arch": "x86_64",
  "python": "3.11.15",
  "torch_version": "2.11.0+cu128",
  "gpu_name": "NVIDIA H100 80GB HBM3",
  "working_dir": "/root/mamba3-hands-on",
  "status": "online",
  "registered_at": 1745366400.0,
  "last_heartbeat": 1745366430.0,
  "pid": 154206,
  "ssh": {
    "host": "ssh2.vast.ai",
    "port": 32783,
    "user": "root"
  }
}
```

#### Status lifecycle

```
online  → last_heartbeat < 60s ago
stale   → last_heartbeat 60-300s ago
offline → last_heartbeat > 300s ago (or explicitly deregistered)
```

### 2. Job Submission

When `mamba submit --target <node>` is called:

1. CLI resolves `<node>` from Firebase `/mamba3/nodes/`
2. Reads SSH info from the node manifest
3. `git pull` on the remote (syncs code + problem YAMLs)
4. Starts `three_populations.py` on the remote via SSH
5. Registers job at `/mamba3/jobs/{job_id}`

#### Job Record (Firebase schema)

```json
{
  "node_id": "h100-vast-001",
  "pid": "154206",
  "problems_dir": "problems",
  "submitted_at": 1745366400.0,
  "status": "running"
}
```

### 3. Training Telemetry

Each node's `three_populations.py` + `specialist_trainer.py` push to Firebase:

| Path | Data | Updated |
|------|------|---------|
| `/mamba3/three_pop/teachers` | graduated teacher specs | on mastery |
| `/mamba3/three_pop/tasks` | per-task status + config | per round |
| `/mamba3/snapshot/` | GPU%, memory, timestamp | per cycle |
| `/mamba3/events/` | mastery, unlock, evolution events | on event |
| `/mamba3/lineage/{task}/` | genetic lineage nodes | per round |

### 4. Node Discovery

Any CLI or MCP client discovers nodes by reading `/mamba3/nodes/` from Firebase.
No direct node-to-node communication. Firebase is the single rendezvous point.

```python
# From any machine:
from server.node_agent import get_online_nodes
nodes = get_online_nodes()  # returns nodes with heartbeat < 120s
```

### 5. Mutation Auto-Extension

When a node registers, its `backends` list extends the mutation registry:

```python
# In coordinator.py, on startup:
from registry.mutation_registry import MutationRegistry
registry = MutationRegistry()
registry.load(["registry/mutations.yaml"])

# When a new node with ROCm joins:
registry.extend_from_capabilities({"backends": ["rocm", "jit"]})
# Now device=rocm is a valid GA mutation
```

The GA discovers whether new backends are competitive through champion/challenger.

## File Layout

```
mamba3-hands-on/
├── cli/
│   └── main.py              # CLI entry point: mamba nodes|status|submit|logs|pull|stop
├── mcp/
│   ├── server.py            # MCP server for Claude Code integration
│   └── __init__.py
├── server/
│   ├── node_agent.py        # Node registration + heartbeat daemon
│   └── __init__.py
├── registry/
│   ├── problem_registry.py  # ProblemSpec + ProblemRegistry (YAML discovery)
│   ├── mutation_registry.py # MutationSpec + MutationRegistry (data-driven GA)
│   ├── mutations.yaml       # 15 mutation specs
│   └── seed_configs.yaml    # 11 initial population configs
├── problems/                # One subdir per task
│   ├── parity/problem.yaml  # Generator ref + curriculum stages
│   ├── repeat_count/problem.yaml
│   └── ... (15 total)
├── generators/
│   └── level0_patterns.py   # gen_parity(), gen_logic_gate(), etc.
├── export/
│   └── gguf_export.py       # PyTorch .pt → GGUF conversion
├── three_populations.py     # GA orchestrator (stateless, reads from StateDB)
├── specialist_trainer.py    # Worker (one task, one model, self-sufficient)
├── coordinator.py           # GA mutations (delegates to MutationRegistry)
├── state_db.py              # SQLite state (immutable teachers/lineage)
├── diagnostician.py         # Detect training signals, prescribe mutations
├── firebase_push.py         # Push telemetry to Firebase RTDB
├── .mcp.json                # MCP server config for Claude Code
└── pyproject.toml           # Package config, `mamba` CLI entry point
```

## Unified Namespace — Transparent Model Access

Nodes don't reference each other by name to access models. The `ModelRegistry`
provides a unified view: "here are all the teachers" — regardless of which
node trained them. When a student needs a teacher, the registry transparently
fetches it from whatever node has it.

```python
from registry.model_registry import ModelRegistry

reg = ModelRegistry()

# What teachers exist? (from any node, via Firebase index)
reg.available()        # → ["parity", "logic_gate", "cumulative_sum", ...]

# Get the parity teacher — fetched transparently
model, config, acc = reg.get_teacher("parity", device="cpu")
logits = model(input_tokens)

# Sync all teachers locally (for offline use)
reg.sync_all()         # → fetches any missing checkpoints
```

### How it works

1. **Firebase is the index**: `/mamba3/three_pop/teachers` lists all teachers
   with their accuracy, config, and which node trained them
2. **Checkpoints live on nodes**: each node stores `.pt` files locally
3. **Transparent fetch**: when a node needs a teacher it doesn't have,
   `ModelRegistry` finds a node that has it (from Firebase) and rsyncs it
4. **Cached locally**: once fetched, the checkpoint is local — no re-fetch

### Distillation flow

When `specialist_trainer.py` starts training a task:
1. Asks `ModelRegistry`: "is there a teacher for this task?"
2. If yes: fetches checkpoint (transparently, from wherever it lives)
3. Loads teacher model, runs inference every 5th step
4. Blends KL divergence on soft targets into the training loss
5. Logs `distilled_from` in lineage for provenance

This means: a teacher trained on the H100 automatically helps students
training on the Mac Mini, with zero manual intervention.

## Node Setup

### Quick start (any machine)

```bash
git clone https://github.com/GauchoAI/mamba3-hands-on.git
cd mamba3-hands-on
pip install -e .                          # installs `mamba` CLI
python server/node_agent.py --register    # registers + heartbeat loop
```

### Start training

```bash
# Option A: start locally
python server/node_agent.py --start

# Option B: submit from another machine
mamba submit --target <node-id>
```

### Verify

```bash
mamba nodes           # see all registered nodes
mamba status          # see task training progress
mamba logs --node h100 --follow   # stream live logs
```

## Problem Registration

### Generator-based

```yaml
# problems/my_new_task/problem.yaml
name: my_new_task
type: generator
generator: generators.my_module:gen_my_task
target_accuracy: 0.95
tags: [reasoning, level1]
description: "My new reasoning task"
curriculum:
  - {stage: 1, difficulty: easy, advance_at: 0.90}
  - {stage: 2, difficulty: hard, advance_at: 0.95}
```

The generator function:
```python
# generators/my_module.py
def gen_my_task(difficulty="easy"):
    if difficulty == "easy":
        # simpler examples
    else:
        # harder examples
    return {"type": "my_new_task", "input": "...", "output": "..."}
```

### Dataset-based

```yaml
name: my_dataset_task
type: dataset
dataset: data/my_task.jsonl
target_accuracy: 0.90
tags: [data, level1]
```

### Via MCP (Claude Code)

```
mamba_register_problem(
  name="fibonacci_next",
  generator_module="generators.level1_math",
  generator_function="gen_fibonacci_next",
  target_accuracy=0.95
)
```

## Curriculum Ratchet

Difficulty only goes up. The system tracks `current_stage` per task in StateDB.

```
Stage 1: gen_parity(max_len=4)     → advance when acc ≥ 90%
Stage 2: gen_parity(max_len=8)     → advance when acc ≥ 90%
Stage 3: gen_parity(max_len=16)    → mastery when acc ≥ 95%
```

The GA mutates architecture (d_model, layers, lr, etc.) independently.
Difficulty is orthogonal — controlled by the curriculum, not the GA.

## Security Model

- Firebase RTDB is the only shared state (no direct SSH between nodes)
- SSH credentials are stored per-node in Firebase manifests
- The CLI uses SSH to deploy code and start processes
- No secrets in the repo — Firebase URL is public (RTDB rules should restrict writes)
- Node agent sends heartbeats but doesn't accept inbound connections
