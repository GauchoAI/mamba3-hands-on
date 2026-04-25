# Register-state hooks — design notes

Future-extension surface for two related capabilities the user wants:

1. **Register inspection** — read the SSM hidden state per timestep so the GA / dashboards can introspect what the model has stored.
2. **Model composition** — let model A's output state seed model B's input state, so trained skills can be chained at inference.

This document specs the API shape so adding the actual implementation doesn't require refactoring the rest of the engine.

## What "register state" means in this engine

Each Mamba-3 SSM layer tracks a hidden state of shape `(n_heads, headdim, d_state)`. For typical configs (`d_model=64, headdim=16, d_state=16`) that's `4 × 16 × 16 = 1024` values per layer per timestep — close to the user's "2000 registers per layer" mental model. With `n_layers=4` and `d_state=32` it lands at the 8192 mark.

The state evolves as the SSM processes tokens left-to-right:

```
S[t] = decay[t] * S[t-1] + B(x[t]) * (something derived from x[t])
y[t] = C(x[t]) * S[t]
```

(See `kernels.cu::ssm_scan_cached` for the exact recurrence; the cache writes `states[(t+1)·H·hd·ds + ...]` so the backward pass can revisit them.)

## Capabilities to expose

### 1. Read-out at every timestep

```rust
pub struct StateSnapshot {
    pub layer: usize,         // which SSM layer
    pub timestep: usize,      // 0..L-1 in the input sequence
    pub state: Vec<f32>,      // (n_heads * headdim * d_state) row-major
}

impl PtxModel {
    /// Forward pass that ALSO returns per-layer per-timestep state. Calls
    /// `ssm_scan_cached` (which already writes the full state cache for the
    /// backward pass), then memcpy_dtov's the cache into host vectors. One
    /// stream sync at the end. Cost is ~free during forward; the data is
    /// already in GPU memory because the backward needs it.
    pub fn forward_with_states(
        &self,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, Vec<StateSnapshot>), Box<dyn Error>>;
}
```

This is the introspection primitive. The register-inspector dashboard calls it on a small sample, computes statistics (state norms, top-k active register indices per timestep), and pushes them to Firebase.

### 2. State injection at the start of forward

```rust
pub struct InitialState {
    pub layer: usize,
    pub state: Vec<f32>,  // (n_heads * headdim * d_state)
}

impl PtxModel {
    /// Forward pass that starts from a NON-ZERO initial state per layer.
    /// `init_states` need not cover every layer — layers not specified
    /// start zeroed as today. This is the primitive for model-A-feeds-
    /// model-B composition: A.forward returns its last-timestep state via
    /// `forward_with_states`, B.forward consumes those as `init_states`.
    pub fn forward_from_state(
        &self,
        tokens: &[u32],
        init_states: &[InitialState],
    ) -> Result<Vec<f32>, Box<dyn Error>>;
}
```

The kernel change is small: `ssm_scan_cached` currently zeroes `states[0]` at the host before launching. We add an optional non-zero seed in that buffer. The scan kernel reads `states[t]` (from any source) and writes `states[t+1]`.

For composition, both models must agree on `(n_heads, headdim, d_state)` — otherwise we'd need a learned projection. That's a future concern; start with same-shape composition.

## How this plugs into ptxd's daemon

```jsonc
// Regular training job (existing):
{"id": "j1", ..., "loss": {"type":"ce"}}

// New: forward-only inspect job — returns state snapshots, no training
{"id": "inspect_42", "type": "inspect",
 "init_from_bin": "/path/checkpoint.bin",
 "tokens_path": "/tmp/inspect_tokens.bin",
 "state_out_path": "/tmp/state_out.bin"}

// New: composition forward — A's state seeds B
{"id": "compose_42", "type": "forward_compose",
 "model_a_bin": "/path/A.bin",
 "model_b_bin": "/path/B.bin",
 "tokens_path": "/tmp/tokens.bin",
 "logits_out_path": "/tmp/logits.bin"}
```

The `Job` struct grows a `kind: JobKind` field (defaults to `Training`) so existing jobs keep parsing. New variants land as new arms — no existing call sites change.

## Storage format for state snapshots (binary)

Mirroring the BTCH batch format style:

```
[magic: u32 = 0x53544154]   ('STAT')
[version: u32 = 1]
[n_snapshots: u32]
[reserved: u32]
for each snapshot:
  [layer: u32] [timestep: u32]
  [n_heads: u32] [headdim: u32] [d_state: u32]
  [data: f32 * n_heads * headdim * d_state]
```

Python helpers `state_writer.py` / `state_reader.py` mirror what we already have for batches. No new dependencies.

## What to NOT design now

- Cross-shape composition (different `d_state` between A and B). Needs a learned projection; design when we have a use case.
- Mid-sequence state injection (override state at timestep T mid-forward). Possible but needs scan kernel changes; defer.
- GPU-resident state passing between two PtxModels in the same process. The simplest first cut roundtrips through host memory — slow but correct. Optimize later if profiling says it matters.
- Backward through composition (gradients flowing from B's output into A's parameters). Inference-time composition first; trainable composition is a separate question.

## Migration note

This doc lands BEFORE the implementation so the existing surface is left untouched. The point is: when we wire `forward_with_states` and `forward_from_state`, they're additive, not refactors. The training path, the slot scheduler, the BatchReader — none of them change.
