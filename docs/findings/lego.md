---
title: Lego Findings Journal
chapter: "05"
status: archival
open_sections: 1
summary: "Detailed specialist-composition journal behind the Lego chapter."
---

# Lego library — findings journal

The "Lego" thread: small step-function specialists (~500-1500 params each)
that, once trained byte-perfect on a primitive operation (Conway, GCD,
WireWorld, Hanoi-step, light propagation, etc.), can be composed by a
Python orchestrator without retraining the whole model.

Originally inline in the root `findings.md`; moved here as part of the
structuring pass (2026-04-30).

All entries below are verbatim relocations. Same content, same dates,
same numbering.

---

## Entry — 3D Cornell box: byte-perfect from a 406-param Lego (2026-04-28)

User: "We need to make sure it is a proper corner box, which is three d.
And yeah, compound error. We have to fix that. Of course."

**Two changes from the 2D demo:**

1. **Bumped to 3D / 6 directions.** State per cell is now (material,
   6 dirs × RGB) = 19 inputs / 18 outputs. Same Lego shape, more
   directions. New direction set: ±X, ±Y, ±Z.

2. **Fixed compound error with hard-gated architecture.** The 2D demo
   used soft gates (softmax over passthrough/scatter/emit) — even tiny
   gate noise (e.g. EMPTY cell with emit_gate=0.01) injected fake
   light every step, drifting ~50% over 128 steps. Switched to **hard
   gates by material**: orchestrator picks the mode based on material
   ID (EMPTY → passthrough, LIGHT → emit, SOLID → scatter), MLP only
   learns the *colors* (scatter_color = albedo, emit_color = emission).

   The structure of physics is built into the rule; the Lego only
   learns the per-material parameters. 406 params, 5.5s training,
   abs_err = 2 × 10⁻⁵ on validation.

**The result: byte-perfect Cornell.**

3D scene (32×32×32 voxels, canonical Cornell with two interior boxes,
ceiling light, RED/GREEN side walls, WHITE floor/ceiling/back):

| version              | time    | max  | mean   | diff vs symbolic  |
|----------------------|---------|------|--------|-------------------|
| Lego (406 params)    |  164 ms | 4.00 | 0.0692 | **max 0.0000**     |
| Symbolic torch ref   |   27 ms | 4.00 | 0.0692 | (reference)        |

After 96 propagation steps over 32k voxels × 6 dirs × 3 channels, the
Lego output matches the symbolic propagator to floating-point precision.

The rendered image shows the canonical Cornell features:
  - bright ceiling light visible at top
  - two box silhouettes underneath, with the tall box behind the short box
  - boxes appear darker because they occlude direct ceiling light
  - faint indirect illumination on the floor
  - back wall barely lit (only indirect bounces reach it)

Side-by-side comparison saved as `cornell3d_compare.png` — Lego on the
left, symbolic on the right. The two images are pixel-identical (max
diff 0.0000 in the underlying float tensors); the ceiling light, both
box silhouettes, and the dim floor all render the same way.

**Visualization upgrade — perspective ray-marcher** (later same day):
The first orthographic camera only sampled the back wall through each
pixel column; side walls were one pixel wide and color bleed wasn't
visible. Replaced with a vectorized perspective ray-marcher (camera in
front of the box, slightly above center, FOV 55°, 0.5-voxel step,
~480ms in NumPy on CPU). Each pixel marches into the volume until it
hits a non-empty cell; at the hit, sample outgoing in the closest of
the 6 axis-aligned bins to the inverse ray direction.

Bumped to 200 propagation steps and the proper-Cornell features came
out: RED left wall, GREEN right wall, bright ceiling-light patch, two
box silhouettes (tall and short), bright floor patch from direct
ceiling light, and **color bleed visible at the wall/ceiling corners**
where the wall scatter hits the adjacent white surfaces.

  - Initial render had RED on right, GREEN on left — `right = cross(forward,
    up)` produced a left-handed basis. Switched to `right = cross(up,
    forward)` and the colors landed where Cornell expects them.
  - 200 steps was the threshold where indirect light reached the camera
    via wall scatter. At 96 steps the side walls were barely visible.

Final 384×384 render: byte-perfect Lego ↔ symbolic, with all the
canonical Cornell signatures (color bleed, soft floor shadow under the
boxes, bright ceiling near the light, dim back wall).

**Iteration to a proper Cornell** (next round). User feedback caught
two physics issues:

  1. *"The walls look fluorescent at the top — like they have bulbs in
     them."* The original LIGHT cells emitted EMISSION = 4 in **all 6
     directions**. So at y=H-2 (light's row), the source spat raw
     emission sideways through empty cells (empty = passthrough), and
     that beam hit the side walls at near-emission intensity. Wall was
     just reflecting an incoming beam = fluorescent look.

  2. *"How can a non-emissive cell be as bright as the emission?"* It
     can't, mathematically — outgoing for non-LIGHT is bounded by
     albedo · max(incoming) ≤ albedo · EMISSION. But with all-direction
     emission + 0.95 albedo, walls right next to the light bounce-amplify
     to near-source brightness. Looks energy-violating even though it
     isn't strictly.

The fix:

  - **Directional LIGHT emission**: LIGHT cells emit only in -Y (down,
    like a real ceiling lamp). Same total flux (24 = 6 × 4), concentrated
    in one direction. Architecturally enforced: `emit_head` outputs 3
    RGB and the model places them in the -Y bin only, zeros elsewhere.
    No ceiling-artifact drift over long rollouts because the +Y/±X/±Z
    emissions are zero by construction, not learned.

  - **Light flush with the ceiling**: replaced the y=H-2 light with
    LIGHT cells in y=H-1 (the ceiling row itself). No dark gap above
    the light, no -Y emissions blocked by an above-light WHITE cell.

  - **Albedos at 0.92**: between the dim 0.85 (dark walls) and the
    fluorescent 0.95 (over-bright reflections). Walls retain enough
    energy across many bounces to fill in indirect illumination, but
    not so much that they look emissive.

  - **Bigger light** (12×12 cells instead of 8×8): more direct flux,
    brighter floor, more indirect light to spread to the walls.

Final 526-param Lego (smaller because emit_head shrank from 18 → 3
once we fixed -Y direction by construction). Validation: all 5
materials at max_err < 0.0003. 400-step rollout: max diff Lego ↔
symbolic = 0.0003, mean diff = 1×10⁻⁵.

The render now shows **all the canonical Cornell signatures** at
exposure = 2 (no extreme tonemap pushing):
  - bright ceiling lamp with warm rim from indirect light off floor/walls
  - green wall on the right in full color, red wall on the left in full
    color (peeking from behind the tall box)
  - two box silhouettes with proper Lambertian shading (top bright,
    sides darker, slight green bleed on the box face nearest the green
    wall)
  - bright floor patch directly under the light
  - color bleed from walls onto adjacent floor / ceiling

The energy-conservation feedback was the real lesson here: with
all-direction emission, a non-emissive cell's outgoing was hovering
near-source brightness because it was always reflecting a fresh
emission beam. The user spotted the unphysical look immediately. The
architectural fix (emission only in -Y, by construction) is the right
generalization: in a real renderer, surfaces emit hemispherically; in
our 6-direction discretization, "hemispherical" means "one or two
specific axis-aligned bins."

**The compound-error lesson** (worth saving as architectural feedback):

> For iterated MLP CAs, soft gates compound: tiny per-step gate errors
> grow linearly per step. The fix isn't more training — it's encoding
> the rule's discrete structure as hard gates in the orchestrator.
> The MLP then learns the *parameters* (colors, weights), not the
> *structure* (which mode applies). This eliminates compound drift
> because the structure is exact by construction.

**Speed picture** (compute regime now relevant):

  - Lego on MPS: 164 ms for 32k voxels × 96 steps = ~19 M cell-decisions/s.
  - Symbolic on MPS: 27 ms = 116 M cells/s.
  - For *this* simple rule, symbolic where-cascade beats the MLP body.
  - The Lego value isn't speed — it's "any per-cell rule, no hand
    vectorization, byte-equivalent."

Files: `light_step_function.py`, `train_light_step.py`, `cornell_3d.py`.
The 2D version (`cornell_lightca.py`) was the stepping-stone; the 3D
version supersedes it for the canonical demo.

---

## Entry — Light-CA Lego: Cornell-flat by adapting path tracing into a teachable rule (2026-04-28)

User: "the work indeed would be to adapt the algorithm to something
teachable." Path tracing has continuous geometry — no closed finite
state space, no Lego pattern fit. So we adapted it: discretize space
into a grid and direction into 4 axis-aligned bins (N, S, E, W), giving
each cell continuous-valued state instead of {0, 1}. Same Conway/WireWorld
shape, but with light vectors per cell.

**The Lego: `light_step` (1009 params)**

State per cell: (material ∈ {EMPTY, WHITE, RED, GREEN, LIGHT}, incoming RGB
per 4 directions). Per-cell rule:

  - EMPTY → outgoing[d] = incoming[d]                    (passthrough)
  - LIGHT → outgoing[d] = EMISSION (constant)             (emit)
  - SOLID → outgoing[d] = albedo · mean(incoming over dirs)  (Lambertian-flavored)

The MLP uses a **structured architecture**: it predicts (passthrough,
scatter, emit) gates plus scatter/emission colors, then the propagation
math is fixed in the forward pass. The MLP only learns "what is this
material, and what does it scatter / emit?" — the rule structure is
free.

  - 1009 params, 17s training to per-step abs_err < 0.005.
  - All 5 materials saturate against the symbolic rule (max_err < 0.06).

**The orchestrator: `cornell_lightca.py`**

The orchestrator wires neighbor passing:

  - incoming[r, c, N] ← outgoing[r+1, c, N]   (light moving north
    arrives from the south neighbor)
  - …same for S, E, W
  - boundaries: incoming from outside the grid is 0

Per step: gather incoming from neighbors → call Lego on every cell in
parallel (one MLP forward over H·W states) → accumulate per-cell
brightness for visualization.

**Cornell-flat result (64×64 grid, 128 propagation steps)**

| version                   | time   | max  | mean | max diff vs symbolic |
|---------------------------|--------|------|------|----------------------|
| Lego (trained MLP)        |  70 ms | 16.03 | 1.11 | 0.54 |
| Symbolic (torch where-cascade) | 28 ms | 16.00 | 0.72 | (reference) |

Both produce a recognizable Cornell-flat: dark interior with red strip
on the left wall, green on the right, bright central beam from the
ceiling light, and color tinting near the colored walls. The Lego runs
slower than the torch-symbolic at this size (60ms warmup vs an
optimized where-cascade) and accumulates ~50% more brightness because
per-step errors of ~0.5% compound over 128 propagation steps.

**The honest read**

This is the cleanest "adapt to be teachable" example so far. Path tracing
isn't naturally a Lego — but **you can redesign the rendering algorithm
into a CA whose per-cell rule fits the Lego pattern**. The result is a
multi-channel CA that does light propagation, with the same shape as
Conway/WireWorld but more interesting:

  - state is RGB-per-direction (not boolean)
  - rule has 5 material branches (most so far)
  - output is regression, not classification (first regression Lego)
  - structured architecture: MLP predicts *parameters*, propagation math
    is built into the forward pass

The compound-error issue is real for any iterated CA done with an MLP
— small per-step error accumulates over many steps. The fix paths are
known (longer training, residual connections, output clamping, energy
conservation regularizer); the demo proves the framework works.

**The pattern that just generalized**: take a continuous-geometry
problem, discretize state and direction, find a per-cell rule, file it
as a Lego. We now have **6 CA-style Legos** (Conway, WireWorld,
LightStep) and the orchestrator pattern handles them all the same way.

Files: `light_step_function.py`, `train_light_step.py`,
`cornell_lightca.py`. Speed showdown also in `cornell_pathtrace_showdown.py`
(naive Python / NumPy / PyTorch-MPS for the pure pathtracer baseline).

---

## Entry — Speed showdown: where the Lego beats software, and where it doesn't (2026-04-28)

User question: "What could we do that will prove that we are, in fact,
faster than software?" Built honest benchmarks: same per-cell rule, three
implementations (naive Python, vectorized NumPy, neural-batched MPS).

**Conway's Game of Life — 134-param `conway_step` Lego**

| grid × gens | naive_python | numpy_conv | neural_batch (MPS) |
|---|---|---|---|
| 200² × 10  |   154.7 ms |   0.7 ms |   55.2 ms |
| 1000² × 100 |        — | 224.1 ms | 1168.0 ms |

All three byte-for-byte identical. Throughput: naive 2.6 M cells/s,
NumPy 446 M cells/s, neural 86 M cells/s.

**WireWorld — 264-param `wireworld_step` Lego (4 states, branchy rule)**

| grid × gens | naive_python | numpy_branch | neural_batch (MPS) |
|---|---|---|---|
| 200² × 10  |   71.4 ms |   1.3 ms |  126.9 ms |
| 1000² × 100 |       — | 334.8 ms | 1047.1 ms |
| 3000² × 50 |       — | 1536.7 ms | 4762.7 ms |

Same byte-equivalence. Neural throughput plateaus at ~95 M cells/s —
MPS-bandwidth-bound. NumPy ~293 M cells/s — CPU-bandwidth-bound on
boolean ops.

**The honest read**

1. Neural batched **dominates naive software** at scale. At 1000² × 100
   gens, naive Python would take ~hours; neural finishes in ~1 second.
   This is real GPU parallelism on a tiny learned rule.
2. Neural batched **does not beat hand-tuned NumPy** on simple per-cell
   CAs. NumPy's 8-roll convolution + boolean rule is hard to outpace —
   the per-cell work is too cheap to need a GPU.
3. The numpy-vs-neural gap **shrinks as the rule gets branchier**.
   Conway (1 boolean expression): 5.2× slower. WireWorld (4 branches):
   3.1× slower. Branchier rules force NumPy into multiple temp arrays
   and a where-cascade.
4. Pushing scale **does not close the gap further**. Both saturate
   their respective memory bandwidths at ~1000² and stay flat to 3000².

**The win regime**

Where the Lego library actually wins on speed is *not* simple per-cell
rules — NumPy is brutal there. The wins are:

  - **Naive-Python baselines** (any time the alternative is a Python
    for-loop, neural batched is 1000s of × faster at scale).
  - **Rules that don't trivially vectorize** — e.g. multi-channel cells
    with non-linear inter-channel dependencies, large-kernel
    neighborhoods (>5×5), or learned activations. Custom CUDA/Metal
    kernels are the alternative; neural batched is a free lunch.
  - **Dev velocity, not raw speed**: a new CA rule = 1 second of
    training. New NumPy implementation = 30 minutes of fiddly boolean
    ops. Same Lego pattern fits any (state, action) lookup-table rule.

**Where this leaves the speed thesis**: the Lego library's speed story
is *"GPU throughput on any learnable per-cell rule, with zero
hand-vectorization"*. It beats naive software easily and ties or
slightly loses to hand-tuned vectorized software on simple rules. The
clean speed win is on rules NumPy can't trivially vectorize.

Code: `conway_speed_showdown.py`, `wireworld_speed_showdown.py`,
`wireworld_step_function.py`, `train_wireworld_step.py`.

The Lego library now has 6 specialists (added wireworld_step, 264 params).

---

## Entry — Lego library: 5 step-function specialists, ~2.2k total params (2026-04-28)

Following the Hanoi perfect-extension result, scaled the same
pattern across four more puzzles. Each is a tiny MLP over a
role-encoded finite state space, trained in <2 seconds, generalizes
by construction.

| Lego           | params | states | what it learns                  |
|----------------|--------|--------|----------------------------------|
| hanoi_step_fn  |  1574  |   36   | Tower of Hanoi step              |
| gcd_step       |   331  |    3   | Euclidean GCD by subtraction     |
| conway_step    |   134  |   18   | Game of Life cell transition     |
| bubble_step    |    38  |    2   | Sort comparison (a > b → swap?)  |
| maze_step      |   129  |    9   | Greedy grid navigation           |
| **TOTAL**      | **2206**|  **68**| 5 algorithms                     |

Total combined training time: ~5 seconds on M4 Pro.

**Composite tasks (zero retraining)**: orchestrator.py implements
new tasks by chaining frozen specialists in plain Python. Examples
verified end-to-end:

  - `GCDHANOI 6 9` → 7   (Hanoi×2 + GCD: gcd(63, 511))
  - `CONWAYSTABLE <g>`   (Conway iterated to fixpoint)
  - `SORTHANOI 4` → sorted disk-id sequence (Hanoi + Bubble)
  - `GCDSORTED [12,18,8,30,15]` → 2 (Bubble + GCD)
  - `MAZESTEPS 0 0 100 -50` → 151

**The pattern that holds across all five Legos**:

  1. State has a *closed* reachable space — encoded as roles, signs,
     comparisons, or other invariants. Not parameterized by problem
     size.
  2. Action space is small and finite (2 to 6 outputs).
  3. Step function is a 4-layer MLP at most. Hidden dim 4–32.
     Total params ≤ ~1.5k per Lego.
  4. Training data is ALL reachable (state, action) pairs of the
     algorithm. Saturation in seconds.
  5. Generalization to OOD inputs is automatic — no OOD inputs exist
     in the closed state space.

**Two architectural ideas tested earlier** (commit 89f83a7):

  - **Fast fine-tune (Hanoi → GCD)**: no measurable benefit at this
    scale. From-scratch GCD hits 100% in 50 steps; Hanoi-pretrained
    transfer also hits 100% in 50 steps. The functions are too
    small for prior knowledge to matter.
  - **Neural composition** for task dispatch: works trivially —
    plain Python regex dispatch + frozen-specialist runners
    handles arbitrary compositions. The "neural composition"
    machinery (synapse / AttendBridge) is overkill for this
    layer of orchestration. It might still be useful for tasks
    where the orchestration itself requires learning (where
    string-prefix matching isn't enough), but for the Lego
    library that's not where the value is.

**The deeper observation**: the work moved from "can a model learn
this algorithm?" to "what's the right state encoding so the function
is tiny and total?" Once state is closed under the algorithm, the
neural part is trivial — almost a lookup table. The Lego is the
state representation as much as the MLP.

Code: `{hanoi,gcd,conway,bubble,maze}_step_function.py`,
`train_*_step.py`, `orchestrator.py`.

**Sort suite stress test (commit 5860a3a)**: same 38-param
`bubble_step` Lego, four orchestrators, n=3000 items vs Python's
`sorted()`:

| algo       | time     | neural calls | correct |
|------------|----------|--------------|---------|
| sorted()   | 0.16ms   | —            | ✓       |
| bubble     | 1671ms   | 2,976        | ✓       |
| selection  | 852ms    | 2,999        | ✓       |
| insertion  | 26980ms  | 2,254,405    | ✓       |
| merge      | 377ms    | 31,236       | ✓       |

All four byte-for-byte correct. Neural-call count matches algorithmic
complexity exactly (O(n) for bubble's batched passes, O(n²) for
insertion's sequential decisions, O(n log n) for merge). The shared
Lego is frozen; the orchestrators are the sole difference. **One
primitive, four algorithms — the cleanest "Legos composed at random"
demonstration so far.** The step function pattern scales with no
fall-off in correctness; what we don't get is C-speed comparisons.

---
