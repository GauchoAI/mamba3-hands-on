"""light_step_function — per-cell light-propagation rule (the Lego).

The teachable adaptation of path tracing: discretize space into a 3D grid
and direction into 6 axis-aligned bins (±X, ±Y, ±Z). Each cell holds:
  - material   ∈ {EMPTY, WHITE, RED, GREEN, LIGHT}   (5 categories)
  - incoming light: 6 directions × 3 RGB channels    (18 floats)

The per-cell rule (one bounce):
  EMPTY → outgoing[d] = incoming[d]                 (passthrough)
  LIGHT → outgoing[d] = EMISSION (constant)         (emit equally)
  WHITE/RED/GREEN → outgoing[d] = albedo · mean(incoming over dirs)
                                                    (Lambertian-flavored)

The MLP learns this rule. The orchestrator wires the cells together via
neighbor-passing — same shape as Conway/WireWorld, but with continuous
RGB-per-direction state instead of {0, 1}.

Output is regression (continuous), not classification — so loss is MSE.
The orchestrator clamps outgoing to be non-negative to prevent leaks.
"""
import numpy as np

EMPTY, WHITE, RED, GREEN, LIGHT = 0, 1, 2, 3, 4
N_MATERIALS = 5
MATERIAL_NAMES = ["EMPTY", "WHITE", "RED", "GREEN", "LIGHT"]

# 6 directions: +X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5
N_DIRS = 6
DIR_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
# delta[d] = (dx, dy, dz) for direction d (the way light moves)
DIR_DELTA = [
    ( 1,  0,  0),  # +X
    (-1,  0,  0),  # -X
    ( 0,  1,  0),  # +Y
    ( 0, -1,  0),  # -Y
    ( 0,  0,  1),  # +Z
    ( 0,  0, -1),  # -Z
]
N_CHANNELS = 3  # RGB

ALBEDO_WHITE = np.array([0.92, 0.92, 0.92], dtype=np.float32)
ALBEDO_RED   = np.array([0.92, 0.10, 0.10], dtype=np.float32)
ALBEDO_GREEN = np.array([0.10, 0.92, 0.10], dtype=np.float32)
# LIGHT emits only in -Y direction (downward, like a real ceiling lamp).
# Total flux is N_DIRS × per-dir-emission of the old all-dirs model = 6×4=24.
EMISSION_DOWN = np.array([24.0, 24.0, 24.0], dtype=np.float32)
# Keep EMISSION around for back-compat / training-data range.
EMISSION = EMISSION_DOWN

ALBEDO = {WHITE: ALBEDO_WHITE, RED: ALBEDO_RED, GREEN: ALBEDO_GREEN}


def correct_outgoing(material: int, incoming: np.ndarray) -> np.ndarray:
    """incoming: (N_DIRS, 3). Returns (N_DIRS, 3) outgoing.

    LIGHT emits only in -Y direction (downward, like a ceiling lamp).
    """
    if material == EMPTY:
        return incoming.copy()
    if material == LIGHT:
        out = np.zeros((N_DIRS, N_CHANNELS), dtype=np.float32)
        # NY index = 3 (see DIR order at top of file)
        out[3] = EMISSION_DOWN
        return out
    alb = ALBEDO[material]
    mean_in = incoming.mean(axis=0)        # (3,)
    return np.tile(alb * mean_in, (N_DIRS, 1)).astype(np.float32)


def harvest_random(n_samples: int = 50_000, seed: int = 0,
                   intensity_max: float = 5.0):
    """Generate (material, incoming, outgoing) training tuples.

    Coverage:
      - All 5 materials roughly equally represented.
      - incoming uniformly random in [0, intensity_max] per channel/dir.
      - This range covers what a Cornell-scale propagation produces
        (light cells emit 4.0; cells near the light see incoming ≤ 4).
    """
    rng = np.random.default_rng(seed)
    mats = rng.integers(0, N_MATERIALS, size=n_samples).astype(np.int64)
    incs = rng.uniform(0.0, intensity_max,
                       size=(n_samples, N_DIRS, N_CHANNELS)).astype(np.float32)
    outs = np.zeros((n_samples, N_DIRS, N_CHANNELS), dtype=np.float32)
    for i in range(n_samples):
        outs[i] = correct_outgoing(int(mats[i]), incs[i])
    return mats, incs, outs


if __name__ == "__main__":
    # Sanity: print rule for each material on a sample input
    sample_inc = np.zeros((N_DIRS, N_CHANNELS), dtype=np.float32)
    sample_inc[0] = [1.0, 0.5, 0.2]  # +X has some light
    sample_inc[4] = [0.3, 0.3, 0.3]  # +Z has some light
    print("Sample incoming:")
    for d, name in enumerate(DIR_NAMES):
        print(f"  {name}: {sample_inc[d]}")
    print()
    for mat in range(N_MATERIALS):
        out = correct_outgoing(mat, sample_inc)
        print(f"{MATERIAL_NAMES[mat]:>6}: outgoing[+X]={out[0]} (all dirs same for solid/light)")
