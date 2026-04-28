"""light_sh_step_function — per-cell light-propagation rule, SH-native.

Replaces the 6-axis-aligned-direction discretization with order-1 spherical
harmonics. Per-cell state goes from 6×3=18 floats to 4×3=12 floats, and
the per-cell rule does proper Lambertian scatter with surface normals.

Per-cell state (per RGB channel):
  c_0   : DC term (constant on the sphere)
  c_x   : linear coefficient along x  (corresponds to Y_1,1)
  c_y   : linear coefficient along y  (corresponds to Y_1,-1)
  c_z   : linear coefficient along z  (corresponds to Y_1,0)

Orthonormal real SH order 1:
  Y_00   = K0     where K0 = 1/(2*sqrt(pi)) ≈ 0.282095
  Y_1,-1 = K1·y   where K1 = sqrt(3/(4*pi)) ≈ 0.488603
  Y_1,0  = K1·z
  Y_1,1  = K1·x

So L_out(d) = c_0·K0 + c_x·K1·d_x + c_y·K1·d_y + c_z·K1·d_z.

Key SH operations:
  • SH of a unit-magnitude delta at direction d: (Y_00, Y_1,-1(d), Y_1,0(d), Y_1,1(d))
    = (K0, K1·d_y, K1·d_z, K1·d_x)
  • Lambertian convolution kernel (Ramamoorthi & Hanrahan 2001):
    A_0 = pi, A_1 = 2*pi/3, A_2 = pi/4 (we use only A_0, A_1).
  • Irradiance at a surface with normal n, given incoming SH:
    E(n) = A_0·c_0·Y_00(n) + A_1·(c_y·Y_1,-1(n) + c_z·Y_1,0(n) + c_x·Y_1,1(n))
         = (sqrt(pi)/2)·c_0 + sqrt(pi/3)·(c_x·n_x + c_y·n_y + c_z·n_z)

Per-material rule:
  EMPTY  → outgoing SH = incoming SH                    (passthrough)
  LIGHT  → outgoing SH = fixed downward-cosine emission (no incoming dependence)
  SOLID  → outgoing SH = albedo · E(n) · hemisphere_SH(n)
           where hemisphere_SH(n) is the SH of a unit hemispherical step
           function oriented along n. Closed-form coefficients:
             c_0      = sqrt(pi)
             c_along_n_axis = sqrt(3*pi)/2 · n_axis
"""
import numpy as np

EMPTY, WHITE, RED, GREEN, LIGHT = 0, 1, 2, 3, 4
N_MATERIALS = 5
MATERIAL_NAMES = ["EMPTY", "WHITE", "RED", "GREEN", "LIGHT"]
N_CHANNELS = 3       # RGB
N_SH = 4             # order-1: c_0, c_x, c_y, c_z

# Constants
K0 = 1.0 / (2.0 * np.sqrt(np.pi))     # Y_00
K1 = np.sqrt(3.0 / (4.0 * np.pi))     # Y_1m peak
SQRT_PI = np.sqrt(np.pi)
SQRT_PI_OVER_3 = np.sqrt(np.pi / 3.0)

ALBEDO_WHITE = np.array([0.92, 0.92, 0.92], dtype=np.float32)
ALBEDO_RED   = np.array([0.92, 0.10, 0.10], dtype=np.float32)
ALBEDO_GREEN = np.array([0.10, 0.92, 0.10], dtype=np.float32)
EMISSION_DOWN = np.array([24.0, 24.0, 24.0], dtype=np.float32)

ALBEDO = {WHITE: ALBEDO_WHITE, RED: ALBEDO_RED, GREEN: ALBEDO_GREEN}


def sh_basis(d: np.ndarray) -> np.ndarray:
    """SH order-1 basis evaluated at unit direction d. Returns (..., 4)."""
    out = np.empty(d.shape[:-1] + (4,), dtype=np.float32)
    out[..., 0] = K0
    out[..., 1] = K1 * d[..., 1]   # Y_1,-1: y component
    out[..., 2] = K1 * d[..., 2]   # Y_1,0:  z component
    out[..., 3] = K1 * d[..., 0]   # Y_1,1:  x component
    return out


def sh_evaluate(coefs: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Evaluate SH at direction d. coefs: (..., 4, C); d: (..., 3). Returns (..., C)."""
    basis = sh_basis(d)                                # (..., 4)
    return (coefs * basis[..., None]).sum(axis=-2)     # (..., C)


def irradiance(coefs: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Lambertian irradiance at a surface with outward normal n.

    SH stores the *traveling direction* of radiance, so light arriving at
    a surface (traveling toward -n at the surface) is captured by the
    cosine kernel oriented at -n:
      E(n) = ∫ L(d) max(0, -d·n) dω
           = (sqrt(pi)/2)·c_0 - sqrt(pi/3)·(c_x·n_x + c_y·n_y + c_z·n_z)

    coefs: (..., 4, C);  n: (..., 3) unit vector. Returns (..., C).
    """
    c0 = coefs[..., 0, :]
    cx = coefs[..., 3, :]
    cy = coefs[..., 1, :]
    cz = coefs[..., 2, :]
    nx, ny, nz = n[..., 0:1], n[..., 1:2], n[..., 2:3]
    return (SQRT_PI / 2.0) * c0 - SQRT_PI_OVER_3 * (cx * nx + cy * ny + cz * nz)


def face_flux(coefs: np.ndarray, face_outward: np.ndarray) -> np.ndarray:
    """Flux through a face with outward normal `face_outward`. Used in LPV
    gather: this is light flowing OUT of the cell through the face.

    F = ∫ L(d) max(0, d·face_outward) dω
      = (sqrt(pi)/2)·c_0 + sqrt(pi/3)·(face_outward · c_vec)

    Note the + sign here vs the - in irradiance() — different cosine
    orientation (light flowing OUT vs IN).
    """
    c0 = coefs[..., 0, :]
    cx = coefs[..., 3, :]
    cy = coefs[..., 1, :]
    cz = coefs[..., 2, :]
    nx = face_outward[..., 0:1]
    ny = face_outward[..., 1:2]
    nz = face_outward[..., 2:3]
    return (SQRT_PI / 2.0) * c0 + SQRT_PI_OVER_3 * (cx * nx + cy * ny + cz * nz)


# Pre-computed SH coefs for a unit hemispherical step function oriented along n.
# Closed form: c_0 = sqrt(pi), c_l1m_along_n = sqrt(3*pi)/2 · n_axis.
# These are the SH of  H_n(d) = 1 if d·n > 0 else 0.
SQRT_3PI_OVER_2 = np.sqrt(3.0 * np.pi) / 2.0


def hemisphere_sh(n: np.ndarray) -> np.ndarray:
    """SH of unit hemispherical step function oriented along n. n: (..., 3); returns (..., 4)."""
    out = np.empty(n.shape[:-1] + (4,), dtype=np.float32)
    out[..., 0] = SQRT_PI
    out[..., 1] = SQRT_3PI_OVER_2 * n[..., 1]
    out[..., 2] = SQRT_3PI_OVER_2 * n[..., 2]
    out[..., 3] = SQRT_3PI_OVER_2 * n[..., 0]
    return out


# Pre-compute the LIGHT emission SH once (cosine-weighted -Y emission).
# A clamped-cosine downward emitter: L(d) = E_max · max(0, -d·Y) = E_max · max(0, -d_y).
# SH coefs (orthonormal):
#   c_0       = ∫ E_max·max(0,-y) · K0 dω = E_max · K0 · pi  = E_max · sqrt(pi)/2
#   c_y(=c_1,-1) = ∫ E_max·max(0,-y) · K1·y dω = -E_max · sqrt(pi/3)  (negative because emission is in -y)
#   c_x = c_z = 0  (by symmetry)
def light_emission_sh(emission: np.ndarray = EMISSION_DOWN) -> np.ndarray:
    """SH coefs for clamped-cosine downward emission. Returns (4, 3)."""
    out = np.zeros((4, N_CHANNELS), dtype=np.float32)
    out[0] = emission * (SQRT_PI / 2.0)
    out[1] = -emission * SQRT_PI_OVER_3      # c_y is negative for -Y emission
    return out


def correct_outgoing_sh(material: int, normal: np.ndarray,
                        incoming_sh: np.ndarray) -> np.ndarray:
    """Symbolic per-cell rule. Returns outgoing SH, shape (4, 3).

    incoming_sh: (4, 3)
    normal: (3,) unit vector (only used for SOLID)
    """
    if material == EMPTY:
        return incoming_sh.copy()
    if material == LIGHT:
        return light_emission_sh()
    # SOLID
    alb = ALBEDO[material]                             # (3,)
    E = irradiance(incoming_sh, normal[None, :])[0]    # (3,)  scalar irradiance per channel
    H = hemisphere_sh(normal[None, :])[0]              # (4,)  hemisphere shape
    # outgoing = albedo * (E / pi) * hemisphere
    # The /pi normalization: Lambertian BRDF is albedo/pi, so output = (albedo/pi)·E·shape
    return (alb[None, :] * (E[None, :] / np.pi)) * H[:, None]


def harvest_random(n_samples: int = 100_000, seed: int = 0,
                   intensity_max: float = 5.0):
    """Generate (material, normal, incoming_SH) → outgoing_SH training tuples.

    For SOLID cells, samples a random axis-aligned normal direction. For
    EMPTY/LIGHT, normal is unused but still generated to keep input shape.
    Incoming SH coefs are sampled to roughly cover the range produced by
    a small-grid Cornell propagation.
    """
    rng = np.random.default_rng(seed)
    mats = rng.integers(0, N_MATERIALS, size=n_samples).astype(np.int64)
    # Random axis-aligned normals (6 possibilities) — for training coverage.
    axis_dirs = np.array([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1],
    ], dtype=np.float32)
    normals = axis_dirs[rng.integers(0, 6, size=n_samples)]
    incs = rng.uniform(-intensity_max, intensity_max,
                       size=(n_samples, N_SH, N_CHANNELS)).astype(np.float32)
    # c_0 (DC) is non-negative for physical radiance; clamp.
    incs[:, 0] = np.abs(incs[:, 0])
    outs = np.zeros((n_samples, N_SH, N_CHANNELS), dtype=np.float32)
    for i in range(n_samples):
        outs[i] = correct_outgoing_sh(int(mats[i]), normals[i], incs[i])
    return mats, normals, incs, outs


if __name__ == "__main__":
    # Sanity prints
    print(f"SH basis sample at (1,0,0): {sh_basis(np.array([1.0, 0, 0], dtype=np.float32))}")
    print(f"SH basis sample at (0,-1,0): {sh_basis(np.array([0, -1.0, 0], dtype=np.float32))}")
    print()
    print(f"LIGHT emission SH (E=24): {light_emission_sh()}")
    print()
    test_n = np.array([0, 1, 0], dtype=np.float32)  # +Y normal (floor)
    test_inc = np.zeros((4, 3), dtype=np.float32)
    test_inc[0] = [10, 10, 10]   # uniform DC
    print(f"Floor (+Y normal) WHITE scattering uniform-10 incoming SH:")
    print(f"  → outgoing SH:")
    print(correct_outgoing_sh(WHITE, test_n, test_inc))
    print()
    test_inc = light_emission_sh()  # what does SOLID do with light's SH as incoming?
    print(f"Floor (+Y normal) WHITE receiving the light's emission SH:")
    print(f"  irradiance E(+Y) = {irradiance(test_inc, test_n[None,:])[0]}")
    print(f"  → outgoing SH:")
    print(correct_outgoing_sh(WHITE, test_n, test_inc))
