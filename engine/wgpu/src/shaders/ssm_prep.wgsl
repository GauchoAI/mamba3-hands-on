// SSM Preprocessing — fully on GPU.
// Takes raw projection output, produces scan-ready tensors.
// Handles: split, B/C norm, DT/decay, RoPE (cumulative phase),
//          outer product, trapezoidal gate, silu precompute.
//
// One workgroup processes ALL timesteps sequentially (for cumsum).
// Parallelism across heads.

struct Params {
    L: u32,
    d_inner: u32,
    d_state: u32,
    n_heads: u32,
    headdim: u32,
    d_in_proj: u32,
    n_angles: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> proj: array<f32>;        // (L, d_in_proj)
@group(0) @binding(1) var<storage, read> dt_bias: array<f32>;     // (H,)
@group(0) @binding(2) var<storage, read> b_norm_w: array<f32>;    // (dS,)
@group(0) @binding(3) var<storage, read> b_norm_b: array<f32>;    // (dS,)
@group(0) @binding(4) var<storage, read> c_norm_w: array<f32>;    // (dS,)
@group(0) @binding(5) var<storage, read> c_norm_b: array<f32>;    // (dS,)
@group(0) @binding(6) var<storage, read_write> inp: array<f32>;   // (L, H, hD, dS)
@group(0) @binding(7) var<storage, read_write> decay_out: array<f32>; // (L, H)
@group(0) @binding(8) var<storage, read_write> Cp_out: array<f32>;    // (L, H, dS)
@group(0) @binding(9) var<storage, read_write> x_skip: array<f32>;    // (L, H, hD)
@group(0) @binding(10) var<storage, read_write> z_silu: array<f32>;   // (L, H, hD)
@group(0) @binding(11) var<uniform> params: Params;

// Workgroup shared: B and C after norm, for all timesteps
var<workgroup> bp_normed: array<f32, 1024>;  // max L*dS = 64*16
var<workgroup> cp_normed: array<f32, 1024>;
var<workgroup> phase: array<f32, 512>;        // max L*n_angles = 64*8
var<workgroup> bx_prev: array<f32, 4096>;     // max H*hD*dS = 8*16*16 = 2048

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // One workgroup per head — process all timesteps sequentially
    let h = gid.x;
    let L = params.L;
    let di = params.d_inner;
    let ds = params.d_state;
    let nh = params.n_heads;
    let hd = params.headdim;
    let dip = params.d_in_proj;
    let na = params.n_angles;

    if (h >= nh) { return; }

    // Only head 0 computes B/C norm and RoPE phase (shared across heads)
    if (h == 0u) {
        // B and C norm for all timesteps
        let bp_off = 2u * di;
        let cp_off = 2u * di + ds;

        for (var t = 0u; t < L; t = t + 1u) {
            let poff = t * dip;

            // Layer norm B
            var mean_b: f32 = 0.0;
            for (var n = 0u; n < ds; n = n + 1u) { mean_b += proj[poff + bp_off + n]; }
            mean_b /= f32(ds);
            var var_b: f32 = 0.0;
            for (var n = 0u; n < ds; n = n + 1u) {
                let d = proj[poff + bp_off + n] - mean_b;
                var_b += d * d;
            }
            var_b /= f32(ds);
            let inv_b = 1.0 / sqrt(var_b + 1e-5);
            for (var n = 0u; n < ds; n = n + 1u) {
                bp_normed[t * ds + n] = (proj[poff + bp_off + n] - mean_b) * inv_b * b_norm_w[n] + b_norm_b[n];
            }

            // Layer norm C
            var mean_c: f32 = 0.0;
            for (var n = 0u; n < ds; n = n + 1u) { mean_c += proj[poff + cp_off + n]; }
            mean_c /= f32(ds);
            var var_c: f32 = 0.0;
            for (var n = 0u; n < ds; n = n + 1u) {
                let d = proj[poff + cp_off + n] - mean_c;
                var_c += d * d;
            }
            var_c /= f32(ds);
            let inv_c = 1.0 / sqrt(var_c + 1e-5);
            for (var n = 0u; n < ds; n = n + 1u) {
                cp_normed[t * ds + n] = (proj[poff + cp_off + n] - mean_c) * inv_c * c_norm_w[n] + c_norm_b[n];
            }
        }

        // RoPE: cumulative phase
        let dt_off = 2u * di + 2u * ds;
        let ang_off = dt_off + 3u * nh;

        for (var k = 0u; k < na; k = k + 1u) {
            var cum: f32 = 0.0;
            for (var t = 0u; t < L; t = t + 1u) {
                // DT_mean = mean across heads of softplus(dd_dt + dt_bias)
                var dt_mean: f32 = 0.0;
                for (var hh = 0u; hh < nh; hh = hh + 1u) {
                    dt_mean += log(1.0 + exp(proj[t * dip + dt_off + hh] + dt_bias[hh]));
                }
                dt_mean /= f32(nh);
                cum += proj[t * dip + ang_off + k] * dt_mean;
                phase[t * na + k] = cum;
            }
        }

        // Apply RoPE to B and C
        for (var t = 0u; t < L; t = t + 1u) {
            for (var k = 0u; k < na; k = k + 1u) {
                let angle = phase[t * na + k];
                let cos_a = cos(angle);
                let sin_a = sin(angle);
                // B
                let be = bp_normed[t * ds + 2u * k];
                let bo = bp_normed[t * ds + 2u * k + 1u];
                bp_normed[t * ds + 2u * k] = be * cos_a - bo * sin_a;
                bp_normed[t * ds + 2u * k + 1u] = be * sin_a + bo * cos_a;
                // C
                let ce = cp_normed[t * ds + 2u * k];
                let co = cp_normed[t * ds + 2u * k + 1u];
                cp_normed[t * ds + 2u * k] = ce * cos_a - co * sin_a;
                cp_normed[t * ds + 2u * k + 1u] = ce * sin_a + co * cos_a;
            }
        }

        // Init bx_prev to zero
        for (var i = 0u; i < nh * hd * ds; i = i + 1u) {
            bx_prev[i] = 0.0;
        }
    }

    workgroupBarrier();

    // Now each head processes its timesteps
    let dt_off = 2u * di + 2u * ds;
    let a_off = dt_off + nh;
    let trap_off = a_off + nh;

    for (var t = 0u; t < L; t = t + 1u) {
        let poff = t * dip;

        // DT, A, decay, trap for this head
        let dt_val = log(1.0 + exp(proj[poff + dt_off + h] + dt_bias[h]));
        let a_raw = -log(1.0 + exp(proj[poff + a_off + h]));
        var a = a_raw;
        if (a > -1e-4) { a = -1e-4; }
        decay_out[t * nh + h] = exp(a * dt_val);
        let trap_val = 1.0 / (1.0 + exp(-proj[poff + trap_off + h]));

        // z_silu
        for (var p = 0u; p < hd; p = p + 1u) {
            let z = proj[poff + h * hd + p];
            z_silu[(t * nh + h) * hd + p] = z * (1.0 / (1.0 + exp(-z)));
        }

        // x_skip
        for (var p = 0u; p < hd; p = p + 1u) {
            x_skip[(t * nh + h) * hd + p] = proj[poff + di + h * hd + p];
        }

        // Cp broadcast
        for (var n = 0u; n < ds; n = n + 1u) {
            Cp_out[(t * nh + h) * ds + n] = cp_normed[t * ds + n];
        }

        // Outer product + trapezoidal: inp = (trap * Bx + (1-trap) * Bx_prev) * dt
        for (var p = 0u; p < hd; p = p + 1u) {
            let x_val = proj[poff + di + h * hd + p];
            for (var n = 0u; n < ds; n = n + 1u) {
                let bx_cur = x_val * bp_normed[t * ds + n];
                let prev = bx_prev[h * hd * ds + p * ds + n];
                let blended = trap_val * bx_cur + (1.0 - trap_val) * prev;
                inp[((t * nh + h) * hd + p) * ds + n] = blended * dt_val;
                bx_prev[h * hd * ds + p * ds + n] = bx_cur;
            }
        }
    }
}
