// Fused Mamba-3 layer — ALL ops in ONE dispatch.
// norm → matmul_in_proj → SSM_prep → SSM_scan → matmul_out_proj → residual
//
// One workgroup per head. Each workgroup processes all timesteps sequentially.
// Eliminates inter-dispatch sync overhead (was 6 dispatches, now 1).
//
// Workgroup shared memory holds per-timestep intermediates.

struct Params {
    L: u32,          // sequence length
    d_model: u32,
    d_inner: u32,
    d_state: u32,
    n_heads: u32,
    headdim: u32,
    d_in_proj: u32,
    n_angles: u32,
}

// Inputs
@group(0) @binding(0) var<storage, read_write> x: array<f32>;        // (L, d_model) — in/out
@group(0) @binding(1) var<storage, read> in_proj_w: array<f32>;       // (d_in_proj, d_model)
@group(0) @binding(2) var<storage, read> out_proj_w: array<f32>;      // (d_model, d_inner)
@group(0) @binding(3) var<storage, read> dt_bias: array<f32>;         // (H,)
@group(0) @binding(4) var<storage, read> d_param: array<f32>;         // (H,)
@group(0) @binding(5) var<storage, read> b_norm_w: array<f32>;        // (dS,)
@group(0) @binding(6) var<storage, read> b_norm_b: array<f32>;        // (dS,)
@group(0) @binding(7) var<storage, read> c_norm_w: array<f32>;        // (dS,)
@group(0) @binding(8) var<storage, read> c_norm_b: array<f32>;        // (dS,)
@group(0) @binding(9) var<storage, read> norm_w: array<f32>;          // (d_model,) layer norm
@group(0) @binding(10) var<storage, read> norm_b: array<f32>;         // (d_model,) layer norm
@group(0) @binding(11) var<uniform> params: Params;

// Scale stored as additional uniform (bitcast from u32)
struct ScaleParam {
    scale_bits: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}
@group(1) @binding(0) var<uniform> scale_param: ScaleParam;

// Per-thread private memory
var<private> x_normed: array<f32, 512>;   // max d_model=512
var<private> proj_row: array<f32, 1280>;  // max d_in_proj=1280
var<private> bp_n: array<f32, 32>;        // d_state after norm
var<private> cp_n: array<f32, 32>;        // d_state after norm
var<private> state: array<f32, 256>;      // hD * dS = 16*16 = 256
var<private> bx_prev: array<f32, 256>;    // hD * dS
var<private> y_head: array<f32, 16>;      // headdim output per timestep

// Single workgroup processes ALL heads — avoids race condition on x output
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Only one workgroup — processes all heads sequentially
    if (gid.x != 0u) { return; }
    let L = params.L;
    let d = params.d_model;
    let di = params.d_inner;
    let ds = params.d_state;
    let nh = params.n_heads;
    let hd = params.headdim;
    let dip = params.d_in_proj;
    let na = params.n_angles;
    let scale = bitcast<f32>(scale_param.scale_bits);

    // Arrays for all heads' states
    var all_state: array<f32, 2048>;   // max H*hD*dS = 8*16*16
    var all_bx_prev: array<f32, 2048>;
    for (var i = 0u; i < nh * hd * ds; i++) { all_state[i] = 0.0; all_bx_prev[i] = 0.0; }

    var cum_phase: array<f32, 16>;
    for (var k = 0u; k < na; k++) { cum_phase[k] = 0.0; }

    // Accumulator for out_proj output per timestep
    var y_full: array<f32, 1024>;  // max d_inner = 512

    for (var t = 0u; t < L; t++) {
        // 1. Layer norm of x[t]
        var mean: f32 = 0.0;
        for (var i = 0u; i < d; i++) { mean += x[t * d + i]; }
        mean /= f32(d);
        var variance: f32 = 0.0;
        for (var i = 0u; i < d; i++) {
            let diff = x[t * d + i] - mean;
            variance += diff * diff;
        }
        variance /= f32(d);
        let inv_std = 1.0 / sqrt(variance + 1e-5);
        for (var i = 0u; i < d; i++) {
            x_normed[i] = (x[t * d + i] - mean) * inv_std * norm_w[i] + norm_b[i];
        }

        // 2. In-projection: proj_row = x_normed @ in_proj_w^T
        for (var j = 0u; j < dip; j++) {
            var s: f32 = 0.0;
            for (var i = 0u; i < d; i++) {
                s += x_normed[i] * in_proj_w[j * d + i];
            }
            proj_row[j] = s;
        }

        // 3. Split + SSM prep — process ALL heads
        let z_off = 0u;
        let x_off = di;
        let bp_off = 2u * di;
        let cp_off = 2u * di + ds;
        let dt_off = 2u * di + 2u * ds;
        let a_off = dt_off + nh;
        let trap_off = a_off + nh;
        let ang_off = trap_off + nh;

        // B norm
        var mean_b: f32 = 0.0;
        for (var n = 0u; n < ds; n++) { mean_b += proj_row[bp_off + n]; }
        mean_b /= f32(ds);
        var var_b: f32 = 0.0;
        for (var n = 0u; n < ds; n++) {
            let diff = proj_row[bp_off + n] - mean_b;
            var_b += diff * diff;
        }
        var_b /= f32(ds);
        let inv_b = 1.0 / sqrt(var_b + 1e-5);
        for (var n = 0u; n < ds; n++) {
            bp_n[n] = (proj_row[bp_off + n] - mean_b) * inv_b * b_norm_w[n] + b_norm_b[n];
        }

        // C norm
        var mean_c: f32 = 0.0;
        for (var n = 0u; n < ds; n++) { mean_c += proj_row[cp_off + n]; }
        mean_c /= f32(ds);
        var var_c: f32 = 0.0;
        for (var n = 0u; n < ds; n++) {
            let diff = proj_row[cp_off + n] - mean_c;
            var_c += diff * diff;
        }
        var_c /= f32(ds);
        let inv_c = 1.0 / sqrt(var_c + 1e-5);
        for (var n = 0u; n < ds; n++) {
            cp_n[n] = (proj_row[cp_off + n] - mean_c) * inv_c * c_norm_w[n] + c_norm_b[n];
        }

        // RoPE: cumulative phase (shared across heads, compute once per timestep)
        var dt_mean: f32 = 0.0;
        for (var hh = 0u; hh < nh; hh++) {
            dt_mean += log(1.0 + exp(proj_row[dt_off + hh] + dt_bias[hh]));
        }
        dt_mean /= f32(nh);
        for (var k = 0u; k < na; k++) {
            cum_phase[k] += proj_row[ang_off + k] * dt_mean;
        }

        // Apply RoPE to B and C (copies that get rotated)
        var bp_rot: array<f32, 32>;
        var cp_rot: array<f32, 32>;
        for (var n = 0u; n < ds; n++) { bp_rot[n] = bp_n[n]; cp_rot[n] = cp_n[n]; }
        for (var k = 0u; k < na; k++) {
            let angle = cum_phase[k];
            let cos_a = cos(angle); let sin_a = sin(angle);
            let be = bp_rot[2u*k]; let bo = bp_rot[2u*k+1u];
            bp_rot[2u*k] = be*cos_a - bo*sin_a; bp_rot[2u*k+1u] = be*sin_a + bo*cos_a;
            let ce = cp_rot[2u*k]; let co = cp_rot[2u*k+1u];
            cp_rot[2u*k] = ce*cos_a - co*sin_a; cp_rot[2u*k+1u] = ce*sin_a + co*cos_a;
        }

        // Process ALL heads — accumulate y into y_full
        for (var h = 0u; h < nh; h++) {
            let dt_val = log(1.0 + exp(proj_row[dt_off + h] + dt_bias[h]));
            let a_raw = -log(1.0 + exp(proj_row[a_off + h]));
            var a_val = a_raw;
            if (a_val > -1e-4) { a_val = -1e-4; }
            let decay_val = exp(a_val * dt_val);
            let trap_val = 1.0 / (1.0 + exp(-proj_row[trap_off + h]));
            let h_off = h * hd * ds;

            for (var p = 0u; p < hd; p++) {
                let x_val = proj_row[x_off + h * hd + p];
                for (var n = 0u; n < ds; n++) {
                    let bx_cur = x_val * bp_rot[n];
                    let prev = all_bx_prev[h_off + p * ds + n];
                    let inp_val = (trap_val * bx_cur + (1.0 - trap_val) * prev) * dt_val;
                    all_state[h_off + p * ds + n] = decay_val * all_state[h_off + p * ds + n] + inp_val;
                    all_bx_prev[h_off + p * ds + n] = bx_cur;
                }
                var s: f32 = 0.0;
                for (var n = 0u; n < ds; n++) {
                    s += all_state[h_off + p * ds + n] * cp_rot[n];
                }
                s += d_param[h] * x_val;
                let z = proj_row[z_off + h * hd + p];
                y_full[h * hd + p] = s * (z / (1.0 + exp(-z)));
            }
        }

        // 4. Out-projection + residual: x[t] += scale * y_full @ out_proj_w^T
        for (var j = 0u; j < d; j++) {
            var s: f32 = 0.0;
            for (var i = 0u; i < di; i++) {
                s += y_full[i] * out_proj_w[j * di + i];
            }
            x[t * d + j] += scale * s;
        }
    }
}
