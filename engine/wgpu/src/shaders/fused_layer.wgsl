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

// Scratch storage buffer — pre-allocated, avoids private/shared memory limits
// Layout: [x_normed(d) | proj_row(dip) | y_full(di) | bp_n(ds) | cp_n(ds) |
//          state(H*hd*ds) | bx_prev(H*hd*ds) | cum_phase(na)]
@group(1) @binding(1) var<storage, read_write> scratch: array<f32>;

// One workgroup with multiple threads for parallel matmul.
// Thread 0 does the SSM scan (sequential), all threads do matmul in parallel.
@compute @workgroup_size(16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    if (gid.x >= 16u) { return; }
    let tid = lid.x; // 0..15
    let L = params.L;
    let d = params.d_model;
    let di = params.d_inner;
    let ds = params.d_state;
    let nh = params.n_heads;
    let hd = params.headdim;
    let dip = params.d_in_proj;
    let na = params.n_angles;
    let scale = bitcast<f32>(scale_param.scale_bits);

    // Scratch buffer layout offsets — all large arrays in storage, not private
    let S_XNORM = 0u;
    let S_PROJ = d;
    let S_YFULL = d + dip;
    let S_BPN = d + dip + di;
    let S_CPN = d + dip + di + ds;
    let S_STATE = d + dip + di + 2u * ds;
    let S_BXPREV = S_STATE + nh * hd * ds;
    let S_PHASE = S_BXPREV + nh * hd * ds;

    // Init state, bx_prev, cum_phase to zero
    if (tid == 0u) {
        for (var i = 0u; i < nh * hd * ds; i++) {
            scratch[S_STATE + i] = 0.0;
            scratch[S_BXPREV + i] = 0.0;
        }
        for (var k = 0u; k < na; k++) { scratch[S_PHASE + k] = 0.0; }
    }
    workgroupBarrier();

    for (var t = 0u; t < L; t++) {
        // 1. Layer norm — thread 0 computes mean/var, all threads normalize
        if (tid == 0u) {
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
                scratch[S_XNORM +i] = (x[t * d + i] - mean) * inv_std * norm_w[i] + norm_b[i];
            }
        }
        workgroupBarrier();

        // 2. In-projection — PARALLEL: each thread handles dip/16 columns
        let cols_per_thread = (dip + 15u) / 16u;
        let j_start = tid * cols_per_thread;
        let j_end = min(j_start + cols_per_thread, dip);
        for (var j = j_start; j < j_end; j++) {
            var s: f32 = 0.0;
            for (var i = 0u; i < d; i++) {
                s += scratch[S_XNORM +i] * in_proj_w[j * d + i];
            }
            scratch[S_PROJ +j] = s;
        }
        workgroupBarrier();

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
        for (var n = 0u; n < ds; n++) { mean_b += scratch[S_PROJ +bp_off + n]; }
        mean_b /= f32(ds);
        var var_b: f32 = 0.0;
        for (var n = 0u; n < ds; n++) {
            let diff = scratch[S_PROJ +bp_off + n] - mean_b;
            var_b += diff * diff;
        }
        var_b /= f32(ds);
        let inv_b = 1.0 / sqrt(var_b + 1e-5);
        for (var n = 0u; n < ds; n++) {
            scratch[S_BPN +n] = (scratch[S_PROJ +bp_off + n] - mean_b) * inv_b * b_norm_w[n] + b_norm_b[n];
        }

        // C norm
        var mean_c: f32 = 0.0;
        for (var n = 0u; n < ds; n++) { mean_c += scratch[S_PROJ +cp_off + n]; }
        mean_c /= f32(ds);
        var var_c: f32 = 0.0;
        for (var n = 0u; n < ds; n++) {
            let diff = scratch[S_PROJ +cp_off + n] - mean_c;
            var_c += diff * diff;
        }
        var_c /= f32(ds);
        let inv_c = 1.0 / sqrt(var_c + 1e-5);
        for (var n = 0u; n < ds; n++) {
            scratch[S_CPN +n] = (scratch[S_PROJ +cp_off + n] - mean_c) * inv_c * c_norm_w[n] + c_norm_b[n];
        }

        // 3. SSM: thread 0 does all heads (sequential scan)
        if (tid == 0u) {
        // RoPE: cumulative phase (shared across heads, compute once per timestep)
        var dt_mean: f32 = 0.0;
        for (var hh = 0u; hh < nh; hh++) {
            dt_mean += log(1.0 + exp(scratch[S_PROJ +dt_off + hh] + dt_bias[hh]));
        }
        dt_mean /= f32(nh);
        for (var k = 0u; k < na; k++) {
            scratch[S_PHASE +k] += scratch[S_PROJ +ang_off + k] * dt_mean;
        }

        // Apply RoPE to B and C (copies that get rotated)
        var bp_rot: array<f32, 32>;
        var cp_rot: array<f32, 32>;
        for (var n = 0u; n < ds; n++) { bp_rot[n] = scratch[S_BPN +n]; cp_rot[n] = scratch[S_CPN +n]; }
        for (var k = 0u; k < na; k++) {
            let angle = scratch[S_PHASE +k];
            let cos_a = cos(angle); let sin_a = sin(angle);
            let be = bp_rot[2u*k]; let bo = bp_rot[2u*k+1u];
            bp_rot[2u*k] = be*cos_a - bo*sin_a; bp_rot[2u*k+1u] = be*sin_a + bo*cos_a;
            let ce = cp_rot[2u*k]; let co = cp_rot[2u*k+1u];
            cp_rot[2u*k] = ce*cos_a - co*sin_a; cp_rot[2u*k+1u] = ce*sin_a + co*cos_a;
        }

        // Process ALL heads — accumulate y into y_full
        for (var h = 0u; h < nh; h++) {
            let dt_val = log(1.0 + exp(scratch[S_PROJ +dt_off + h] + dt_bias[h]));
            let a_raw = -log(1.0 + exp(scratch[S_PROJ +a_off + h]));
            var a_val = a_raw;
            if (a_val > -1e-4) { a_val = -1e-4; }
            let decay_val = exp(a_val * dt_val);
            let trap_val = 1.0 / (1.0 + exp(-scratch[S_PROJ +trap_off + h]));
            let h_off = h * hd * ds;

            for (var p = 0u; p < hd; p++) {
                let x_val = scratch[S_PROJ +x_off + h * hd + p];
                for (var n = 0u; n < ds; n++) {
                    let bx_cur = x_val * bp_rot[n];
                    let prev = scratch[S_BXPREV +h_off + p * ds + n];
                    let inp_val = (trap_val * bx_cur + (1.0 - trap_val) * prev) * dt_val;
                    scratch[S_STATE +h_off + p * ds + n] = decay_val * scratch[S_STATE +h_off + p * ds + n] + inp_val;
                    scratch[S_BXPREV +h_off + p * ds + n] = bx_cur;
                }
                var s: f32 = 0.0;
                for (var n = 0u; n < ds; n++) {
                    s += scratch[S_STATE +h_off + p * ds + n] * cp_rot[n];
                }
                s += d_param[h] * x_val;
                let z = scratch[S_PROJ +z_off + h * hd + p];
                scratch[S_YFULL +h * hd + p] = s * (z / (1.0 + exp(-z)));
            }
        }

        // Copy y_full to shared memory for parallel out_proj
            for (var i = 0u; i < di; i++) { scratch[S_YFULL +i] = scratch[S_YFULL +i]; }
        } // end tid==0 SSM block
        workgroupBarrier();

        // 4. Out-projection + residual — PARALLEL
        let d_per_thread = (d + 15u) / 16u;
        let jj_start = tid * d_per_thread;
        let jj_end = min(jj_start + d_per_thread, d);
        for (var j = jj_start; j < jj_end; j++) {
            var s: f32 = 0.0;
            for (var i = 0u; i < di; i++) {
                s += scratch[S_YFULL +i] * out_proj_w[j * di + i];
            }
            x[t * d + j] += scale * s;
        }
        workgroupBarrier();
    }
}
