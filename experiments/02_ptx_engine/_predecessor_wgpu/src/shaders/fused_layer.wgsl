// Fused Mamba-3 layer — ALL ops in ONE dispatch.
// norm → matmul_in_proj → SSM_prep → SSM_scan → matmul_out_proj
//
// dispatch_workgroups(n_heads, 1, 1) — one workgroup per head, 16 threads each.
// Thread 0 does SSM, all threads do matmul.
// All large arrays in scratch storage buffer (no private memory limits).
//
// Scratch layout:
//   [per_head_0 | per_head_1 | ... | per_head_{nh-1} | out_proj[L * nh * d]]
// Per-head region: [x_normed(d) | proj(dip) | y(hd) | state(hd*ds) | bx_prev(hd*ds) | phase(na)]
// Out-proj region: indexed [t * nh * d + h * d + j] — per-timestep, per-head results
// Residual (x += scale * sum_h(out_proj)) done in separate head_reduce dispatch.

struct Params {
    L: u32, d_model: u32, d_inner: u32, d_state: u32,
    n_heads: u32, headdim: u32, d_in_proj: u32, n_angles: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> in_proj_w: array<f32>;
@group(0) @binding(2) var<storage, read> out_proj_w: array<f32>;
@group(0) @binding(3) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(4) var<storage, read> d_param: array<f32>;
@group(0) @binding(5) var<storage, read> b_norm_w: array<f32>;
@group(0) @binding(6) var<storage, read> b_norm_b: array<f32>;
@group(0) @binding(7) var<storage, read> c_norm_w: array<f32>;
@group(0) @binding(8) var<storage, read> c_norm_b: array<f32>;
@group(0) @binding(9) var<storage, read> norm_w: array<f32>;
@group(0) @binding(10) var<storage, read> norm_b: array<f32>;
@group(0) @binding(11) var<uniform> params: Params;

struct ScaleParam { scale_bits: u32, _p0: u32, _p1: u32, _p2: u32, }
@group(1) @binding(0) var<uniform> scale_param: ScaleParam;
@group(1) @binding(1) var<storage, read_write> scratch: array<f32>;

var<private> bp_rot: array<f32, 32>;
var<private> cp_rot: array<f32, 32>;

@compute @workgroup_size(16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let h = wid.x;  // which head this workgroup handles
    let L = params.L; let d = params.d_model; let di = params.d_inner;
    let ds = params.d_state; let nh = params.n_heads;
    let hd = params.headdim; let dip = params.d_in_proj; let na = params.n_angles;

    if (h >= nh) { return; }

    // Per-head scratch region
    let per_head = d + dip + hd + hd * ds + hd * ds + na;
    let base = h * per_head;
    let S_X = base;
    let S_P = base + d;
    let S_Y = base + d + dip;
    let S_ST = base + d + dip + hd;
    let S_BX = S_ST + hd * ds;
    let S_PH = S_BX + hd * ds;

    // Out-proj results: stored per-timestep so reduce can handle all at once
    // Layout: out_proj_base + t * nh * d + h * d + j
    let out_proj_base = nh * per_head;

    // Init state + bx_prev + phase
    if (tid == 0u) {
        for (var i = 0u; i < hd * ds; i++) { scratch[S_ST + i] = 0.0; scratch[S_BX + i] = 0.0; }
        for (var k = 0u; k < na; k++) { scratch[S_PH + k] = 0.0; }
    }
    workgroupBarrier();

    for (var t = 0u; t < L; t++) {
        // 1. Norm (thread 0)
        if (tid == 0u) {
            var mean: f32 = 0.0;
            for (var i = 0u; i < d; i++) { mean += x[t * d + i]; }
            mean /= f32(d);
            var v: f32 = 0.0;
            for (var i = 0u; i < d; i++) { let df = x[t*d+i] - mean; v += df*df; }
            v /= f32(d);
            let is = 1.0 / sqrt(v + 1e-5);
            for (var i = 0u; i < d; i++) {
                scratch[S_X + i] = (x[t*d+i] - mean) * is * norm_w[i] + norm_b[i];
            }
        }
        workgroupBarrier();

        // 2. In-proj (parallel across 16 threads)
        let cpt = (dip + 15u) / 16u;
        for (var j = tid * cpt; j < min((tid+1u)*cpt, dip); j++) {
            var s: f32 = 0.0;
            for (var i = 0u; i < d; i++) { s += scratch[S_X + i] * in_proj_w[j * d + i]; }
            scratch[S_P + j] = s;
        }
        workgroupBarrier();

        // 3. SSM (thread 0 only)
        if (tid == 0u) {
            let bp_off = 2u*di; let cp_off = bp_off + ds;
            let dt_off = 2u*di + 2u*ds; let a_off = dt_off + nh;
            let trap_off = a_off + nh; let ang_off = trap_off + nh;

            // B/C norm
            var mb: f32 = 0.0; for (var n = 0u; n < ds; n++) { mb += scratch[S_P + bp_off + n]; }
            mb /= f32(ds);
            var vb: f32 = 0.0; for (var n = 0u; n < ds; n++) { let df = scratch[S_P+bp_off+n]-mb; vb += df*df; }
            vb /= f32(ds); let ib = 1.0/sqrt(vb+1e-5);
            for (var n = 0u; n < ds; n++) { bp_rot[n] = (scratch[S_P+bp_off+n]-mb)*ib*b_norm_w[n]+b_norm_b[n]; }

            var mc: f32 = 0.0; for (var n = 0u; n < ds; n++) { mc += scratch[S_P + cp_off + n]; }
            mc /= f32(ds);
            var vc: f32 = 0.0; for (var n = 0u; n < ds; n++) { let df = scratch[S_P+cp_off+n]-mc; vc += df*df; }
            vc /= f32(ds); let ic = 1.0/sqrt(vc+1e-5);
            for (var n = 0u; n < ds; n++) { cp_rot[n] = (scratch[S_P+cp_off+n]-mc)*ic*c_norm_w[n]+c_norm_b[n]; }

            // RoPE phase
            var dtm: f32 = 0.0;
            for (var hh = 0u; hh < nh; hh++) { dtm += log(1.0+exp(scratch[S_P+dt_off+hh]+dt_bias[hh])); }
            dtm /= f32(nh);
            for (var k = 0u; k < na; k++) { scratch[S_PH + k] += scratch[S_P + ang_off + k] * dtm; }
            for (var k = 0u; k < na; k++) {
                let a = scratch[S_PH + k]; let ca = cos(a); let sa = sin(a);
                let be = bp_rot[2u*k]; let bo = bp_rot[2u*k+1u];
                bp_rot[2u*k] = be*ca - bo*sa; bp_rot[2u*k+1u] = be*sa + bo*ca;
                let ce = cp_rot[2u*k]; let co = cp_rot[2u*k+1u];
                cp_rot[2u*k] = ce*ca - co*sa; cp_rot[2u*k+1u] = ce*sa + co*ca;
            }

            // This head's SSM scan
            let dtv = log(1.0+exp(scratch[S_P+dt_off+h]+dt_bias[h]));
            let ar = -log(1.0+exp(scratch[S_P+a_off+h]));
            var av = ar; if (av > -1e-4) { av = -1e-4; }
            let dec = exp(av * dtv);
            let tr = 1.0/(1.0+exp(-scratch[S_P+trap_off+h]));

            for (var p = 0u; p < hd; p++) {
                let xv = scratch[S_P + di + h*hd + p];
                for (var n = 0u; n < ds; n++) {
                    let bx = xv * bp_rot[n];
                    let pv = scratch[S_BX + p*ds + n];
                    scratch[S_ST + p*ds + n] = dec * scratch[S_ST + p*ds + n] + (tr*bx + (1.0-tr)*pv)*dtv;
                    scratch[S_BX + p*ds + n] = bx;
                }
                var s: f32 = 0.0;
                for (var n = 0u; n < ds; n++) { s += scratch[S_ST + p*ds + n] * cp_rot[n]; }
                s += d_param[h] * xv;
                let z = scratch[S_P + h*hd + p];
                scratch[S_Y + p] = s * (z / (1.0 + exp(-z)));
            }
        }
        workgroupBarrier();

        // 4. Out-proj (parallel) — write to per-timestep region
        let dpt = (d + 15u) / 16u;
        for (var j = tid * dpt; j < min((tid+1u)*dpt, d); j++) {
            var s: f32 = 0.0;
            for (var p = 0u; p < hd; p++) {
                s += scratch[S_Y + p] * out_proj_w[j * di + h * hd + p];
            }
            scratch[out_proj_base + t * nh * d + h * d + j] = s;
        }
        workgroupBarrier();
    }
}
