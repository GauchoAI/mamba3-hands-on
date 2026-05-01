// Reduce: x[t*d+j] += scale * sum_h(scratch[out_proj_base + t*nh*d + h*d + j])
// Called AFTER fused layer dispatch completes (all heads finished).
// Per-timestep out_proj results stored in scratch.

struct Params {
    L: u32,
    d: u32,
    nh: u32,
    scale_bits: u32,
    per_head_size: u32,  // per-head scratch stride (to find out_proj_base)
    hd: u32,             // headdim (unused, kept for struct alignment)
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> scratch: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.L * params.d;
    if (idx >= n) { return; }

    let t = idx / params.d;
    let j = idx % params.d;
    let scale = bitcast<f32>(params.scale_bits);

    // Out-proj results start after all per-head regions
    let out_proj_base = params.nh * params.per_head_size;

    var total: f32 = 0.0;
    for (var h = 0u; h < params.nh; h++) {
        total += scratch[out_proj_base + t * params.nh * params.d + h * params.d + j];
    }
    x[idx] += scale * total;
}
