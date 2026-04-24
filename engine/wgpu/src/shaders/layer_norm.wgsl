// Layer norm: x = (x - mean) / sqrt(var + eps) * w + b
// One workgroup per sequence position.

struct Params {
    seq_len: u32,
    d: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;  // (seq_len, d)
@group(0) @binding(1) var<storage, read> w: array<f32>;         // (d,)
@group(0) @binding(2) var<storage, read> b: array<f32>;         // (d,)
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= params.seq_len) { return; }

    let d = params.d;
    let off = t * d;
    let eps: f32 = 1e-5;

    // Mean
    var mean: f32 = 0.0;
    for (var i = 0u; i < d; i = i + 1u) {
        mean = mean + x[off + i];
    }
    mean = mean / f32(d);

    // Variance
    var v: f32 = 0.0;
    for (var i = 0u; i < d; i = i + 1u) {
        let diff = x[off + i] - mean;
        v = v + diff * diff;
    }
    v = v / f32(d);
    let inv_std = 1.0 / sqrt(v + eps);

    // Normalize + scale + shift
    for (var i = 0u; i < d; i = i + 1u) {
        x[off + i] = (x[off + i] - mean) * inv_std * w[i] + b[i];
    }
}
