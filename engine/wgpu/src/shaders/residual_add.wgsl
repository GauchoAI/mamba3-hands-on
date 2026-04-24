// Residual add: x[i] += scale * y[i]

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// scale is packed as params._pad0 reinterpreted as f32
// Actually, let's put it in the params properly

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    // Scale stored as bitcast of _pad0
    let scale = bitcast<f32>(params._pad0);
    x[i] = x[i] + scale * y[i];
}
