// Raw GPU compute benchmark — multiply-add on large buffer.
// Each thread: out[i] = a[i] * b[i] + c[i]  (1 mul + 1 add = 2 FLOP per element)

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Params { n: u32, iters: u32, _p0: u32, _p1: u32, }
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 256u;
    if (i >= params.n) { return; }

    var v = c[i];
    for (var it = 0u; it < params.iters; it++) {
        v = a[i] * v + b[i];  // 2 FLOP per iteration
    }
    c[i] = v;
}
