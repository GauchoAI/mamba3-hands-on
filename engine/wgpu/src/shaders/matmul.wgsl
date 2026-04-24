// Matrix multiply: C = A × B^T
// A is (M, K), B is (N, K), C is (M, N)
// Each workgroup computes one tile of C.

struct Params {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16;

@compute @workgroup_size(TILE, TILE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;

    if (row >= params.M || col >= params.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k = 0u; k < params.K; k = k + 1u) {
        sum = sum + A[row * params.K + k] * B[col * params.K + k];
    }
    C[row * params.N + col] = sum;
}
