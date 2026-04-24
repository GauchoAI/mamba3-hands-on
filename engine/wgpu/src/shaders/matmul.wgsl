// Tiled Matrix multiply: C = A × B^T
// A is (M, K), B is (N, K), C is (M, N)
// Uses shared memory tiles for coalesced memory access.

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

var<workgroup> tileA: array<array<f32, TILE>, TILE>;
var<workgroup> tileB: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x * TILE + lid.x;
    let col = wid.y * TILE + lid.y;
    let lr = lid.x;
    let lc = lid.y;

    var sum: f32 = 0.0;

    let n_tiles = (params.K + TILE - 1) / TILE;

    for (var t = 0u; t < n_tiles; t = t + 1u) {
        // Load tile of A into shared memory
        let a_col = t * TILE + lc;
        if (row < params.M && a_col < params.K) {
            tileA[lr][lc] = A[row * params.K + a_col];
        } else {
            tileA[lr][lc] = 0.0;
        }

        // Load tile of B^T into shared memory
        // B is (N, K), B^T access: B[col, t*TILE + lr]
        let b_col = t * TILE + lr;
        if (col < params.N && b_col < params.K) {
            tileB[lc][lr] = B[col * params.K + b_col];
        } else {
            tileB[lc][lr] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product from tiles
        for (var kk = 0u; kk < TILE; kk = kk + 1u) {
            sum = sum + tileA[lr][kk] * tileB[lc][kk];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        C[row * params.N + col] = sum;
    }
}
