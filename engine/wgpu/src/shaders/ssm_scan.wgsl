// SSM Scan — WebGPU compute shader
// Explicit fp32 arithmetic, no FMA, sequential accumulation.
// One workgroup per (batch, head) pair.

struct Params {
    B: u32,
    L: u32,
    H: u32,
    hD: u32,
    dS: u32,
}

@group(0) @binding(0) var<storage, read> inp: array<f32>;       // (B, L, H, hD, dS)
@group(0) @binding(1) var<storage, read> decay: array<f32>;     // (B, L, H)
@group(0) @binding(2) var<storage, read> C: array<f32>;         // (B, L, H, dS)
@group(0) @binding(3) var<storage, read> x: array<f32>;         // (B, L, H, hD)
@group(0) @binding(4) var<storage, read> z_silu: array<f32>;    // (B, L, H, hD) precomputed
@group(0) @binding(5) var<storage, read> D_param: array<f32>;   // (H,)
@group(0) @binding(6) var<storage, read_write> y: array<f32>;   // (B, L, H, hD)
@group(0) @binding(7) var<uniform> params: Params;

// State in workgroup shared memory — one state per workgroup
var<workgroup> state: array<f32, 256>;  // hD * dS, max 16*16 = 256

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let B = params.B;
    let L = params.L;
    let H = params.H;
    let hD = params.hD;
    let dS = params.dS;

    if (idx >= B * H) {
        return;
    }

    let b = idx / H;
    let h = idx % H;
    let D_h = D_param[h];

    // Initialize state to zero
    for (var i = 0u; i < hD * dS; i = i + 1u) {
        state[i] = 0.0;
    }

    // Sequential scan over time
    for (var t = 0u; t < L; t = t + 1u) {
        // Load decay for this (b, t, h)
        let dec = decay[b * L * H + t * H + h];

        // State update: state = decay * state + inp
        for (var p = 0u; p < hD; p = p + 1u) {
            for (var n = 0u; n < dS; n = n + 1u) {
                let si = p * dS + n;
                let inp_idx = ((b * L + t) * H + h) * hD * dS + p * dS + n;
                let inp_val = inp[inp_idx];
                // Explicit: no FMA. WGSL f32 ops are IEEE 754 compliant.
                state[si] = dec * state[si] + inp_val;
            }
        }

        // Output projection: y_t[p] = sum_n(state[p,n] * C[n])
        for (var p = 0u; p < hD; p = p + 1u) {
            var sum = 0.0f;
            for (var n = 0u; n < dS; n = n + 1u) {
                let c_idx = ((b * L + t) * H + h) * dS + n;
                let c_val = C[c_idx];
                sum = sum + state[p * dS + n] * c_val;
            }

            // Skip connection
            let x_idx = ((b * L + t) * H + h) * hD + p;
            sum = sum + D_h * x[x_idx];

            // Gate (z_silu precomputed on CPU)
            let gate = z_silu[((b * L + t) * H + h) * hD + p];
            y[((b * L + t) * H + h) * hD + p] = sum * gate;
        }
    }
}
