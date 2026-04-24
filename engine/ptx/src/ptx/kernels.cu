// Mamba-3 kernels in CUDA C — compiled to PTX at runtime via NVRTC with strict
// FP32 flags (no fast-math, no contraction surprises). All arithmetic uses
// explicit __fmaf_rn (IEEE round-to-nearest FMA) to match Rust f32::mul_add.
//
// Generated PTX can be inspected via cudarc's module dump (see runtime.rs).
//
// Design is v1 (correctness-first): one kernel per op, many dispatches per
// layer. v2+ fuses ambitiously.

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ---------- helpers ---------------------------------------------------------

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(x * -1.0f));
}

// Matches Rust (1.0 + x.exp()).ln(). Simple form; overflows at x >> 80, same
// behavior as CPU reference.
__device__ __forceinline__ float softplus_f(float x) {
    return __logf(1.0f + __expf(x));
}

// Warp-wide sum over 32 lanes using butterfly shuffle.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

// ---------- copy_f32 --------------------------------------------------------
// dst[i] = src[i] for i in [0, n).
// Grid: (ceil(n/256),), Block: (256,)
extern "C" __global__ void copy_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ---------- embed_gather ----------------------------------------------------
// x[t, i] = embed[tokens[t], i]  for t in [0,L), i in [0,d)
// Grid: (L,), Block: (32,)
extern "C" __global__ void embed_gather(
    const unsigned int* __restrict__ tokens,
    const float* __restrict__ embed,
    float* __restrict__ x,
    int L, int d, int vocab
) {
    int t = blockIdx.x;
    if (t >= L) return;
    unsigned int tok = tokens[t];
    bool valid = tok < (unsigned int)vocab;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        x[t * d + i] = valid ? embed[tok * d + i] : 0.0f;
    }
}

// ---------- layer_norm ------------------------------------------------------
// Per-row LayerNorm: x[t,:] = ((x[t,:] - mean) * rsqrt(var + eps)) * w + b
// Grid: (L,), Block: (32,)
extern "C" __global__ void layer_norm(
    float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int L, int d
) {
    int t = blockIdx.x;
    if (t >= L) return;
    float* row = &x[t * d];
    const float eps = 1e-5f;
    int tid = threadIdx.x;

    float s = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) s += row[i];
    s = warp_reduce_sum(s);
    float mean = s / (float)d;

    float vs = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float diff = row[i] - mean;
        vs = __fmaf_rn(diff, diff, vs);
    }
    vs = warp_reduce_sum(vs);
    float inv_std = rsqrtf(vs / (float)d + eps);

    for (int i = tid; i < d; i += blockDim.x) {
        row[i] = __fmaf_rn((row[i] - mean) * inv_std, w[i], b[i]);
    }
}

// ---------- matmul_t --------------------------------------------------------
// C[M, N] = A[M, K] @ B[N, K]^T       (i.e. C[i,j] = sum_k A[i,k] * B[j,k])
// Grid: (ceil(N/16), ceil(M/16)), Block: (16, 16)
extern "C" __global__ void matmul_t(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;
    float acc = 0.0f;
    const float* a_row = &A[i * K];
    const float* b_row = &B[j * K];
    for (int k = 0; k < K; k++) {
        acc = __fmaf_rn(a_row[k], b_row[k], acc);
    }
    C[i * N + j] = acc;
}

// ---------- compute_ssm_params_and_dt_mean ---------------------------------
// Fused: per-(t,h) dt/decay/trap AND per-t dt_mean (reduce over heads).
// Replaces compute_ssm_params + compute_dt_mean (2 dispatches -> 1).
// Grid: (L,), Block: (max(H, 32),)  — single warp does dt_mean reduction.
extern "C" __global__ void compute_ssm_params_and_dt_mean(
    const float* __restrict__ proj,
    const float* __restrict__ dt_bias,
    float* __restrict__ dt,
    float* __restrict__ decay,
    float* __restrict__ trap,
    float* __restrict__ dt_mean,
    int L, int H, int dip, int di, int ds
) {
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= L) return;
    int dt_off = 2 * di + 2 * ds;
    int a_off = dt_off + H;
    int tr_off = a_off + H;

    // Compute dt/decay/trap for each head this thread is responsible for.
    // Gather this thread's dt contribution for the reduction.
    float my_dt = 0.0f;
    for (int h = tid; h < H; h += blockDim.x) {
        float dt_raw = proj[t * dip + dt_off + h] + dt_bias[h];
        float dt_v = softplus_f(dt_raw);
        dt[t * H + h] = dt_v;
        my_dt += dt_v;

        float a_raw = proj[t * dip + a_off + h];
        float a = -softplus_f(a_raw);
        if (a > -1e-4f) a = -1e-4f;
        decay[t * H + h] = __expf(a * dt_v);

        float tr_raw = proj[t * dip + tr_off + h];
        trap[t * H + h] = sigmoid_f(tr_raw);
    }
    // Warp reduce within a single warp (assuming blockDim.x <= 32)
    my_dt = warp_reduce_sum(my_dt);
    if (tid == 0) dt_mean[t] = my_dt / (float)H;
}

// ---------- matmul_t_tiled --------------------------------------------------
// Tiled matmul: C[M, N] = A[M, K] @ B[N, K]^T. 16x16 block, 16x16 SMEM tiles.
// Each thread computes one C element; block cooperates on tile loads.
// Reduces HBM traffic by ~K/tile_k (16x for typical K).
// Grid: (ceil(N/16), ceil(M/16)), Block: (16, 16)
extern "C" __global__ void matmul_t_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[16 * 16];
    __shared__ float Bs[16 * 16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * 16 + tx;
    int row = blockIdx.y * 16 + ty;

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        // Load A[row, k_tile + tx] → As[ty, tx]
        int a_k = k_tile + tx;
        As[ty * 16 + tx] = (row < M && a_k < K) ? A[row * K + a_k] : 0.0f;
        // Load B[col, k_tile + ty] → Bs[tx, ty]  (swap so inner loop is strided by 1)
        int b_k = k_tile + ty;
        Bs[tx * 16 + ty] = (col < N && b_k < K) ? B[col * K + b_k] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; k++) {
            acc = __fmaf_rn(As[ty * 16 + k], Bs[tx * 16 + k], acc);
        }
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------- compute_ssm_params ---------------------------------------------
// Per (t, h): dt = softplus(dd_dt + dt_bias); a = max(-softplus(dd_a), -1e-4);
//             decay = exp(a*dt);  trap = sigmoid(trap_raw)
// Offsets into proj: dt_off = 2*di + 2*ds;  a_off = dt_off + H;  trap_off = a_off + H
// Grid: (L,), Block: (H,)
extern "C" __global__ void compute_ssm_params(
    const float* __restrict__ proj,
    const float* __restrict__ dt_bias,
    float* __restrict__ dt,
    float* __restrict__ decay,
    float* __restrict__ trap,
    int L, int H, int dip, int di, int ds
) {
    int t = blockIdx.x;
    int h = threadIdx.x;
    if (t >= L || h >= H) return;
    int dt_off = 2 * di + 2 * ds;
    int a_off = dt_off + H;
    int tr_off = a_off + H;

    float dt_raw = proj[t * dip + dt_off + h] + dt_bias[h];
    float dt_v = softplus_f(dt_raw);
    dt[t * H + h] = dt_v;

    float a_raw = proj[t * dip + a_off + h];
    float a = -softplus_f(a_raw);
    if (a > -1e-4f) a = -1e-4f;
    decay[t * H + h] = __expf(a * dt_v);

    float tr_raw = proj[t * dip + tr_off + h];
    trap[t * H + h] = sigmoid_f(tr_raw);
}

// ---------- compute_dt_mean ------------------------------------------------
// dt_mean[t] = mean over heads of dt[t, :]
// Grid: (L,), Block: (32,)
extern "C" __global__ void compute_dt_mean(
    const float* __restrict__ dt,
    float* __restrict__ dt_mean,
    int L, int H
) {
    int t = blockIdx.x;
    if (t >= L) return;
    float s = 0.0f;
    for (int h = threadIdx.x; h < H; h += blockDim.x) s += dt[t * H + h];
    s = warp_reduce_sum(s);
    if (threadIdx.x == 0) dt_mean[t] = s / (float)H;
}

// ---------- compute_phase ---------------------------------------------------
// phase[t, k] = sum_{t' <= t} angles[t', k] * dt_mean[t']
// Sequential in t. Grid: (1,), Block: (n_angles,)
// angles_off = 2*di + 2*ds + 3*H
extern "C" __global__ void compute_phase(
    const float* __restrict__ proj,
    const float* __restrict__ dt_mean,
    float* __restrict__ phase,
    int L, int n_angles, int dip, int di, int ds, int H
) {
    int k = threadIdx.x;
    if (k >= n_angles) return;
    int ang_off = 2 * di + 2 * ds + 3 * H;
    float cum = 0.0f;
    for (int t = 0; t < L; t++) {
        cum = __fmaf_rn(proj[t * dip + ang_off + k], dt_mean[t], cum);
        phase[t * n_angles + k] = cum;
    }
}

// ---------- extract_bp_cp ---------------------------------------------------
// Copies bp_raw and cp_raw slices out of proj into contiguous buffers.
// Grid: (L,), Block: (32,)
extern "C" __global__ void extract_bp_cp(
    const float* __restrict__ proj,
    float* __restrict__ bp,
    float* __restrict__ cp,
    int L, int ds, int di, int dip
) {
    int t = blockIdx.x;
    if (t >= L) return;
    int bp_off = 2 * di;
    int cp_off = bp_off + ds;
    for (int n = threadIdx.x; n < ds; n += blockDim.x) {
        bp[t * ds + n] = proj[t * dip + bp_off + n];
        cp[t * ds + n] = proj[t * dip + cp_off + n];
    }
}

// ---------- apply_rope ------------------------------------------------------
// Rotates consecutive pairs (v[t, 2k], v[t, 2k+1]) by angle phase[t, k].
// Grid: (L,), Block: (n_angles,)
extern "C" __global__ void apply_rope(
    float* __restrict__ v,
    const float* __restrict__ phase,
    int L, int ds, int n_angles
) {
    int t = blockIdx.x;
    int k = threadIdx.x;
    if (t >= L || k >= n_angles) return;
    float a = phase[t * n_angles + k];
    float c = __cosf(a);
    float s = __sinf(a);
    int idx_e = t * ds + 2 * k;
    int idx_o = idx_e + 1;
    float e = v[idx_e];
    float o = v[idx_o];
    v[idx_e] = __fmaf_rn(e, c, -o * s);
    v[idx_o] = __fmaf_rn(e, s,  o * c);
}

// ---------- compute_z_silu --------------------------------------------------
// z_silu[t, i] = z_raw[t, i] * sigmoid(z_raw[t, i])   (first di entries of proj)
// Grid: (ceil(di/32), L), Block: (32,)
extern "C" __global__ void compute_z_silu(
    const float* __restrict__ proj,
    float* __restrict__ z_silu,
    int L, int di, int dip
) {
    int t = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L || i >= di) return;
    float z = proj[t * dip + i];
    z_silu[t * di + i] = z * sigmoid_f(z);
}

// ---------- prepare_bc ------------------------------------------------------
// Fused: extract bp_raw/cp_raw from proj -> LayerNorm both -> apply RoPE both.
// Replaces: extract_bp_cp + 2x layer_norm + 2x apply_rope  (4 dispatches -> 1).
//
// Per timestep t we need:
//   bp[t,n] = rope(LN(proj[t, 2*di + n]), phase[t, n/2])
//   cp[t,n] = rope(LN(proj[t, 2*di + ds + n]), phase[t, n/2])
//
// Block layout: one block per t (grid_dim.x = L). Block size must be >= ds.
// Uses shared memory for the LayerNorm reduction and to stage rotated pairs.
// Only supports ds <= 32 (single warp); bigger ds would need multi-warp reduce.
extern "C" __global__ void prepare_bc(
    const float* __restrict__ proj,       // (L, dip)
    const float* __restrict__ b_norm_w,   // (ds,)
    const float* __restrict__ b_norm_b,   // (ds,)
    const float* __restrict__ c_norm_w,   // (ds,)
    const float* __restrict__ c_norm_b,   // (ds,)
    const float* __restrict__ phase,      // (L, n_angles)
    float* __restrict__ bp,               // (L, ds)
    float* __restrict__ cp,               // (L, ds)
    int L, int ds, int n_angles, int di, int dip
) {
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= L) return;
    const float eps = 1e-5f;

    int bp_off = 2 * di;
    int cp_off = bp_off + ds;

    // --- bp ---
    float bv = (tid < ds) ? proj[t * dip + bp_off + tid] : 0.0f;
    // LayerNorm reduction
    float s = warp_reduce_sum(bv);
    float mean = s / (float)ds;
    float diff = bv - mean;
    float vs = warp_reduce_sum((tid < ds) ? diff * diff : 0.0f);
    float inv_std = rsqrtf(vs / (float)ds + eps);
    float bn = (tid < ds) ? __fmaf_rn(diff * inv_std, b_norm_w[tid], b_norm_b[tid]) : 0.0f;

    // Apply RoPE using phase[t, tid/2]. Pairs (2k, 2k+1) rotate together.
    // IMPORTANT: shfl_xor_sync must be called with ALL 32 lanes participating,
    // so we do the shuffle outside any divergent branch. Out-of-bounds lanes
    // compute garbage which is discarded by the guarded write.
    float bp_partner = __shfl_xor_sync(0xffffffff, bn, 1);
    {
        int k = tid >> 1;
        int is_odd = tid & 1;
        float a = (tid < ds) ? phase[t * n_angles + k] : 0.0f;
        float c = __cosf(a);
        float sn = __sinf(a);
        float result;
        if (is_odd == 0) {
            result = __fmaf_rn(bn, c, -bp_partner * sn);
        } else {
            result = __fmaf_rn(bp_partner, sn, bn * c);
        }
        if (tid < ds) bp[t * ds + tid] = result;
    }

    // --- cp --- (same pattern)
    float cv = (tid < ds) ? proj[t * dip + cp_off + tid] : 0.0f;
    s = warp_reduce_sum(cv);
    mean = s / (float)ds;
    diff = cv - mean;
    vs = warp_reduce_sum((tid < ds) ? diff * diff : 0.0f);
    inv_std = rsqrtf(vs / (float)ds + eps);
    float cn = (tid < ds) ? __fmaf_rn(diff * inv_std, c_norm_w[tid], c_norm_b[tid]) : 0.0f;

    float cp_partner = __shfl_xor_sync(0xffffffff, cn, 1);
    {
        int k = tid >> 1;
        int is_odd = tid & 1;
        float a = (tid < ds) ? phase[t * n_angles + k] : 0.0f;
        float c = __cosf(a);
        float sn = __sinf(a);
        float result;
        if (is_odd == 0) {
            result = __fmaf_rn(cn, c, -cp_partner * sn);
        } else {
            result = __fmaf_rn(cp_partner, sn, cn * c);
        }
        if (tid < ds) cp[t * ds + tid] = result;
    }
}

// ---------- ssm_scan_sequential --------------------------------------------
// For each head h (one block), sequential over t:
//   - Compute bx_cur[p,n] = x_raw[h*hd+p] * bp[t,n]  (register per-thread)
//   - inp[p,n]  = (trap*bx_cur + (1-trap)*bx_prev) * dt
//   - state    = decay * state + inp
//   - Reduce sum_n(state[p,n] * cp[t,n]) per p, add D*x_raw, multiply by z_silu
//   - Write y[t, h, p]
//
// Grid: (H,)  Block: (hd * ds,), tid = p * ds + n
// shared memory: hd*ds floats (for reduction) + hd floats (for y_reduce)
// NOTE: z_silu is read inline from proj (z_raw is the first di slice of proj).
// The standalone compute_z_silu kernel is no longer called — one fewer dispatch.
extern "C" __global__ void ssm_scan_sequential(
    const float* __restrict__ proj,
    const float* __restrict__ bp,
    const float* __restrict__ cp,
    const float* __restrict__ dt_in,
    const float* __restrict__ decay_in,
    const float* __restrict__ trap_in,
    const float* __restrict__ d_param,
    float* __restrict__ y,
    int L, int H, int hd, int ds, int di, int dip
) {
    extern __shared__ float smem[];
    float* y_reduce = &smem[hd * ds];

    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= H || tid >= hd * ds) return;
    int p = tid / ds;
    int n = tid % ds;

    int x_off = di;  // x_raw follows z_raw in proj

    float state = 0.0f;
    float bx_prev = 0.0f;

    for (int t = 0; t < L; t++) {
        float bp_tn = bp[t * ds + n];
        float cp_tn = cp[t * ds + n];
        float dt_v = dt_in[t * H + h];
        float dec = decay_in[t * H + h];
        float tr = trap_in[t * H + h];
        float x_val = proj[t * dip + x_off + h * hd + p];

        float bx_cur = x_val * bp_tn;
        float blended = __fmaf_rn(tr, bx_cur, (1.0f - tr) * bx_prev);
        float inp_val = blended * dt_v;
        bx_prev = bx_cur;
        state = __fmaf_rn(dec, state, inp_val);

        // Partial for output reduction: state * cp
        smem[tid] = state * cp_tn;
        __syncthreads();

        // Tree-halving reduce across n for each p-group of ds threads
        for (int stride = ds >> 1; stride > 0; stride >>= 1) {
            if (n < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }

        if (n == 0) {
            float sum = smem[p * ds];
            sum = __fmaf_rn(d_param[h], x_val, sum);
            float z_raw = proj[t * dip + h * hd + p];    // inline z_silu
            float z_silu = z_raw * sigmoid_f(z_raw);
            y_reduce[p] = sum * z_silu;
        }
        __syncthreads();

        if (tid < hd) {
            // Write: y[t, h, p]  layout (L, H, hd)
            y[(t * H + h) * hd + tid] = y_reduce[tid];
        }
        __syncthreads();
    }
}

// ---------- argmax_f32 ------------------------------------------------------
// Per-row argmax of logits: preds[t] = argmax_v(logits[t, v])
// Grid: (L,), Block: (256,). Single warp reduction via shared memory.
extern "C" __global__ void argmax_f32(
    const float* __restrict__ logits,  // (L, V)
    unsigned int* __restrict__ preds,  // (L,)
    int L, int V
) {
    int t = blockIdx.x;
    if (t >= L) return;
    int tid = threadIdx.x;

    __shared__ float smax[32];
    __shared__ int sidx[32];

    float local_max = -3.4028235e38f;  // -FLT_MAX (INFINITY macro unavailable in NVRTC)
    int local_idx = 0;
    for (int v = tid; v < V; v += blockDim.x) {
        float val = logits[t * V + v];
        if (val > local_max) {
            local_max = val;
            local_idx = v;
        }
    }
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1) {
        float other_max = __shfl_xor_sync(0xffffffff, local_max, off);
        int other_idx  = __shfl_xor_sync(0xffffffff, local_idx, off);
        if (other_max > local_max || (other_max == local_max && other_idx < local_idx)) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }
    int warp = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) {
        smax[warp] = local_max;
        sidx[warp] = local_idx;
    }
    __syncthreads();

    if (warp == 0) {
        int nwarps = blockDim.x / 32;
        float mv = (lane < nwarps) ? smax[lane] : -3.4028235e38f;
        int mi = (lane < nwarps) ? sidx[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            float other_mv = __shfl_xor_sync(0xffffffff, mv, off);
            int other_mi  = __shfl_xor_sync(0xffffffff, mi, off);
            if (other_mv > mv || (other_mv == mv && other_mi < mi)) {
                mv = other_mv;
                mi = other_mi;
            }
        }
        if (lane == 0) {
            preds[t] = (unsigned int)mi;
        }
    }
}

// ---------- mamba3_forward_persistent --------------------------------------
//
// THE persistent forward: entire Mamba-3 inference in a single kernel launch.
// One 256-thread block, loops over layers internally, uses shared memory as
// activation scratch. Replaces 49 kernel launches with 1 launch.
//
// Memory budget (per the d=64 L=3 run_length_next model):
//   x, x_normed: L*d = 448 f32 = 1.8 KB each
//   proj: L*dip = 7*320 = 2240 f32 = 8.9 KB
//   dt, decay, trap: L*H = 56 f32 = 224 B each
//   dt_mean: L = 28 B
//   phase: L*n_angles = 56 f32 = 224 B
//   bp, cp: L*ds = 112 f32 = 448 B each
//   y_inner: L*di = 896 f32 = 3.6 KB
//   y_out: L*d = 1.8 KB
//   reduce: 256 f32 = 1 KB
//   Total: ~21 KB — fits comfortably in 228 KB H100 SMEM.
//
// Block: 256 threads (minimum for SSM scan: hd*ds = 16*16 = 256).
// Heads processed sequentially inside the scan phase (H iterations).
//
// Per-layer weight buffers are concatenated on the host side: e.g.
// layers_in_proj_w is [n_layers × (dip * d)] contiguous, indexed by
// (l * dip + j) * d + k.
//
// All arithmetic via __fmaf_rn / explicit sigmoid_f / softplus_f to stay
// bit-identical with the per-op path and CPU reference.
// __launch_bounds__(1024, 1): cap at 1024 threads/block, hint 1 resident block
// per SM. Forces the compiler to budget registers so we fit: 65536 / 1024 = 64
// regs/thread. Without this, nvcc chooses a higher register count and the
// driver rejects the launch with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES at 1024.
extern "C" __launch_bounds__(1024, 1) __global__ void mamba3_forward_persistent(
    const unsigned int* __restrict__ tokens,    // (L,)
    const float* __restrict__ embed_w,          // (V, d)
    const float* __restrict__ embed_norm_w,     // (d,)
    const float* __restrict__ embed_norm_b,     // (d,)
    const float* __restrict__ layers_in_proj_w, // (n_layers, dip, d)
    const float* __restrict__ layers_out_proj_w,// (n_layers, d, di)
    const float* __restrict__ layers_dt_bias,   // (n_layers, H)
    const float* __restrict__ layers_d_param,   // (n_layers, H)
    const float* __restrict__ layers_b_norm_w,  // (n_layers, ds)
    const float* __restrict__ layers_b_norm_b,  // (n_layers, ds)
    const float* __restrict__ layers_c_norm_w,  // (n_layers, ds)
    const float* __restrict__ layers_c_norm_b,  // (n_layers, ds)
    const float* __restrict__ layers_ln_w,      // (n_layers, d)
    const float* __restrict__ layers_ln_b,      // (n_layers, d)
    const float* __restrict__ layers_scale,     // (n_layers,)
    const float* __restrict__ final_norm_w,     // (d,)
    const float* __restrict__ final_norm_b,     // (d,)
    float* __restrict__ logits,                 // (L, V) — HBM output
    int L, int n_layers, int d, int di, int ds, int H, int hd,
    int n_angles, int V, int dip
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    const float eps = 1e-5f;

    // SMEM layout (all offsets in floats).
    float* x         = smem;
    float* x_normed  = x         + L * d;
    float* proj      = x_normed  + L * d;
    float* dt        = proj      + L * dip;
    float* decay     = dt        + L * H;
    float* trap      = decay     + L * H;
    float* dt_mean   = trap      + L * H;
    float* phase     = dt_mean   + L;
    float* bp        = phase     + L * n_angles;
    float* cp        = bp        + L * ds;
    float* y_inner   = cp        + L * ds;
    float* y_out     = y_inner   + L * di;  // di = H * hd
    float* reduce_buf= y_out     + L * d;   // hd * ds slots (256)

    // ---- 1. Embed gather ----
    for (int t = 0; t < L; t++) {
        unsigned int tok = tokens[t];
        bool valid = tok < (unsigned int)V;
        for (int i = tid; i < d; i += num_threads) {
            x[t * d + i] = valid ? embed_w[tok * d + i] : 0.0f;
        }
    }
    __syncthreads();

    // ---- 2. Embed norm (warp per row, first L warps active) ----
    if (wid < L) {
        int t = wid;
        float v = (lid < d) ? x[t * d + lid] : 0.0f;
        for (int i = lid + 32; i < d; i += 32) v += x[t * d + i];
        float s = warp_reduce_sum(v);
        float mean = s / (float)d;
        float diff_sum = 0.0f;
        for (int i = lid; i < d; i += 32) {
            float diff = x[t * d + i] - mean;
            diff_sum = __fmaf_rn(diff, diff, diff_sum);
        }
        float vs = warp_reduce_sum(diff_sum);
        float inv_std = rsqrtf(vs / (float)d + eps);
        for (int i = lid; i < d; i += 32) {
            x[t * d + i] = __fmaf_rn((x[t * d + i] - mean) * inv_std,
                                     embed_norm_w[i], embed_norm_b[i]);
        }
    }
    __syncthreads();

    // ---- 3. Layers ----
    for (int l = 0; l < n_layers; l++) {
        // 3a. Copy x -> x_normed
        for (int i = tid; i < L * d; i += num_threads) x_normed[i] = x[i];
        __syncthreads();

        // 3b. Pre-norm on x_normed
        if (wid < L) {
            int t = wid;
            float v = 0.0f;
            for (int i = lid; i < d; i += 32) v += x_normed[t * d + i];
            float s = warp_reduce_sum(v);
            float mean = s / (float)d;
            float diff_sum = 0.0f;
            for (int i = lid; i < d; i += 32) {
                float diff = x_normed[t * d + i] - mean;
                diff_sum = __fmaf_rn(diff, diff, diff_sum);
            }
            float vs = warp_reduce_sum(diff_sum);
            float inv_std = rsqrtf(vs / (float)d + eps);
            const float* lw = layers_ln_w + l * d;
            const float* lb = layers_ln_b + l * d;
            for (int i = lid; i < d; i += 32) {
                x_normed[t * d + i] = __fmaf_rn(
                    (x_normed[t * d + i] - mean) * inv_std, lw[i], lb[i]);
            }
        }
        __syncthreads();

        // 3c. In-proj matmul: proj[i,j] = sum_k x_normed[i,k] * in_proj_w[j,k]
        const float* ipw = layers_in_proj_w + (size_t)l * dip * d;
        for (int idx = tid; idx < L * dip; idx += num_threads) {
            int ti = idx / dip;
            int ji = idx % dip;
            float acc = 0.0f;
            const float* xrow = x_normed + ti * d;
            const float* brow = ipw + ji * d;
            for (int k = 0; k < d; k++) acc = __fmaf_rn(xrow[k], brow[k], acc);
            proj[idx] = acc;
        }
        __syncthreads();

        // 3d. SSM params: dt, decay, trap per (t, h)
        int dt_off = 2 * di + 2 * ds;
        int a_off  = dt_off + H;
        int tr_off = a_off + H;
        int ang_off= tr_off + H;
        const float* dbias = layers_dt_bias + l * H;
        for (int idx = tid; idx < L * H; idx += num_threads) {
            int ti = idx / H;
            int hi = idx % H;
            float dt_raw = proj[ti * dip + dt_off + hi] + dbias[hi];
            float dt_v = softplus_f(dt_raw);
            dt[idx] = dt_v;
            float a_raw = proj[ti * dip + a_off + hi];
            float a = -softplus_f(a_raw);
            if (a > -1e-4f) a = -1e-4f;
            decay[idx] = __expf(a * dt_v);
            float tr_raw = proj[ti * dip + tr_off + hi];
            trap[idx] = sigmoid_f(tr_raw);
        }
        __syncthreads();

        // 3e. dt_mean[t] = mean over heads
        for (int t = tid; t < L; t += num_threads) {
            float s = 0.0f;
            for (int h = 0; h < H; h++) s += dt[t * H + h];
            dt_mean[t] = s / (float)H;
        }
        __syncthreads();

        // 3f. phase[t, k] = cumsum over t of angle[t,k] * dt_mean[t]
        if (tid < n_angles) {
            int k = tid;
            float cum = 0.0f;
            for (int t = 0; t < L; t++) {
                cum = __fmaf_rn(proj[t * dip + ang_off + k], dt_mean[t], cum);
                phase[t * n_angles + k] = cum;
            }
        }
        __syncthreads();

        // 3g. Extract bp/cp, LayerNorm, apply RoPE (all per timestep, warp per t)
        int bp_off = 2 * di;
        int cp_off = bp_off + ds;
        const float* bnw = layers_b_norm_w + l * ds;
        const float* bnb = layers_b_norm_b + l * ds;
        const float* cnw = layers_c_norm_w + l * ds;
        const float* cnb = layers_c_norm_b + l * ds;
        if (wid < L) {
            int t = wid;
            // bp
            float bv = (lid < ds) ? proj[t * dip + bp_off + lid] : 0.0f;
            float sb = warp_reduce_sum(bv);
            float meanb = sb / (float)ds;
            float diffb = bv - meanb;
            float varb = warp_reduce_sum((lid < ds) ? diffb * diffb : 0.0f);
            float inv_std_b = rsqrtf(varb / (float)ds + eps);
            float bn = (lid < ds) ? __fmaf_rn(diffb * inv_std_b, bnw[lid], bnb[lid]) : 0.0f;
            float bp_partner = __shfl_xor_sync(0xffffffff, bn, 1);
            int kr = lid >> 1;
            int is_odd = lid & 1;
            float ang = (lid < ds) ? phase[t * n_angles + kr] : 0.0f;
            float co = __cosf(ang);
            float si = __sinf(ang);
            float b_res = (is_odd == 0)
                ? __fmaf_rn(bn, co, -bp_partner * si)
                : __fmaf_rn(bp_partner, si, bn * co);
            if (lid < ds) bp[t * ds + lid] = b_res;

            // cp
            float cv = (lid < ds) ? proj[t * dip + cp_off + lid] : 0.0f;
            float sc = warp_reduce_sum(cv);
            float meanc = sc / (float)ds;
            float diffc = cv - meanc;
            float varc = warp_reduce_sum((lid < ds) ? diffc * diffc : 0.0f);
            float inv_std_c = rsqrtf(varc / (float)ds + eps);
            float cn = (lid < ds) ? __fmaf_rn(diffc * inv_std_c, cnw[lid], cnb[lid]) : 0.0f;
            float cp_partner = __shfl_xor_sync(0xffffffff, cn, 1);
            float c_res = (is_odd == 0)
                ? __fmaf_rn(cn, co, -cp_partner * si)
                : __fmaf_rn(cp_partner, si, cn * co);
            if (lid < ds) cp[t * ds + lid] = c_res;
        }
        __syncthreads();

        // 3h. SSM scan — process HEADS_PARALLEL heads at a time, using
        // num_threads / (hd*ds) head-slots simultaneously. With 1024 threads
        // and hd*ds=256, we do 4 heads in parallel per outer iteration.
        const float* dparam = layers_d_param + l * H;
        int head_stride = hd * ds;                          // 256 threads per head slot
        int heads_parallel = num_threads / head_stride;     // 4 slots with 1024 threads
        int h_local = tid / head_stride;                    // 0..heads_parallel-1
        int tid_in_head = tid - h_local * head_stride;
        int p = tid_in_head / ds;
        int n = tid_in_head - p * ds;
        bool in_slot = tid < heads_parallel * head_stride;  // drops any tail threads

        for (int h_base = 0; h_base < H; h_base += heads_parallel) {
            int h = h_base + h_local;
            bool head_ok = in_slot && (h < H);
            float state = 0.0f;
            float bx_prev = 0.0f;

            for (int t = 0; t < L; t++) {
                // Preload state-advance inputs (per-thread).
                float bp_tn = bp[t * ds + n];
                float cp_tn = cp[t * ds + n];
                float dt_v = head_ok ? dt[t * H + h]      : 0.0f;
                float dec  = head_ok ? decay[t * H + h]   : 0.0f;
                float tr   = head_ok ? trap[t * H + h]    : 0.0f;
                float x_val = head_ok ? proj[t * dip + di + h * hd + p] : 0.0f;

                if (head_ok) {
                    float bx_cur = x_val * bp_tn;
                    float blended = __fmaf_rn(tr, bx_cur, (1.0f - tr) * bx_prev);
                    float inp_val = blended * dt_v;
                    bx_prev = bx_cur;
                    state = __fmaf_rn(dec, state, inp_val);
                    reduce_buf[tid] = state * cp_tn;
                }
                __syncthreads();

                // Tree reduction across n within each head slot's p-group of ds threads.
                for (int stride = ds >> 1; stride > 0; stride >>= 1) {
                    if (head_ok && n < stride) {
                        reduce_buf[tid] += reduce_buf[tid + stride];
                    }
                    __syncthreads();
                }

                if (head_ok && n == 0) {
                    float sum = reduce_buf[h_local * head_stride + p * ds];
                    sum = __fmaf_rn(dparam[h], x_val, sum);
                    float z = proj[t * dip + h * hd + p];  // z_raw for this (h, p)
                    float z_silu = z * sigmoid_f(z);
                    y_inner[(t * H + h) * hd + p] = sum * z_silu;
                }
                __syncthreads();
            }
        }

        // 3i. Out-proj matmul: y_out[i,j] = sum_k y_inner[i,k] * out_proj_w[j,k]
        const float* opw = layers_out_proj_w + (size_t)l * d * di;
        for (int idx = tid; idx < L * d; idx += num_threads) {
            int ti = idx / d;
            int ji = idx % d;
            float acc = 0.0f;
            const float* yrow = y_inner + ti * di;
            const float* brow = opw + ji * di;
            for (int k = 0; k < di; k++) acc = __fmaf_rn(yrow[k], brow[k], acc);
            y_out[idx] = acc;
        }
        __syncthreads();

        // 3j. Residual: x += scale * y_out
        float scl = layers_scale[l];
        for (int i = tid; i < L * d; i += num_threads) {
            x[i] = __fmaf_rn(scl, y_out[i], x[i]);
        }
        __syncthreads();
    }

    // ---- 4. Final norm ----
    if (wid < L) {
        int t = wid;
        float v = 0.0f;
        for (int i = lid; i < d; i += 32) v += x[t * d + i];
        float s = warp_reduce_sum(v);
        float mean = s / (float)d;
        float diff_sum = 0.0f;
        for (int i = lid; i < d; i += 32) {
            float diff = x[t * d + i] - mean;
            diff_sum = __fmaf_rn(diff, diff, diff_sum);
        }
        float vs = warp_reduce_sum(diff_sum);
        float inv_std = rsqrtf(vs / (float)d + eps);
        for (int i = lid; i < d; i += 32) {
            x[t * d + i] = __fmaf_rn(
                (x[t * d + i] - mean) * inv_std,
                final_norm_w[i], final_norm_b[i]);
        }
    }
    __syncthreads();

    // ---- 5. LM head (writes to HBM) ----
    for (int idx = tid; idx < L * V; idx += num_threads) {
        int ti = idx / V;
        int vi = idx % V;
        float acc = 0.0f;
        const float* xrow = x + ti * d;
        const float* brow = embed_w + vi * d;
        for (int k = 0; k < d; k++) acc = __fmaf_rn(xrow[k], brow[k], acc);
        logits[idx] = acc;
    }
}

// ---------- mamba3_forward_coop --------------------------------------------
//
// Cooperative-groups multi-block persistent forward. ONE kernel launch for the
// entire Mamba-3 inference. Many blocks (one per SM), `cg::this_grid().sync()`
// between phases, intermediate tensors live in HBM scratch (not SMEM).
//
// Why: on a contested H100, each per-op kernel dispatch has to fight for SM
// slots. 45 dispatches per forward = 45 contention events. This kernel makes
// it 1 dispatch, 1 contention event, then the whole forward runs through.
//
// Launch requirements:
//   grid_dim.x   = COOP_BLOCKS (default 64; must not exceed device max active
//                  blocks for this kernel)
//   block_dim.x  = 256 threads
//   cooperative launch (cuLaunchCooperativeKernel)
//
// __launch_bounds__(256, 2): 256 threads/block, min 2 blocks/SM.  That caps
// regs at 65536/(256*2) = 128 per thread.  Ample headroom.
extern "C" __launch_bounds__(256, 2) __global__ void mamba3_forward_coop(
    const unsigned int* __restrict__ tokens,
    const float* __restrict__ embed_w,
    const float* __restrict__ embed_norm_w,
    const float* __restrict__ embed_norm_b,
    const float* __restrict__ layers_in_proj_w,
    const float* __restrict__ layers_out_proj_w,
    const float* __restrict__ layers_dt_bias,
    const float* __restrict__ layers_d_param,
    const float* __restrict__ layers_b_norm_w,
    const float* __restrict__ layers_b_norm_b,
    const float* __restrict__ layers_c_norm_w,
    const float* __restrict__ layers_c_norm_b,
    const float* __restrict__ layers_ln_w,
    const float* __restrict__ layers_ln_b,
    const float* __restrict__ layers_scale,
    const float* __restrict__ final_norm_w,
    const float* __restrict__ final_norm_b,
    float* __restrict__ x_buf,
    float* __restrict__ x_normed_buf,
    float* __restrict__ proj_buf,
    float* __restrict__ dt_buf,
    float* __restrict__ decay_buf,
    float* __restrict__ trap_buf,
    float* __restrict__ dt_mean_buf,
    float* __restrict__ phase_buf,
    float* __restrict__ bp_buf,
    float* __restrict__ cp_buf,
    float* __restrict__ y_inner_buf,
    float* __restrict__ y_out_buf,
    float* __restrict__ logits,
    int L, int n_layers, int d, int di, int ds, int H, int hd,
    int n_angles, int V, int dip
) {
    cg::grid_group grid = cg::this_grid();
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gtid = bid * blockDim.x + tid;
    int gdim = gridDim.x * blockDim.x;
    int wid = tid >> 5;
    int lid = tid & 31;
    const float eps = 1e-5f;

    __shared__ float warp_scratch[8];   // for block-wide reductions (assuming <= 8 warps)
    __shared__ float ssm_reduce[256];   // scan reduction (used in Phase 8 only)
    __shared__ float y_local[64];       // per-head SSM output per timestep

    // --- Phase 1: embed gather ---
    for (int idx = gtid; idx < L * d; idx += gdim) {
        int t = idx / d;
        int i = idx - t * d;
        unsigned int tok = tokens[t];
        x_buf[idx] = (tok < (unsigned int)V) ? embed_w[tok * d + i] : 0.0f;
    }
    grid.sync();

    // --- Phase 2: embed norm (one block per row) ---
    if (bid < L) {
        int t = bid;
        float v = 0.0f;
        for (int i = tid; i < d; i += blockDim.x) v += x_buf[t * d + i];
        v = warp_reduce_sum(v);
        if (lid == 0) warp_scratch[wid] = v;
        __syncthreads();
        if (wid == 0) {
            int nw = (blockDim.x + 31) >> 5;
            v = (lid < nw) ? warp_scratch[lid] : 0.0f;
            v = warp_reduce_sum(v);
            if (lid == 0) warp_scratch[0] = v;
        }
        __syncthreads();
        float mean = warp_scratch[0] / (float)d;

        float vs = 0.0f;
        for (int i = tid; i < d; i += blockDim.x) {
            float diff = x_buf[t * d + i] - mean;
            vs = __fmaf_rn(diff, diff, vs);
        }
        vs = warp_reduce_sum(vs);
        if (lid == 0) warp_scratch[wid] = vs;
        __syncthreads();
        if (wid == 0) {
            int nw = (blockDim.x + 31) >> 5;
            vs = (lid < nw) ? warp_scratch[lid] : 0.0f;
            vs = warp_reduce_sum(vs);
            if (lid == 0) warp_scratch[0] = vs;
        }
        __syncthreads();
        float inv_std = rsqrtf(warp_scratch[0] / (float)d + eps);

        for (int i = tid; i < d; i += blockDim.x) {
            x_buf[t * d + i] = __fmaf_rn((x_buf[t * d + i] - mean) * inv_std,
                                          embed_norm_w[i], embed_norm_b[i]);
        }
    }
    grid.sync();

    // --- Layers ---
    for (int l = 0; l < n_layers; l++) {
        // 3a: copy x -> x_normed
        for (int idx = gtid; idx < L * d; idx += gdim) x_normed_buf[idx] = x_buf[idx];
        grid.sync();

        // 3b: pre-norm on x_normed
        if (bid < L) {
            int t = bid;
            const float* lw = &layers_ln_w[l * d];
            const float* lb = &layers_ln_b[l * d];

            float v = 0.0f;
            for (int i = tid; i < d; i += blockDim.x) v += x_normed_buf[t * d + i];
            v = warp_reduce_sum(v);
            if (lid == 0) warp_scratch[wid] = v;
            __syncthreads();
            if (wid == 0) {
                int nw = (blockDim.x + 31) >> 5;
                v = (lid < nw) ? warp_scratch[lid] : 0.0f;
                v = warp_reduce_sum(v);
                if (lid == 0) warp_scratch[0] = v;
            }
            __syncthreads();
            float mean = warp_scratch[0] / (float)d;

            float vs = 0.0f;
            for (int i = tid; i < d; i += blockDim.x) {
                float diff = x_normed_buf[t * d + i] - mean;
                vs = __fmaf_rn(diff, diff, vs);
            }
            vs = warp_reduce_sum(vs);
            if (lid == 0) warp_scratch[wid] = vs;
            __syncthreads();
            if (wid == 0) {
                int nw = (blockDim.x + 31) >> 5;
                vs = (lid < nw) ? warp_scratch[lid] : 0.0f;
                vs = warp_reduce_sum(vs);
                if (lid == 0) warp_scratch[0] = vs;
            }
            __syncthreads();
            float inv_std = rsqrtf(warp_scratch[0] / (float)d + eps);

            for (int i = tid; i < d; i += blockDim.x) {
                x_normed_buf[t * d + i] = __fmaf_rn(
                    (x_normed_buf[t * d + i] - mean) * inv_std, lw[i], lb[i]);
            }
        }
        grid.sync();

        // 3c: in_proj matmul, grid-striped
        const float* ipw = &layers_in_proj_w[(size_t)l * dip * d];
        for (int idx = gtid; idx < L * dip; idx += gdim) {
            int ti = idx / dip;
            int ji = idx - ti * dip;
            float acc = 0.0f;
            const float* xr = &x_normed_buf[ti * d];
            const float* br = &ipw[ji * d];
            for (int k = 0; k < d; k++) acc = __fmaf_rn(xr[k], br[k], acc);
            proj_buf[idx] = acc;
        }
        grid.sync();

        // 3d: dt/decay/trap and dt_mean (one block per t)
        if (bid < L) {
            int t = bid;
            int dt_off = 2 * di + 2 * ds;
            int a_off = dt_off + H;
            int tr_off = a_off + H;
            const float* dbias = &layers_dt_bias[l * H];

            float my_dt = 0.0f;
            for (int h = tid; h < H; h += blockDim.x) {
                float dt_raw = proj_buf[t * dip + dt_off + h] + dbias[h];
                float dt_v = softplus_f(dt_raw);
                dt_buf[t * H + h] = dt_v;
                my_dt += dt_v;
                float a_raw = proj_buf[t * dip + a_off + h];
                float a = -softplus_f(a_raw);
                if (a > -1e-4f) a = -1e-4f;
                decay_buf[t * H + h] = __expf(a * dt_v);
                trap_buf[t * H + h] = sigmoid_f(proj_buf[t * dip + tr_off + h]);
            }
            // H <= 32 → single-warp reduce suffices.
            my_dt = warp_reduce_sum(my_dt);
            if (tid == 0) dt_mean_buf[t] = my_dt / (float)H;
        }
        grid.sync();

        // 3e: phase cumulative sum (single block, thread per angle)
        if (bid == 0 && tid < n_angles) {
            int ang_off = 2 * di + 2 * ds + 3 * H;
            float cum = 0.0f;
            for (int t = 0; t < L; t++) {
                cum = __fmaf_rn(proj_buf[t * dip + ang_off + tid], dt_mean_buf[t], cum);
                phase_buf[t * n_angles + tid] = cum;
            }
        }
        grid.sync();

        // 3f: extract bp/cp + LN + RoPE (one block per t, single warp works since ds <= 32)
        if (bid < L) {
            int t = bid;
            int bp_off = 2 * di;
            int cp_off = bp_off + ds;
            const float* bnw = &layers_b_norm_w[l * ds];
            const float* bnb = &layers_b_norm_b[l * ds];
            const float* cnw = &layers_c_norm_w[l * ds];
            const float* cnb = &layers_c_norm_b[l * ds];

            if (tid < 32) {
                // bp branch
                float bv = (tid < ds) ? proj_buf[t * dip + bp_off + tid] : 0.0f;
                float sb = warp_reduce_sum(bv);
                float meanb = sb / (float)ds;
                float diffb = bv - meanb;
                float varb = warp_reduce_sum((tid < ds) ? diffb * diffb : 0.0f);
                float inv_std_b = rsqrtf(varb / (float)ds + eps);
                float bn = (tid < ds) ? __fmaf_rn(diffb * inv_std_b, bnw[tid], bnb[tid]) : 0.0f;
                float bp_partner = __shfl_xor_sync(0xffffffff, bn, 1);
                int kr = tid >> 1;
                int is_odd = tid & 1;
                float ang = (tid < ds) ? phase_buf[t * n_angles + kr] : 0.0f;
                float co = __cosf(ang);
                float si = __sinf(ang);
                float b_res = (is_odd == 0)
                    ? __fmaf_rn(bn, co, -bp_partner * si)
                    : __fmaf_rn(bp_partner, si, bn * co);
                if (tid < ds) bp_buf[t * ds + tid] = b_res;

                // cp branch
                float cv = (tid < ds) ? proj_buf[t * dip + cp_off + tid] : 0.0f;
                float sc = warp_reduce_sum(cv);
                float meanc = sc / (float)ds;
                float diffc = cv - meanc;
                float varc = warp_reduce_sum((tid < ds) ? diffc * diffc : 0.0f);
                float inv_std_c = rsqrtf(varc / (float)ds + eps);
                float cn = (tid < ds) ? __fmaf_rn(diffc * inv_std_c, cnw[tid], cnb[tid]) : 0.0f;
                float cp_partner = __shfl_xor_sync(0xffffffff, cn, 1);
                float c_res = (is_odd == 0)
                    ? __fmaf_rn(cn, co, -cp_partner * si)
                    : __fmaf_rn(cp_partner, si, cn * co);
                if (tid < ds) cp_buf[t * ds + tid] = c_res;
            }
        }
        grid.sync();

        // 3g: SSM scan — one block per head
        if (bid < H) {
            int h = bid;
            const float* dparam = &layers_d_param[l * H];

            int p = tid / ds;
            int n = tid - p * ds;
            bool active = tid < hd * ds;
            float state = 0.0f;
            float bx_prev = 0.0f;

            for (int t = 0; t < L; t++) {
                float bp_tn = active ? bp_buf[t * ds + n] : 0.0f;
                float cp_tn = active ? cp_buf[t * ds + n] : 0.0f;
                float dt_v = active ? dt_buf[t * H + h] : 0.0f;
                float dec  = active ? decay_buf[t * H + h] : 0.0f;
                float tr   = active ? trap_buf[t * H + h] : 0.0f;
                float x_val = active ? proj_buf[t * dip + di + h * hd + p] : 0.0f;

                if (active) {
                    float bx_cur = x_val * bp_tn;
                    float blended = __fmaf_rn(tr, bx_cur, (1.0f - tr) * bx_prev);
                    float inp_val = blended * dt_v;
                    bx_prev = bx_cur;
                    state = __fmaf_rn(dec, state, inp_val);
                    ssm_reduce[tid] = state * cp_tn;
                }
                __syncthreads();

                for (int stride = ds >> 1; stride > 0; stride >>= 1) {
                    if (active && n < stride) {
                        ssm_reduce[tid] += ssm_reduce[tid + stride];
                    }
                    __syncthreads();
                }

                if (active && n == 0) {
                    float sum = ssm_reduce[p * ds];
                    sum = __fmaf_rn(dparam[h], x_val, sum);
                    float z_raw = proj_buf[t * dip + h * hd + p];
                    float z_silu = z_raw * sigmoid_f(z_raw);
                    y_local[p] = sum * z_silu;
                }
                __syncthreads();

                if (tid < hd) {
                    y_inner_buf[(t * H + h) * hd + tid] = y_local[tid];
                }
                __syncthreads();
            }
        }
        grid.sync();

        // 3h: out_proj matmul, grid-striped
        const float* opw = &layers_out_proj_w[(size_t)l * d * di];
        for (int idx = gtid; idx < L * d; idx += gdim) {
            int ti = idx / d;
            int ji = idx - ti * d;
            float acc = 0.0f;
            const float* yr = &y_inner_buf[ti * di];
            const float* br = &opw[ji * di];
            for (int k = 0; k < di; k++) acc = __fmaf_rn(yr[k], br[k], acc);
            y_out_buf[idx] = acc;
        }
        grid.sync();

        // 3i: residual, grid-striped
        float scl = layers_scale[l];
        for (int idx = gtid; idx < L * d; idx += gdim) {
            x_buf[idx] = __fmaf_rn(scl, y_out_buf[idx], x_buf[idx]);
        }
        grid.sync();
    }

    // --- Phase 4: final norm (one block per row) ---
    if (bid < L) {
        int t = bid;
        float v = 0.0f;
        for (int i = tid; i < d; i += blockDim.x) v += x_buf[t * d + i];
        v = warp_reduce_sum(v);
        if (lid == 0) warp_scratch[wid] = v;
        __syncthreads();
        if (wid == 0) {
            int nw = (blockDim.x + 31) >> 5;
            v = (lid < nw) ? warp_scratch[lid] : 0.0f;
            v = warp_reduce_sum(v);
            if (lid == 0) warp_scratch[0] = v;
        }
        __syncthreads();
        float mean = warp_scratch[0] / (float)d;

        float vs = 0.0f;
        for (int i = tid; i < d; i += blockDim.x) {
            float diff = x_buf[t * d + i] - mean;
            vs = __fmaf_rn(diff, diff, vs);
        }
        vs = warp_reduce_sum(vs);
        if (lid == 0) warp_scratch[wid] = vs;
        __syncthreads();
        if (wid == 0) {
            int nw = (blockDim.x + 31) >> 5;
            vs = (lid < nw) ? warp_scratch[lid] : 0.0f;
            vs = warp_reduce_sum(vs);
            if (lid == 0) warp_scratch[0] = vs;
        }
        __syncthreads();
        float inv_std = rsqrtf(warp_scratch[0] / (float)d + eps);

        for (int i = tid; i < d; i += blockDim.x) {
            x_buf[t * d + i] = __fmaf_rn(
                (x_buf[t * d + i] - mean) * inv_std,
                final_norm_w[i], final_norm_b[i]);
        }
    }
    grid.sync();

    // --- Phase 5: LM head, grid-striped, output to HBM logits ---
    for (int idx = gtid; idx < L * V; idx += gdim) {
        int ti = idx / V;
        int vi = idx - ti * V;
        float acc = 0.0f;
        const float* xr = &x_buf[ti * d];
        const float* br = &embed_w[vi * d];
        for (int k = 0; k < d; k++) acc = __fmaf_rn(xr[k], br[k], acc);
        logits[idx] = acc;
    }
}

// ============================================================================
//                              TRAINING KERNELS
// ============================================================================

// ---------- fill_zero -------------------------------------------------------
extern "C" __global__ void fill_zero(float* __restrict__ buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.0f;
}

// ---------- matmul_ab_tiled -------------------------------------------------
// C = A @ B.  A(M, K), B(K, N), C(M, N).  Standard (non-transpose) matmul.
// Grid: (ceil(N/16), ceil(M/16)), Block: (16, 16).
extern "C" __global__ void matmul_ab_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[16 * 16];
    __shared__ float Bs[16 * 16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * 16 + tx;
    int row = blockIdx.y * 16 + ty;
    float acc = 0.0f;
    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        int a_k = k_tile + tx;
        As[ty * 16 + tx] = (row < M && a_k < K) ? A[row * K + a_k] : 0.0f;
        int b_k = k_tile + ty;
        Bs[ty * 16 + tx] = (col < N && b_k < K) ? B[b_k * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            acc = __fmaf_rn(As[ty * 16 + k], Bs[k * 16 + tx], acc);
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------- matmul_atb_tiled ------------------------------------------------
// C = A^T @ B.  A(K, M), B(K, N), C(M, N).
// C[m, n] = sum_k A[k, m] * B[k, n]
extern "C" __global__ void matmul_atb_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[16 * 16];
    __shared__ float Bs[16 * 16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int m_col = blockIdx.y * 16 + tx;  // M dimension at tx (for load)
    int n_col = blockIdx.x * 16 + tx;  // N dimension at tx (for load)
    int row = blockIdx.y * 16 + ty;    // output m index (for compute)
    int col = blockIdx.x * 16 + tx;    // output n index (for compute)
    float acc = 0.0f;
    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        int k = k_tile + ty;
        As[ty * 16 + tx] = (k < K && m_col < M) ? A[k * M + m_col] : 0.0f;
        Bs[ty * 16 + tx] = (k < K && n_col < N) ? B[k * N + n_col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            // A[k_tile+kk, row] is As[kk, ty]
            // B[k_tile+kk, col] is Bs[kk, tx]
            acc = __fmaf_rn(As[kk * 16 + ty], Bs[kk * 16 + tx], acc);
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------- layer_norm_bwd --------------------------------------------------
// For each row t of x (L, d): compute d_x, accumulate d_w, d_b via atomics.
// Matches mamba3_engine::backward::layer_norm_backward exactly.
// Grid: (L,), Block: (64,) — or (32,) single warp.
extern "C" __global__ void layer_norm_bwd(
    const float* __restrict__ d_out,   // (L, d)
    const float* __restrict__ x,       // (L, d) - input saved from forward
    const float* __restrict__ w,       // (d,)
    float* __restrict__ d_x,           // (L, d)
    float* __restrict__ d_w,           // (d,) - accumulated via atomicAdd
    float* __restrict__ d_b,           // (d,) - accumulated via atomicAdd
    int L, int d
) {
    int t = blockIdx.x;
    if (t >= L) return;
    int tid = threadIdx.x;
    int off = t * d;
    const float eps = 1e-5f;
    int wid = tid >> 5;
    int lid = tid & 31;

    __shared__ float s_mean, s_var, s_dvar, s_dmean;
    __shared__ float warp_buf[8];

    // mean
    float v = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) v += x[off + i];
    v = warp_reduce_sum(v);
    if (lid == 0) warp_buf[wid] = v;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) >> 5;
        v = (lid < nw) ? warp_buf[lid] : 0.0f;
        v = warp_reduce_sum(v);
        if (lid == 0) s_mean = v / (float)d;
    }
    __syncthreads();
    float mean = s_mean;

    // var
    float vs = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float diff = x[off + i] - mean;
        vs = __fmaf_rn(diff, diff, vs);
    }
    vs = warp_reduce_sum(vs);
    if (lid == 0) warp_buf[wid] = vs;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) >> 5;
        vs = (lid < nw) ? warp_buf[lid] : 0.0f;
        vs = warp_reduce_sum(vs);
        if (lid == 0) s_var = vs / (float)d;
    }
    __syncthreads();
    float var = s_var;
    float inv_std = rsqrtf(var + eps);

    // d_var, d_mean reductions
    float my_dvar = 0.0f;
    float my_dmean_a = 0.0f;
    float my_dmean_b = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float x_mm = x[off + i] - mean;
        float d_x_norm = d_out[off + i] * w[i];
        my_dvar += d_x_norm * x_mm * (-0.5f) * inv_std * inv_std * inv_std;
        my_dmean_a += -d_x_norm * inv_std;
        my_dmean_b += -2.0f * x_mm;
    }
    my_dvar    = warp_reduce_sum(my_dvar);
    my_dmean_a = warp_reduce_sum(my_dmean_a);
    my_dmean_b = warp_reduce_sum(my_dmean_b);

    __shared__ float wb_dvar[8], wb_dmean_a[8], wb_dmean_b[8];
    if (lid == 0) {
        wb_dvar[wid]    = my_dvar;
        wb_dmean_a[wid] = my_dmean_a;
        wb_dmean_b[wid] = my_dmean_b;
    }
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) >> 5;
        my_dvar    = (lid < nw) ? wb_dvar[lid]    : 0.0f;
        my_dmean_a = (lid < nw) ? wb_dmean_a[lid] : 0.0f;
        my_dmean_b = (lid < nw) ? wb_dmean_b[lid] : 0.0f;
        my_dvar    = warp_reduce_sum(my_dvar);
        my_dmean_a = warp_reduce_sum(my_dmean_a);
        my_dmean_b = warp_reduce_sum(my_dmean_b);
        if (lid == 0) {
            s_dvar  = my_dvar;
            s_dmean = my_dmean_a + my_dvar * my_dmean_b / (float)d;
        }
    }
    __syncthreads();
    float d_var  = s_dvar;
    float d_mean = s_dmean;

    // final d_x write + atomic d_w, d_b accumulation
    for (int i = tid; i < d; i += blockDim.x) {
        float x_mm = x[off + i] - mean;
        float x_norm = x_mm * inv_std;
        float d_x_norm = d_out[off + i] * w[i];

        d_x[off + i] = d_x_norm * inv_std
            + d_var * 2.0f * x_mm / (float)d
            + d_mean / (float)d;

        atomicAdd(&d_w[i], d_out[off + i] * x_norm);
        atomicAdd(&d_b[i], d_out[off + i]);
    }
}

// ---------- gate_bwd --------------------------------------------------------
// Matches the gate-backward section in train.rs backward_analytical.
//   d_y_pregate[t, i] = d_y_inner[t, i] * z_silu(z_raw)
//   d_proj[t, i]      = d_y_inner[t, i] * y_pregate(z_raw, y_inner) * dsilu_dz(z_raw)
// where z_silu = z_raw * sigmoid(z_raw), dsilu_dz = sigmoid + z*sigmoid*(1-sigmoid).
// Grid: (ceil(L*di/256),), Block: (256,)
extern "C" __global__ void gate_bwd(
    const float* __restrict__ d_y_inner,
    const float* __restrict__ y_inner,
    const float* __restrict__ proj,       // z_raw in first di per row
    float* __restrict__ d_proj,           // writes z slice [0..di]
    float* __restrict__ d_y_pregate,
    int L, int di, int dip
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = L * di;
    if (idx >= total) return;
    int t = idx / di;
    int i = idx - t * di;

    float z_raw = proj[t * dip + i];
    float s = sigmoid_f(z_raw);
    float z_silu = z_raw * s;

    float yi = y_inner[idx];
    float y_pregate = (fabsf(z_silu) > 1e-8f) ? yi / z_silu : 0.0f;

    float dy = d_y_inner[idx];
    d_y_pregate[idx] = dy * z_silu;

    float dsilu_dz = s + z_raw * s * (1.0f - s);
    d_proj[t * dip + i] = dy * y_pregate * dsilu_dz;
}

// ---------- ssm_scan_bwd ----------------------------------------------------
// Adjoint scan: reverse over t.  One block per head. Each thread owns (p, n).
// Inputs:   d_y_pregate, proj, cp_saved, decay_saved, states, d_param
// Outputs:  d_proj (atomicAdd to x_skip slice and cp slice), d_scan_inp
// NOTE: d_proj must be pre-zeroed (except the z-slice from gate_bwd which is
// a non-accumulating assignment).  Different slices of d_proj are written by
// different kernels; all use atomicAdd except gate_bwd which assigns the z
// slice exclusively.  Zero d_proj first, then run gate_bwd, then ssm_scan_bwd,
// then bx_bwd.
extern "C" __global__ void ssm_scan_bwd(
    const float* __restrict__ d_y_pregate,
    const float* __restrict__ proj,
    const float* __restrict__ cp,
    const float* __restrict__ decay,
    const float* __restrict__ states,
    const float* __restrict__ d_param,
    float* __restrict__ d_proj,
    float* __restrict__ d_scan_inp,
    int L, int H, int hd, int ds, int di, int dip
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= H || tid >= hd * ds) return;
    int p = tid / ds;
    int n = tid - p * ds;

    float dh = 0.0f;

    for (int t = L - 1; t >= 0; t--) {
        float dy_pre = d_y_pregate[t * di + h * hd + p];
        float cp_tn = cp[t * ds + n];

        // Step 1: dh += dy_pre * cp
        dh = __fmaf_rn(dy_pre, cp_tn, dh);

        // Step 2: d_proj[x_skip + h*hd+p] += dy_pre * d_param[h] (only n=0 writes)
        if (n == 0) {
            atomicAdd(&d_proj[t * dip + di + h * hd + p],
                      dy_pre * d_param[h]);
        }

        // Step 3: d_cp[t, n] += sum_p(dy_pre * state[t+1, h, p, n])
        float state_tp1 = states[(((t + 1) * H + h) * hd + p) * ds + n];
        atomicAdd(&d_proj[t * dip + 2 * di + ds + n],
                  dy_pre * state_tp1);

        // Step 4: d_scan_inp[t, h, p, n] = dh (current value)
        d_scan_inp[((t * H + h) * hd + p) * ds + n] = dh;

        // Step 5: propagate adjoint state
        float dec = decay[t * H + h];
        dh *= dec;
    }
}

// ---------- ssm_scan_bwd_full ----------------------------------------------
// Extension of ssm_scan_bwd: also accumulates
//   d_decay[t, h]      (via block-wide reduction of dh · state[t])
//   d_d_param[h]       (via atomicAdd of dy_pre · x_raw across t, p)
//   d_dt_from_inp[t, h] (via block-wide reduction of dh · blended)
//
// blended is recomputed as inp_val / dt since inp_val = states[t+1] - decay·states[t]
// is implicit in the scan (states are saved).
//
// Grid: (H,), Block: (hd*ds,)
extern "C" __launch_bounds__(256, 2) __global__ void ssm_scan_bwd_full(
    const float* __restrict__ d_y_pregate,
    const float* __restrict__ proj,
    const float* __restrict__ cp,
    const float* __restrict__ decay,
    const float* __restrict__ dt_in,
    const float* __restrict__ states,
    const float* __restrict__ d_param,
    float* __restrict__ d_proj,
    float* __restrict__ d_scan_inp,
    float* __restrict__ d_decay,        // (L, H)
    float* __restrict__ d_d_param,      // (H,)
    float* __restrict__ d_dt_from_inp,  // (L, H) — inp-path contribution only
    int L, int H, int hd, int ds, int di, int dip
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= H || tid >= hd * ds) return;
    int p = tid / ds;
    int n = tid - p * ds;

    __shared__ float smem[256];  // reduction buffer (hd*ds slots)

    float dh = 0.0f;

    for (int t = L - 1; t >= 0; t--) {
        float dy_pre = d_y_pregate[t * di + h * hd + p];
        float cp_tn = cp[t * ds + n];
        float dec = decay[t * H + h];

        // Update adjoint state: dh += dy_pre * cp
        dh = __fmaf_rn(dy_pre, cp_tn, dh);

        // d_proj[x_skip] and d_d_param  (only n=0 thread writes per (h, p))
        if (n == 0) {
            float x_val = proj[t * dip + di + h * hd + p];
            atomicAdd(&d_proj[t * dip + di + h * hd + p],
                      dy_pre * d_param[h]);
            atomicAdd(&d_d_param[h], dy_pre * x_val);
        }

        // d_cp[t, n]: atomic across heads
        float state_tp1 = states[(((t + 1) * H + h) * hd + p) * ds + n];
        atomicAdd(&d_proj[t * dip + 2 * di + ds + n],
                  dy_pre * state_tp1);

        // d_scan_inp[t, h, p, n] = current dh
        d_scan_inp[((t * H + h) * hd + p) * ds + n] = dh;

        // Block-wide reduction: d_decay[t, h] = Σ dh · state[t, h, p, n]
        float state_t = states[((t * H + h) * hd + p) * ds + n];
        smem[tid] = dh * state_t;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }
        if (tid == 0) d_decay[t * H + h] = smem[0];
        __syncthreads();

        // Block-wide reduction: d_dt_from_inp[t, h] = Σ dh · blended
        // blended = inp_val / dt_v,  inp_val = states[t+1] - decay·states[t]
        float dt_v = dt_in[t * H + h];
        float inp_val = state_tp1 - dec * state_t;
        float blended = (dt_v > 1e-12f) ? (inp_val / dt_v) : 0.0f;
        smem[tid] = dh * blended;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }
        if (tid == 0) d_dt_from_inp[t * H + h] = smem[0];
        __syncthreads();

        // Propagate dh through state update
        dh *= dec;
    }
}

// ---------- ssm_param_grads -------------------------------------------------
// From d_decay[t, h] and d_dt_from_inp[t, h], compute:
//   d_dt[t, h]      = d_decay · a · decay + d_dt_from_inp     (scalar)
//   d_a[t, h]       = d_decay · dt · decay                     (scalar)
//   d_dd_dt[t, h]   = d_dt · sigmoid(dd_dt + dt_bias)          → d_proj[dt_off+h]
//   d_dd_a[t, h]    = -d_a · sigmoid(dd_a) if a > -1e-4 else 0 → d_proj[a_off+h]
//   d_dt_bias[h]   += Σ_t d_dt · sigmoid(dd_dt + dt_bias)     (atomic)
//
// Grid: (L,), Block: (max(H,32),)
extern "C" __global__ void ssm_param_grads(
    const float* __restrict__ proj,
    const float* __restrict__ dt_bias,
    const float* __restrict__ dt_in,
    const float* __restrict__ decay,
    const float* __restrict__ d_decay,
    const float* __restrict__ d_dt_from_inp,
    float* __restrict__ d_proj,
    float* __restrict__ d_dt_bias,
    int L, int H, int dip, int di, int ds
) {
    int t = blockIdx.x;
    int h = threadIdx.x;
    if (t >= L || h >= H) return;
    int dt_off = 2 * di + 2 * ds;
    int a_off = dt_off + H;
    // int tr_off = a_off + H; // not used here

    float dt_v = dt_in[t * H + h];
    float dec = decay[t * H + h];
    // Recompute a from proj (clamp-aware)
    float dd_a = proj[t * dip + a_off + h];
    float a_unclamped = -softplus_f(dd_a);
    float a = a_unclamped;
    bool clamped = false;
    if (a > -1e-4f) { a = -1e-4f; clamped = true; }

    float d_decay_th = d_decay[t * H + h];
    float d_dt_inp = d_dt_from_inp[t * H + h];

    // d_dt = d_decay · a · decay + d_dt_from_inp
    float d_dt = d_decay_th * a * dec + d_dt_inp;
    // d_a  = d_decay · dt · decay
    float d_a = d_decay_th * dt_v * dec;

    // dt = softplus(dd_dt + dt_bias[h]) → dt_prime = sigmoid(dd_dt + dt_bias)
    float dd_dt = proj[t * dip + dt_off + h];
    float sig_dt = sigmoid_f(dd_dt + dt_bias[h]);
    float d_dd_dt = d_dt * sig_dt;
    d_proj[t * dip + dt_off + h] = d_dd_dt;
    atomicAdd(&d_dt_bias[h], d_dd_dt);

    // a_raw = -softplus(dd_a). dsoftplus/d(dd_a) = sigmoid(dd_a).
    // d_dd_a = d_a_unclamped · (-sigmoid(dd_a)) when not clamped, else 0
    float d_dd_a = clamped ? 0.0f : (-d_a * sigmoid_f(dd_a));
    d_proj[t * dip + a_off + h] = d_dd_a;
}

// ---------- bx_bwd ----------------------------------------------------------
// From d_scan_inp, produce d_proj[bp_off] (atomic) and d_proj[x_off] (atomic).
// Per timestep/head/p: d_bx[n] = d_scan_inp[n] / (dt*trap + 1e-8);
//                      d_proj[bp_off + n] += d_bx[n] * x_raw[h*hd+p]
//                      d_proj[x_off + h*hd+p] += sum_n(d_bx[n] * bp_raw[n])
// Grid: (H,), Block: (hd*ds,) — same layout as ssm_scan_bwd.
extern "C" __global__ void bx_bwd(
    const float* __restrict__ d_scan_inp,
    const float* __restrict__ proj,
    const float* __restrict__ dt_bias,
    float* __restrict__ d_proj,
    int L, int H, int hd, int ds, int di, int dip
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= H || tid >= hd * ds) return;
    int p = tid / ds;
    int n = tid - p * ds;

    int dt_off = 2 * di + 2 * ds;
    int a_off = dt_off + H;
    int tr_off = a_off + H;
    int bp_off = 2 * di;

    extern __shared__ float smem[];
    // Tree-reduce buffer per (h, p) for d_x_val sum over n.
    // smem size = hd*ds

    for (int t = 0; t < L; t++) {
        float dt_raw = proj[t * dip + dt_off + h] + dt_bias[h];
        float dt_v = softplus_f(dt_raw);
        float tr_raw = proj[t * dip + tr_off + h];
        float tr = sigmoid_f(tr_raw);

        // Correct math: inp_val = blended * dt, blended ≈ trap * bx_cur,
        // so d_bx_cur = d_inp_val * dt * trap.  Previously divided.
        float d_bx = d_scan_inp[((t * H + h) * hd + p) * ds + n] * (dt_v * tr);
        float x_val = proj[t * dip + di + h * hd + p];
        float bp_raw = proj[t * dip + bp_off + n];

        // d_Bp[n] += d_bx * x_val   (x_val same for all n in thread's (h, p))
        atomicAdd(&d_proj[t * dip + bp_off + n], d_bx * x_val);

        // d_x[h*hd+p] partial: d_bx * bp_raw; sum over n in p-group
        smem[tid] = d_bx * bp_raw;
        __syncthreads();

        // Tree reduce across n for each p-group of ds threads
        for (int stride = ds >> 1; stride > 0; stride >>= 1) {
            if (n < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }

        if (n == 0) {
            atomicAdd(&d_proj[t * dip + di + h * hd + p], smem[p * ds]);
        }
        __syncthreads();
    }
}

// ---------- embed_scatter_bwd -----------------------------------------------
// For each token t: atomicAdd d_embed[token[t], :] += d_x[t, :]
// Grid: (L,), Block: (min(d, 256),)
extern "C" __global__ void embed_scatter_bwd(
    const float* __restrict__ d_x,
    const unsigned int* __restrict__ tokens,
    float* __restrict__ d_embed,
    int L, int d, int vocab
) {
    int t = blockIdx.x;
    if (t >= L) return;
    unsigned int tok = tokens[t];
    if (tok >= (unsigned int)vocab) return;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        atomicAdd(&d_embed[tok * d + i], d_x[t * d + i]);
    }
}


// ---------- adamw_step ------------------------------------------------------
// Elementwise AdamW update. Matches mamba3_engine::backward::adamw_step exactly.
//   p *= 1 - lr*wd
//   m = β1·m + (1-β1)·g
//   v = β2·v + (1-β2)·g²
//   p -= lr · (m/bc1) / (sqrt(v/bc2) + eps)
// bc1_inv, bc2_inv precomputed on host as 1/(1-β^step).
// Grid: (ceil(n/256),), Block: (256,)
extern "C" __global__ void adamw_step(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1_inv, float bc2_inv,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grads[i];
    // Simple per-element clipping: caps runaway gradients and NaN propagation
    // that can arise from newly-enabled gradient paths (e.g. d_dt_bias when
    // dt is small, making inp/dt blow up). Matches PyTorch's common
    // clip_grad_norm behavior in spirit.
    if (!isfinite(g)) g = 0.0f;
    if (g > 1.0f)  g = 1.0f;
    if (g < -1.0f) g = -1.0f;
    float p = params[i] * (1.0f - lr * wd);
    float mi = __fmaf_rn(1.0f - beta1, g, beta1 * m[i]);
    float vi = __fmaf_rn(1.0f - beta2, g * g, beta2 * v[i]);
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi * bc1_inv;
    float v_hat = vi * bc2_inv;
    params[i] = p - lr * m_hat / (sqrtf(v_hat) + eps);
}

// ---------- cross_entropy_fwd_bwd ------------------------------------------
// Per-row softmax with max-subtraction for numerical stability.  Writes:
//   loss_out[0] += -log(softmax[target]) / L  (atomic across L rows)
//   d_logits[t, v] = (softmax(t, v) - [v==target]) / L
// Matches mamba3_engine::backward::cross_entropy_loss.
// Grid: (L,), Block: (256,) — one block per timestep
extern "C" __global__ void cross_entropy_fwd_bwd(
    const float* __restrict__ logits,     // (L, V)
    const unsigned int* __restrict__ targets,  // (L,)
    float* __restrict__ d_logits,         // (L, V)
    float* __restrict__ loss_out,         // (1,) accumulator — caller must zero
    int L, int V
) {
    int t = blockIdx.x;
    if (t >= L) return;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    unsigned int target = targets[t];

    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float warp_buf[8];

    // --- find row max ---
    float my_max = -3.4028235e38f;
    for (int v = tid; v < V; v += blockDim.x) {
        my_max = fmaxf(my_max, logits[t * V + v]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, my_max, off);
        my_max = fmaxf(my_max, other);
    }
    if (lane == 0) warp_buf[warp] = my_max;
    __syncthreads();
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        my_max = (lane < nw) ? warp_buf[lane] : -3.4028235e38f;
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, my_max, off);
            my_max = fmaxf(my_max, other);
        }
        if (lane == 0) s_max = my_max;
    }
    __syncthreads();
    float row_max = s_max;

    // --- compute exp, sum, store unnormalized into d_logits ---
    float my_sum = 0.0f;
    for (int v = tid; v < V; v += blockDim.x) {
        float e = __expf(logits[t * V + v] - row_max);
        d_logits[t * V + v] = e;
        my_sum += e;
    }
    my_sum = warp_reduce_sum(my_sum);
    if (lane == 0) warp_buf[warp] = my_sum;
    __syncthreads();
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        my_sum = (lane < nw) ? warp_buf[lane] : 0.0f;
        my_sum = warp_reduce_sum(my_sum);
        if (lane == 0) s_sum = my_sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / s_sum;
    float inv_L = 1.0f / (float)L;

    // --- normalize to softmax, subtract one-hot, scale by 1/L ---
    for (int v = tid; v < V; v += blockDim.x) {
        float sm = d_logits[t * V + v] * inv_sum;
        float g = sm - ((unsigned int)v == target ? 1.0f : 0.0f);
        d_logits[t * V + v] = g * inv_L;
    }

    // --- loss contribution for this row ---
    if (tid == 0) {
        // pred = exp(logits[target] - max) * inv_sum   (avoids re-reading d_logits after write)
        float pred = __expf(logits[t * V + target] - row_max) * inv_sum;
        float contrib = -__logf(pred) * inv_L;
        atomicAdd(loss_out, contrib);
    }
}

// ---------- ssm_scan_cached ------------------------------------------------
// Same forward math as ssm_scan_sequential, but ALSO writes the full state
// sequence to `states[(t+1)*H*hd*ds + h*hd*ds + p*ds + n]`.  states[0] is
// zero-initialized by the host; each timestep writes states[t+1].
// This is the cache the backward adjoint scan reads.
extern "C" __global__ void ssm_scan_cached(
    const float* __restrict__ proj,
    const float* __restrict__ bp,
    const float* __restrict__ cp,
    const float* __restrict__ dt_in,
    const float* __restrict__ decay_in,
    const float* __restrict__ trap_in,
    const float* __restrict__ d_param,
    float* __restrict__ y,
    float* __restrict__ states,  // (L+1, H, hd, ds); states[0] = 0
    int L, int H, int hd, int ds, int di, int dip
) {
    extern __shared__ float smem[];
    float* y_reduce = &smem[hd * ds];

    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= H || tid >= hd * ds) return;
    int p = tid / ds;
    int n = tid % ds;

    int x_off = di;

    float state = 0.0f;
    float bx_prev = 0.0f;

    for (int t = 0; t < L; t++) {
        float bp_tn = bp[t * ds + n];
        float cp_tn = cp[t * ds + n];
        float dt_v = dt_in[t * H + h];
        float dec = decay_in[t * H + h];
        float tr = trap_in[t * H + h];
        float x_val = proj[t * dip + x_off + h * hd + p];

        float bx_cur = x_val * bp_tn;
        float blended = __fmaf_rn(tr, bx_cur, (1.0f - tr) * bx_prev);
        float inp_val = blended * dt_v;
        bx_prev = bx_cur;
        state = __fmaf_rn(dec, state, inp_val);

        // Write state for backward: states[t+1, h, p, n]
        states[(((t + 1) * H + h) * hd + p) * ds + n] = state;

        smem[tid] = state * cp_tn;
        __syncthreads();

        for (int stride = ds >> 1; stride > 0; stride >>= 1) {
            if (n < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }

        if (n == 0) {
            float sum = smem[p * ds];
            sum = __fmaf_rn(d_param[h], x_val, sum);
            float z_raw = proj[t * dip + h * hd + p];
            float z_silu = z_raw * sigmoid_f(z_raw);
            y_reduce[p] = sum * z_silu;
        }
        __syncthreads();

        if (tid < hd) {
            y[(t * H + h) * hd + tid] = y_reduce[tid];
        }
        __syncthreads();
    }
}

// ---------- residual_add ----------------------------------------------------
// x[i] = x[i] + scale * y[i]
// Grid: (ceil(n/256),), Block: (256,)
extern "C" __global__ void residual_add(
    float* __restrict__ x,
    const float* __restrict__ y,
    float scale,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __fmaf_rn(scale, y[i], x[i]);
}
