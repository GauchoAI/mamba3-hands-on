// Mamba-3 kernels in CUDA C — compiled to PTX at runtime via NVRTC with strict
// FP32 flags (no fast-math, no contraction surprises). All arithmetic uses
// explicit __fmaf_rn (IEEE round-to-nearest FMA) to match Rust f32::mul_add.
//
// Generated PTX can be inspected via cudarc's module dump (see runtime.rs).
//
// Design is v1 (correctness-first): one kernel per op, many dispatches per
// layer. v2+ fuses ambitiously.

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
extern "C" __global__ void ssm_scan_sequential(
    const float* __restrict__ proj,
    const float* __restrict__ bp,
    const float* __restrict__ cp,
    const float* __restrict__ dt_in,
    const float* __restrict__ decay_in,
    const float* __restrict__ trap_in,
    const float* __restrict__ d_param,
    const float* __restrict__ z_silu,
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
            float z = z_silu[t * di + h * hd + p];
            y_reduce[p] = sum * z;
        }
        __syncthreads();

        if (tid < hd) {
            // Write: y[t, h, p]  layout (L, H, hd)
            y[(t * H + h) * hd + tid] = y_reduce[tid];
        }
        __syncthreads();
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
