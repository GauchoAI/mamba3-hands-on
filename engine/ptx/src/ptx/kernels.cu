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
