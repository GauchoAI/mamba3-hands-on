//! Full Mamba-3 model — forward pass in pure Rust.
//! Matches mamba3_minimal.py + progressive_model.py exactly.
//! All fp32, explicit arithmetic, no framework dependency.

use std::path::Path;

pub struct Mamba3Model {
    pub d_model: usize,
    pub d_state: usize,
    pub d_inner: usize,
    pub headdim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub vocab_size: usize,

    pub embed_w: Vec<f32>,           // (vocab, d_model)
    pub embed_norm_w: Vec<f32>,      // (d_model,)
    pub embed_norm_b: Vec<f32>,      // (d_model,)
    pub layers: Vec<LayerWeights>,
    pub final_norm_w: Vec<f32>,      // (d_model,)
    pub final_norm_b: Vec<f32>,      // (d_model,)
    // head = embed_w (weight-tied)
}

pub struct LayerWeights {
    pub in_proj_w: Vec<f32>,         // (d_in_proj, d_model)
    pub d_in_proj: usize,
    pub out_proj_w: Vec<f32>,        // (d_model, d_inner)
    pub dt_bias: Vec<f32>,           // (n_heads,)
    pub d_param: Vec<f32>,           // (n_heads,)
    pub b_norm_w: Vec<f32>,          // (d_state,)
    pub b_norm_b: Vec<f32>,          // (d_state,)
    pub c_norm_w: Vec<f32>,          // (d_state,)
    pub c_norm_b: Vec<f32>,          // (d_state,)
    pub layer_norm_w: Vec<f32>,      // (d_model,)
    pub layer_norm_b: Vec<f32>,      // (d_model,)
    pub scale: f32,
    pub num_rope_angles: usize,
}

impl Mamba3Model {
    /// Create a new model with random initialization
    pub fn new_random(d_model: usize, d_state: usize, headdim: usize, n_layers: usize, vocab_size: usize) -> Self {
        let d_inner = d_model * 2;
        let n_heads = d_inner / headdim;

        // Xavier/Glorot initialization
        let scale = |fan_in: usize, fan_out: usize| -> f32 {
            (6.0 / (fan_in + fan_out) as f32).sqrt()
        };

        let rand_vec = |n: usize, s: f32| -> Vec<f32> {
            // Simple LCG PRNG — deterministic, no dependency
            use std::cell::Cell;
            thread_local! { static SEED: Cell<u64> = Cell::new(42); }
            (0..n).map(|_| {
                SEED.with(|seed| {
                    let s_val = seed.get();
                    let next = s_val.wrapping_mul(6364136223846793005).wrapping_add(1);
                    seed.set(next);
                    let bits = ((next >> 33) as u32) as f32 / u32::MAX as f32;
                    (bits * 2.0 - 1.0) * s
                })
            }).collect()
        };

        let embed_w = rand_vec(vocab_size * d_model, scale(vocab_size, d_model));
        let embed_norm_w = vec![1.0f32; d_model];
        let embed_norm_b = vec![0.0f32; d_model];

        let mut layers = Vec::new();
        for _ in 0..n_layers {
            let num_rope_angles = d_state / 2;
            let d_in_proj = 2 * d_inner + 2 * d_state + 3 * n_heads + num_rope_angles;
            layers.push(LayerWeights {
                in_proj_w: rand_vec(d_in_proj * d_model, scale(d_model, d_in_proj)),
                d_in_proj,
                out_proj_w: rand_vec(d_model * d_inner, scale(d_inner, d_model) * 0.01), // near-zero init
                dt_bias: vec![-3.0f32; n_heads], // softplus(-3) ≈ 0.05
                d_param: vec![1.0f32; n_heads],
                b_norm_w: vec![1.0f32; d_state],
                b_norm_b: vec![0.0f32; d_state],
                c_norm_w: vec![1.0f32; d_state],
                c_norm_b: vec![0.0f32; d_state],
                layer_norm_w: vec![1.0f32; d_model],
                layer_norm_b: vec![0.0f32; d_model],
                scale: 0.01,
                num_rope_angles,
            });
        }

        let final_norm_w = vec![1.0f32; d_model];
        let final_norm_b = vec![0.0f32; d_model];

        Self {
            d_model, d_state, d_inner, headdim, n_heads, n_layers, vocab_size,
            embed_w, embed_norm_w, embed_norm_b, layers, final_norm_w, final_norm_b,
        }
    }

    /// Load from exported binary (see export/rust_export.py)
    pub fn from_bin(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let header: &[u32] = bytemuck::cast_slice(&data[..28]);
        let d_model = header[0] as usize;
        let d_state = header[1] as usize;
        let headdim = header[2] as usize;
        let n_layers = header[3] as usize;
        let vocab_size = header[4] as usize;
        let d_inner = d_model * 2;
        let n_heads = d_inner / headdim;

        let floats: &[f32] = bytemuck::cast_slice(&data[28..]);
        let mut off = 0usize;

        fn read_slice(floats: &[f32], off: &mut usize, n: usize) -> Vec<f32> {
            let v = floats[*off..*off + n].to_vec();
            *off += n;
            v
        }

        let embed_w = read_slice(floats, &mut off, vocab_size * d_model);
        let embed_norm_w = read_slice(floats, &mut off, d_model);
        let embed_norm_b = read_slice(floats, &mut off, d_model);

        let mut layers = Vec::new();
        for _ in 0..n_layers {
            let num_rope_angles = d_state / 2;
            let d_in_proj = 2 * d_inner + 2 * d_state + 3 * n_heads + num_rope_angles;
            let in_proj_w = read_slice(floats, &mut off, d_in_proj * d_model);
            let out_proj_w = read_slice(floats, &mut off, d_model * d_inner);
            let dt_bias = read_slice(floats, &mut off, n_heads);
            let d_param = read_slice(floats, &mut off, n_heads);
            let b_norm_w = read_slice(floats, &mut off, d_state);
            let b_norm_b = read_slice(floats, &mut off, d_state);
            let c_norm_w = read_slice(floats, &mut off, d_state);
            let c_norm_b = read_slice(floats, &mut off, d_state);
            let layer_norm_w = read_slice(floats, &mut off, d_model);
            let layer_norm_b = read_slice(floats, &mut off, d_model);
            let scale = read_slice(floats, &mut off, 1)[0];
            layers.push(LayerWeights {
                in_proj_w, d_in_proj, out_proj_w, dt_bias, d_param,
                b_norm_w, b_norm_b, c_norm_w, c_norm_b,
                layer_norm_w, layer_norm_b, scale, num_rope_angles,
            });
        }

        let final_norm_w = read_slice(floats, &mut off, d_model);
        let final_norm_b = read_slice(floats, &mut off, d_model);

        Ok(Self {
            d_model, d_state, d_inner, headdim, n_heads, n_layers, vocab_size,
            embed_w, embed_norm_w, embed_norm_b, layers, final_norm_w, final_norm_b,
        })
    }

    /// Forward pass: tokens → logits. Single sequence (batch=1).
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let l = tokens.len();
        let d = self.d_model;

        // 1. Embedding + norm
        let mut x = vec![0.0f32; l * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok < self.vocab_size {
                for i in 0..d {
                    x[t * d + i] = self.embed_w[tok * d + i];
                }
            }
        }
        layer_norm(&mut x, &self.embed_norm_w, &self.embed_norm_b, l, d);

        // 2. SSM layers: x = x + scale * block(norm(x))
        for layer in &self.layers {
            // Pre-norm BEFORE the block (PyTorch: layer["block"](layer["norm"](x)))
            let mut x_normed = x.clone();
            layer_norm(&mut x_normed, &layer.layer_norm_w, &layer.layer_norm_b, l, d);
            let y = self.mamba3_block(&x_normed, layer, l);
            // Residual + scale
            for i in 0..l * d {
                x[i] += layer.scale * y[i];
            }
        }

        // 3. Final norm
        layer_norm(&mut x, &self.final_norm_w, &self.final_norm_b, l, d);

        // 4. LM head (weight-tied with embed)
        let mut logits = vec![0.0f32; l * self.vocab_size];
        for t in 0..l {
            for v in 0..self.vocab_size {
                let mut s = 0.0f32;
                for i in 0..d {
                    s += x[t * d + i] * self.embed_w[v * d + i];
                }
                logits[t * self.vocab_size + v] = s;
            }
        }
        logits
    }

    fn mamba3_block(&self, u: &[f32], lw: &LayerWeights, l: usize) -> Vec<f32> {
        let d = self.d_model;
        let di = self.d_inner;

        // In-projection
        let dip = lw.d_in_proj;
        let mut proj = vec![0.0f32; l * dip];
        matmul_t(&mut proj, u, &lw.in_proj_w, l, d, dip);

        // SSM + out_proj
        let y_inner = self.mamba3_block_inner(&proj, lw, l);

        // Out-projection
        let mut out = vec![0.0f32; l * d];
        matmul_t(&mut out, &y_inner, &lw.out_proj_w, l, di, d);
        out
    }

    /// Inner SSM computation: proj → y_inner (without in_proj and out_proj matmuls)
    pub fn mamba3_block_inner_pub(&self, proj: &[f32], lw: &LayerWeights, l: usize) -> Vec<f32> {
        self.mamba3_block_inner(proj, lw, l)
    }

    /// Extract scan inputs from projection — for GPU scan dispatch.
    /// Returns (inp, decay, Cp, x_skip, z_silu, D) flattened for ssm_scan shader.
    pub fn extract_scan_inputs(&self, proj: &[f32], lw: &LayerWeights, l: usize)
        -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)
    {
        let di = self.d_inner;
        let nh = self.n_heads;
        let hd = self.headdim;
        let ds = self.d_state;
        let dip = lw.d_in_proj;

        // Split projection
        let mut off = 0;
        let z_raw = proj_slice(proj, l, dip, off, di); off += di;
        let x_raw = proj_slice(proj, l, dip, off, di); off += di;
        let bp_raw = proj_slice(proj, l, dip, off, ds); off += ds;
        let cp_raw = proj_slice(proj, l, dip, off, ds); off += ds;
        let dd_dt = proj_slice(proj, l, dip, off, nh); off += nh;
        let dd_a = proj_slice(proj, l, dip, off, nh); off += nh;
        let trap_raw = proj_slice(proj, l, dip, off, nh); off += nh;
        let angles = proj_slice(proj, l, dip, off, lw.num_rope_angles);

        // Norm B, C
        let mut bp = bp_raw;
        layer_norm(&mut bp, &lw.b_norm_w, &lw.b_norm_b, l, ds);
        let mut cp = cp_raw;
        layer_norm(&mut cp, &lw.c_norm_w, &lw.c_norm_b, l, ds);

        // DT, decay, trap
        let mut dt = vec![0.0f32; l * nh];
        let mut decay = vec![0.0f32; l * nh];
        let mut trap = vec![0.0f32; l * nh];
        for t in 0..l {
            for h in 0..nh {
                dt[t * nh + h] = softplus(dd_dt[t * nh + h] + lw.dt_bias[h]);
                let a_raw = -softplus(dd_a[t * nh + h]);
                let a = if a_raw > -1e-4 { -1e-4 } else { a_raw };
                decay[t * nh + h] = (a * dt[t * nh + h]).exp();
                trap[t * nh + h] = sigmoid(trap_raw[t * nh + h]);
            }
        }

        // RoPE
        let n_angles = lw.num_rope_angles;
        let mut dt_mean = vec![0.0f32; l];
        for t in 0..l {
            let mut s = 0.0f32;
            for h in 0..nh { s += dt[t * nh + h]; }
            dt_mean[t] = s / nh as f32;
        }
        let mut phase = vec![0.0f32; l * n_angles];
        let mut cumphase = vec![0.0f32; n_angles];
        for t in 0..l {
            for k in 0..n_angles {
                cumphase[k] += angles[t * n_angles + k] * dt_mean[t];
                phase[t * n_angles + k] = cumphase[k];
            }
        }
        apply_rope(&mut bp, &phase, l, ds, n_angles);
        apply_rope(&mut cp, &phase, l, ds, n_angles);

        // Compute inp with trapezoidal: (L, H, hD, dS)
        let mut inp = vec![0.0f32; l * nh * hd * ds];
        let mut bx_prev = vec![0.0f32; nh * hd * ds];
        for t in 0..l {
            for h in 0..nh {
                let mut bx_cur = vec![0.0f32; hd * ds];
                for p in 0..hd {
                    let xv = x_raw[t * di + h * hd + p];
                    for n in 0..ds { bx_cur[p * ds + n] = xv * bp[t * ds + n]; }
                }
                let tr = trap[t * nh + h];
                let dtv = dt[t * nh + h];
                for p in 0..hd {
                    for n in 0..ds {
                        let idx = ((t * nh + h) * hd + p) * ds + n;
                        inp[idx] = (tr * bx_cur[p * ds + n] + (1.0 - tr) * bx_prev[h * hd * ds + p * ds + n]) * dtv;
                    }
                }
                for i in 0..hd * ds { bx_prev[h * hd * ds + i] = bx_cur[i]; }
            }
        }

        // Reshape for ssm_scan: needs (B=1, L, H, dS) for Cp
        // Broadcast Cp across heads
        let mut cp_broadcast = vec![0.0f32; l * nh * ds];
        for t in 0..l {
            for h in 0..nh {
                for n in 0..ds { cp_broadcast[(t * nh + h) * ds + n] = cp[t * ds + n]; }
            }
        }

        // x reshaped to (L, H, hD)
        let mut x_skip = vec![0.0f32; l * nh * hd];
        for t in 0..l {
            for h in 0..nh {
                for p in 0..hd { x_skip[(t * nh + h) * hd + p] = x_raw[t * di + h * hd + p]; }
            }
        }

        // z_silu reshaped to (L, H, hD) — precomputed silu gate
        let mut z_reshaped = vec![0.0f32; l * nh * hd];
        for t in 0..l {
            for h in 0..nh {
                for p in 0..hd {
                    let z = z_raw[t * di + h * hd + p];
                    z_reshaped[(t * nh + h) * hd + p] = z * sigmoid(z);
                }
            }
        }

        // D param
        let d_param = lw.d_param.clone();

        (inp, decay, cp_broadcast, x_skip, z_reshaped, d_param)
    }
    fn mamba3_block_inner(&self, proj: &[f32], lw: &LayerWeights, l: usize) -> Vec<f32> {
        let di = self.d_inner;
        let nh = self.n_heads;
        let hd = self.headdim;
        let ds = self.d_state;
        let dip = lw.d_in_proj;

        // Split: z, x, Bp, Cp, dd_dt, dd_A, trap_raw, angles
        let mut off = 0;
        let z_raw = &proj_slice(&proj, l, dip, off, di); off += di;
        let x_raw = &proj_slice(&proj, l, dip, off, di); off += di;
        let bp_raw = &proj_slice(&proj, l, dip, off, ds); off += ds;
        let cp_raw = &proj_slice(&proj, l, dip, off, ds); off += ds;
        let dd_dt = &proj_slice(&proj, l, dip, off, nh); off += nh;
        let dd_a = &proj_slice(&proj, l, dip, off, nh); off += nh;
        let trap_raw = &proj_slice(&proj, l, dip, off, nh); off += nh;
        let angles = &proj_slice(&proj, l, dip, off, lw.num_rope_angles);

        // Layer-norm B and C
        let mut bp = bp_raw.clone();
        layer_norm(&mut bp, &lw.b_norm_w, &lw.b_norm_b, l, ds);
        let mut cp = cp_raw.clone();
        layer_norm(&mut cp, &lw.c_norm_w, &lw.c_norm_b, l, ds);

        // DT = softplus(dd_dt + dt_bias)
        let mut dt = vec![0.0f32; l * nh];
        for t in 0..l {
            for h in 0..nh {
                let v = dd_dt[t * nh + h] + lw.dt_bias[h];
                dt[t * nh + h] = softplus(v);
            }
        }

        // A = -softplus(dd_A), decay = exp(A * DT)
        let mut decay = vec![0.0f32; l * nh];
        for t in 0..l {
            for h in 0..nh {
                // A = (-softplus(dd_A)).clamp(max=-A_floor) — A is always negative
                let a_raw = -softplus(dd_a[t * nh + h]);
                let a = if a_raw > -1e-4 { -1e-4 } else { a_raw }; // clamp to ≤ -A_floor
                decay[t * nh + h] = (a * dt[t * nh + h]).exp();
            }
        }

        // RoPE on B and C
        let n_angles = lw.num_rope_angles;
        let mut dt_mean = vec![0.0f32; l];
        for t in 0..l {
            let mut s = 0.0f32;
            for h in 0..nh { s += dt[t * nh + h]; }
            dt_mean[t] = s / nh as f32;
        }
        let mut phase = vec![0.0f32; l * n_angles];
        let mut cumphase = vec![0.0f32; n_angles];
        for t in 0..l {
            for k in 0..n_angles {
                cumphase[k] += angles[t * n_angles + k] * dt_mean[t];
                phase[t * n_angles + k] = cumphase[k];
            }
        }
        apply_rope(&mut bp, &phase, l, ds, n_angles);
        apply_rope(&mut cp, &phase, l, ds, n_angles);

        // Trap = sigmoid(trap_raw)
        let mut trap = vec![0.0f32; l * nh];
        for t in 0..l {
            for h in 0..nh {
                trap[t * nh + h] = sigmoid(trap_raw[t * nh + h]);
            }
        }

        // Reshape x to (L, H, hD) and compute Bx = outer(x, Bp): (L, H, hD, dS)
        // Then inp = trap * Bx + (1-trap) * Bx_prev, scaled by dt
        let mut inp = vec![0.0f32; l * nh * hd * ds];
        let mut bx_prev = vec![0.0f32; nh * hd * ds]; // zero for t=0

        for t in 0..l {
            for h in 0..nh {
                // Current Bx
                let mut bx_cur = vec![0.0f32; hd * ds];
                for p in 0..hd {
                    let x_val = x_raw[t * di + h * hd + p];
                    for n in 0..ds {
                        bx_cur[p * ds + n] = x_val * bp[t * ds + n]; // Bp broadcast across heads
                    }
                }

                let tr = trap[t * nh + h];
                let dt_val = dt[t * nh + h];

                for p in 0..hd {
                    for n in 0..ds {
                        let idx = ((t * nh + h) * hd + p) * ds + n;
                        let cur = bx_cur[p * ds + n];
                        let prev = bx_prev[h * hd * ds + p * ds + n];
                        inp[idx] = (tr * cur + (1.0 - tr) * prev) * dt_val;
                    }
                }

                // Save current as prev for next timestep
                for i in 0..hd * ds {
                    bx_prev[h * hd * ds + i] = bx_cur[i];
                }
            }
        }

        // SSM scan
        let mut state = vec![0.0f32; nh * hd * ds];
        let mut y = vec![0.0f32; l * nh * hd];

        // Precompute silu(z) — OUTSIDE the scan loop
        let mut z_silu = vec![0.0f32; l * di];
        for i in 0..l * di {
            let z = z_raw[i];
            z_silu[i] = z * sigmoid(z);
        }

        for t in 0..l {
            for h in 0..nh {
                let dec = decay[t * nh + h];

                // State update — contiguous memory access, 4-wide
                let state_base = h * hd * ds;
                let inp_base = ((t * nh + h) * hd) * ds;
                for pd in 0..hd * ds {
                    let si = state_base + pd;
                    state[si] = dec * state[si] + inp[inp_base + pd];
                }

                // Output projection + skip + gate — with unrolled dot product
                let cp_slice = &cp[t * ds..(t + 1) * ds];
                for p in 0..hd {
                    let s_off = state_base + p * ds;
                    let mut s0 = 0.0f32;
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    let mut s3 = 0.0f32;
                    let ds4 = ds / 4 * 4;
                    let mut n = 0;
                    while n < ds4 {
                        s0 += state[s_off + n] * cp_slice[n];
                        s1 += state[s_off + n + 1] * cp_slice[n + 1];
                        s2 += state[s_off + n + 2] * cp_slice[n + 2];
                        s3 += state[s_off + n + 3] * cp_slice[n + 3];
                        n += 4;
                    }
                    let mut sum = s0 + s1 + s2 + s3;
                    while n < ds { sum += state[s_off + n] * cp_slice[n]; n += 1; }
                    // Skip
                    sum += lw.d_param[h] * x_raw[t * di + h * hd + p];
                    // Gate
                    sum *= z_silu[t * di + h * hd + p];
                    y[(t * nh + h) * hd + p] = sum;
                }
            }
        }

        y
    }

    /// Forward pass using GPU for matmuls. Significantly faster on GPU hardware.
    pub fn forward_gpu(&self, tokens: &[u32], gpu: &crate::scan::GpuContext) -> Vec<f32> {
        let l = tokens.len();
        let d = self.d_model;

        // 1. Embedding + norm (CPU — small, not worth GPU transfer)
        let mut x = vec![0.0f32; l * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok < self.vocab_size {
                for i in 0..d { x[t * d + i] = self.embed_w[tok * d + i]; }
            }
        }
        layer_norm(&mut x, &self.embed_norm_w, &self.embed_norm_b, l, d);

        // 2. SSM layers — GPU matmuls + CPU scan
        for layer in &self.layers {
            let mut x_normed = x.clone();
            layer_norm(&mut x_normed, &layer.layer_norm_w, &layer.layer_norm_b, l, d);

            // GPU in_proj: (l, d) × (d_in_proj, d)^T → (l, d_in_proj)
            let dip = layer.d_in_proj;
            let proj = pollster::block_on(gpu.run_matmul(
                &x_normed, &layer.in_proj_w, l as u32, d as u32, dip as u32
            )).unwrap_or_else(|_| {
                let mut out = vec![0.0f32; l * dip];
                matmul_t(&mut out, &x_normed, &layer.in_proj_w, l, d, dip);
                out
            });

            // SSM (CPU — sequential, can't parallelize across time)
            let y_inner = self.run_ssm_from_proj_cpu(&proj, layer, l);

            // GPU out_proj: (l, d_inner) × (d, d_inner)^T → (l, d)
            let di = self.d_inner;
            let y_out = pollster::block_on(gpu.run_matmul(
                &y_inner, &layer.out_proj_w, l as u32, di as u32, d as u32
            )).unwrap_or_else(|_| {
                let mut out = vec![0.0f32; l * d];
                matmul_t(&mut out, &y_inner, &layer.out_proj_w, l, di, d);
                out
            });

            for i in 0..l * d { x[i] += layer.scale * y_out[i]; }
        }

        // 3. Final norm (CPU)
        layer_norm(&mut x, &self.final_norm_w, &self.final_norm_b, l, d);

        // 4. GPU LM head: (l, d) × (vocab, d)^T → (l, vocab)
        let logits = pollster::block_on(gpu.run_matmul(
            &x, &self.embed_w, l as u32, d as u32, self.vocab_size as u32
        )).unwrap_or_else(|_| {
            let mut out = vec![0.0f32; l * self.vocab_size];
            matmul_t(&mut out, &x, &self.embed_w, l, d, self.vocab_size);
            out
        });

        logits
    }

    /// SSM computation from projection (CPU only, used by forward_gpu)
    fn run_ssm_from_proj_cpu(&self, proj: &[f32], lw: &LayerWeights, l: usize) -> Vec<f32> {
        self.mamba3_block_inner(proj, lw, l)
    }

    /// Predict: run forward, return argmax token at each position
    pub fn predict(&self, tokens: &[u32]) -> Vec<u32> {
        let logits = self.forward(tokens);
        let l = tokens.len();
        let v = self.vocab_size;
        let mut preds = vec![0u32; l];
        for t in 0..l {
            let mut best = f32::NEG_INFINITY;
            let mut best_idx = 0u32;
            for i in 0..v {
                if logits[t * v + i] > best {
                    best = logits[t * v + i];
                    best_idx = i as u32;
                }
            }
            preds[t] = best_idx;
        }
        preds
    }
}

// ── Helpers ──────────────────────────────────────────────

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn softplus(x: f32) -> f32 { (1.0 + x.exp()).ln() }

/// SIMD dot product — uses platform intrinsics
#[inline]
pub fn dot_simd(a: &[f32], b: &[f32], k: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let k16 = k / 16 * 16;
        let mut sum = unsafe { vdupq_n_f32(0.0) };
        let mut p = 0;
        while p < k16 {
            unsafe {
                sum = vfmaq_f32(sum, vld1q_f32(a.as_ptr().add(p)), vld1q_f32(b.as_ptr().add(p)));
                sum = vfmaq_f32(sum, vld1q_f32(a.as_ptr().add(p+4)), vld1q_f32(b.as_ptr().add(p+4)));
                sum = vfmaq_f32(sum, vld1q_f32(a.as_ptr().add(p+8)), vld1q_f32(b.as_ptr().add(p+8)));
                sum = vfmaq_f32(sum, vld1q_f32(a.as_ptr().add(p+12)), vld1q_f32(b.as_ptr().add(p+12)));
            }
            p += 16;
        }
        let mut s = unsafe { vaddvq_f32(sum) };
        while p < k { s += a[p] * b[p]; p += 1; }
        s
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let k4 = k / 4 * 4;
        let mut p = 0;
        while p < k4 {
            s0 += a[p] * b[p] + a[p+2] * b[p+2];
            s1 += a[p+1] * b[p+1] + a[p+3] * b[p+3];
            p += 4;
        }
        let mut s = s0 + s1;
        while p < k { s += a[p] * b[p]; p += 1; }
        s
    }
}

fn apply_rope(v: &mut [f32], angles: &[f32], l: usize, s: usize, n: usize) {
    for t in 0..l {
        for k in 0..n {
            let a = angles[t * n + k];
            let c = a.cos();
            let s_val = a.sin();
            let even = v[t * s + 2 * k];
            let odd = v[t * s + 2 * k + 1];
            v[t * s + 2 * k] = even * c - odd * s_val;
            v[t * s + 2 * k + 1] = even * s_val + odd * c;
        }
    }
}

pub fn layer_norm_pub(x: &mut [f32], w: &[f32], b: &[f32], seq_len: usize, d: usize) {
    layer_norm(x, w, b, seq_len, d);
}
fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], seq_len: usize, d: usize) {
    let eps = 1e-5f32;
    for t in 0..seq_len {
        let off = t * d;
        let mean: f32 = (0..d).map(|i| x[off + i]).sum::<f32>() / d as f32;
        let var: f32 = (0..d).map(|i| (x[off + i] - mean).powi(2)).sum::<f32>() / d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            x[off + i] = (x[off + i] - mean) * inv_std * w[i] + b[i];
        }
    }
}

/// Matrix multiply: out = a × b^T. a is (m, k), b is (n, k), out is (m, n).
/// Multithreaded via rayon for large matrices, SIMD for inner loops.
/// Matmul dispatcher — uses persisted calibration to choose strategy.
pub fn matmul_t_pub(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    let profile = crate::calibrate::get_profile();
    let ops = m * n * k;

    if ops >= profile.parallel_threshold {
        use rayon::prelude::*;
        out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let a_row = &a[i * k..(i + 1) * k];
            for j in 0..n {
                row[j] = dot_simd(a_row, &b[j * k..(j + 1) * k], k);
            }
        });
    } else {
        matmul_t(out, a, b, m, k, n);
    }
}


fn matmul_t(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        for i in 0..m {
            let a_row = &a[i * k..];
            for j in 0..n {
                let b_row = &b[j * k..];
                let k16 = k / 16 * 16;
                let mut sum = unsafe { vdupq_n_f32(0.0) };
                let mut p = 0;
                while p < k16 {
                    unsafe {
                        let a0 = vld1q_f32(a_row.as_ptr().add(p));
                        let b0 = vld1q_f32(b_row.as_ptr().add(p));
                        sum = vfmaq_f32(sum, a0, b0);
                        let a1 = vld1q_f32(a_row.as_ptr().add(p + 4));
                        let b1 = vld1q_f32(b_row.as_ptr().add(p + 4));
                        sum = vfmaq_f32(sum, a1, b1);
                        let a2 = vld1q_f32(a_row.as_ptr().add(p + 8));
                        let b2 = vld1q_f32(b_row.as_ptr().add(p + 8));
                        sum = vfmaq_f32(sum, a2, b2);
                        let a3 = vld1q_f32(a_row.as_ptr().add(p + 12));
                        let b3 = vld1q_f32(b_row.as_ptr().add(p + 12));
                        sum = vfmaq_f32(sum, a3, b3);
                    }
                    p += 16;
                }
                let mut s = unsafe { vaddvq_f32(sum) };
                while p < k {
                    s += a_row[p] * b_row[p];
                    p += 1;
                }
                out[i * n + j] = s;
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        {
            use std::arch::x86_64::*;
            for i in 0..m {
                let a_row = &a[i * k..];
                for j in 0..n {
                    let b_row = &b[j * k..];
                    let k32 = k / 32 * 32;
                    let mut sum0 = unsafe { _mm256_setzero_ps() };
                    let mut sum1 = unsafe { _mm256_setzero_ps() };
                    let mut sum2 = unsafe { _mm256_setzero_ps() };
                    let mut sum3 = unsafe { _mm256_setzero_ps() };
                    let mut p = 0;
                    while p < k32 {
                        unsafe {
                            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row.as_ptr().add(p)), _mm256_loadu_ps(b_row.as_ptr().add(p)), sum0);
                            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row.as_ptr().add(p+8)), _mm256_loadu_ps(b_row.as_ptr().add(p+8)), sum1);
                            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row.as_ptr().add(p+16)), _mm256_loadu_ps(b_row.as_ptr().add(p+16)), sum2);
                            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row.as_ptr().add(p+24)), _mm256_loadu_ps(b_row.as_ptr().add(p+24)), sum3);
                        }
                        p += 32;
                    }
                    let combined = unsafe {
                        let s01 = _mm256_add_ps(sum0, sum1);
                        let s23 = _mm256_add_ps(sum2, sum3);
                        let s = _mm256_add_ps(s01, s23);
                        // Horizontal sum of 8 floats
                        let hi = _mm256_extractf128_ps(s, 1);
                        let lo = _mm256_castps256_ps128(s);
                        let sum128 = _mm_add_ps(lo, hi);
                        let shuf = _mm_movehdup_ps(sum128);
                        let sums = _mm_add_ps(sum128, shuf);
                        let shuf2 = _mm_movehl_ps(sums, sums);
                        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
                    };
                    let mut s = combined;
                    while p < k { s += a_row[p] * b_row[p]; p += 1; }
                    out[i * n + j] = s;
                }
            }
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            // SSE2 fallback (always available on x86_64)
            for i in 0..m {
                let a_row = &a[i * k..(i + 1) * k];
                for j in 0..n {
                    let b_row = &b[j * k..(j + 1) * k];
                    let mut s0 = 0.0f32;
                    let mut s1 = 0.0f32;
                    let k4 = k / 4 * 4;
                    let mut p = 0;
                    while p < k4 {
                        s0 += a_row[p] * b_row[p] + a_row[p+2] * b_row[p+2];
                        s1 += a_row[p+1] * b_row[p+1] + a_row[p+3] * b_row[p+3];
                        p += 4;
                    }
                    let mut s = s0 + s1;
                    while p < k { s += a_row[p] * b_row[p]; p += 1; }
                    out[i * n + j] = s;
                }
            }
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k { s += a[i*k+p] * b[j*k+p]; }
                out[i * n + j] = s;
            }
        }
    }
}

/// Extract a slice from a packed projection: (L, total_dim) → (L, slice_dim)
fn proj_slice(proj: &[f32], l: usize, total: usize, offset: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; l * dim];
    for t in 0..l {
        for i in 0..dim {
            out[t * dim + i] = proj[t * total + offset + i];
        }
    }
    out
}
