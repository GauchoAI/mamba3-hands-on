//! Training loop — forward with activation caching, analytical backward, AdamW.

use crate::model::{Mamba3Model, LayerWeights};
use crate::backward::{cross_entropy_loss, adamw_step, layer_norm_backward, embedding_backward};

pub struct TrainState {
    pub model: Mamba3Model,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub step: u32,
    pub lr: f32,
    pub weight_decay: f32,
}

/// Saved activations for backward pass
struct ForwardCache {
    tokens: Vec<u32>,
    seq_len: usize,
    // Per-layer inputs (before each layer)
    layer_inputs: Vec<Vec<f32>>,     // x before each SSM layer
    // Per-layer SSM cache
    layer_projs: Vec<Vec<f32>>,      // in_proj output
    layer_y_inner: Vec<Vec<f32>>,    // y before out_proj (L, d_inner)
    layer_scan_caches: Vec<ScanCache>,
    // Final
    x_before_head: Vec<f32>,         // after final norm
    x_after_norm_input: Vec<f32>,    // embedding output before layers (for embed grad)
}

impl TrainState {
    pub fn new(model: Mamba3Model, lr: f32, weight_decay: f32) -> Self {
        let n = model.param_count();
        Self {
            model, m: vec![0.0; n], v: vec![0.0; n],
            step: 0, lr, weight_decay,
        }
    }

    pub fn train_step(&mut self, tokens: &[u32], targets: &[u32]) -> f32 {
        self.step += 1;
        let l = tokens.len();

        // Forward with cache
        let (logits, cache) = self.forward_cached(tokens);

        // Loss
        let (loss, d_logits) = cross_entropy_loss(&logits, targets, self.model.vocab_size, l);

        // Analytical backward
        let grads = self.backward_analytical(&d_logits, &cache);

        // Optimizer
        let mut params = self.model.collect_params();
        adamw_step(
            &mut params, &grads, &mut self.m, &mut self.v,
            self.lr, 0.9, 0.999, 1e-8, self.weight_decay, self.step,
        );
        self.model.scatter_params(&params);

        loss
    }

    fn forward_cached(&self, tokens: &[u32]) -> (Vec<f32>, ForwardCache) {
        let l = tokens.len();
        let d = self.model.d_model;
        let v = self.model.vocab_size;

        // Embedding
        let mut x = vec![0.0f32; l * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok < v {
                for i in 0..d {
                    x[t * d + i] = self.model.embed_w[tok * d + i];
                }
            }
        }
        let x_pre_norm = x.clone();
        layer_norm_inplace(&mut x, &self.model.embed_norm_w, &self.model.embed_norm_b, l, d);
        let x_after_norm = x.clone();

        // Layers
        let mut layer_inputs = Vec::new();
        let mut layer_projs = Vec::new();
        let mut layer_y_inner = Vec::new();
        let mut layer_scan_caches = Vec::new();

        for layer in &self.model.layers {
            layer_inputs.push(x.clone());
            let (y, proj, y_inner, scan_cache) = self.ssm_layer_cached(&x, layer, l);
            layer_projs.push(proj);
            layer_y_inner.push(y_inner);
            layer_scan_caches.push(scan_cache);
            for i in 0..l * d {
                x[i] += layer.scale * y[i];
            }
        }

        // Final norm
        let x_before_final_norm = x.clone();
        layer_norm_inplace(&mut x, &self.model.final_norm_w, &self.model.final_norm_b, l, d);
        let x_before_head = x.clone();

        // LM head
        let mut logits = vec![0.0f32; l * v];
        for t in 0..l {
            for vi in 0..v {
                let mut s = 0.0f32;
                for i in 0..d {
                    s += x[t * d + i] * self.model.embed_w[vi * d + i];
                }
                logits[t * v + vi] = s;
            }
        }

        let cache = ForwardCache {
            tokens: tokens.to_vec(),
            seq_len: l,
            layer_inputs,
            layer_projs,
            layer_y_inner,
            layer_scan_caches,
            x_before_head,
            x_after_norm_input: x_after_norm,
        };

        (logits, cache)
    }

    fn ssm_layer_cached(&self, x_in: &[f32], lw: &LayerWeights, l: usize)
        -> (Vec<f32>, Vec<f32>, Vec<f32>, ScanCache)
    {
        let d = self.model.d_model;
        let di = self.model.d_inner;

        // In-projection
        let dip = lw.d_in_proj;
        let mut proj = vec![0.0f32; l * dip];
        matmul_t(&mut proj, x_in, &lw.in_proj_w, l, d, dip);

        // SSM with state caching
        let (y_inner, scan_cache) = self.model.run_ssm_from_proj(&proj, lw, l);

        // Out-projection
        let mut out = vec![0.0f32; l * d];
        for t in 0..l {
            for j in 0..d {
                let mut s = 0.0f32;
                for i in 0..di {
                    s += y_inner[t * di + i] * lw.out_proj_w[j * di + i];
                }
                out[t * d + j] = s;
            }
        }

        (out, proj, y_inner, scan_cache)
    }

    fn backward_analytical(&self, d_logits: &[f32], cache: &ForwardCache) -> Vec<f32> {
        let l = cache.seq_len;
        let d = self.model.d_model;
        let v = self.model.vocab_size;
        let di = self.model.d_inner;
        let n_params = self.model.param_count();
        let mut grads = vec![0.0f32; n_params];
        let mut g_off = 0usize;

        // === LM Head backward (weight-tied with embedding) ===
        // logits = x_final @ embed_w^T
        // d_x_final = d_logits @ embed_w  (M,V) × (V,D) → (M,D)
        // d_embed_w += d_logits^T @ x_final  (V,M) × (M,D) → (V,D)

        let mut d_x = vec![0.0f32; l * d];
        let mut d_embed = vec![0.0f32; v * d];

        for t in 0..l {
            for i in 0..d {
                let mut s = 0.0f32;
                for vi in 0..v {
                    s += d_logits[t * v + vi] * self.model.embed_w[vi * d + i];
                }
                d_x[t * d + i] = s;
            }
            for vi in 0..v {
                for i in 0..d {
                    d_embed[vi * d + i] += d_logits[t * v + vi] * cache.x_before_head[t * d + i];
                }
            }
        }

        // === Final layer norm backward ===
        let (d_x_pre_norm, d_fnorm_w, d_fnorm_b) = layer_norm_backward(
            &d_x, &cache.layer_inputs.last().map_or_else(
                || cache.x_after_norm_input.clone(),
                |_| {
                    // Reconstruct pre-final-norm x from last layer output
                    // This is approximate — ideally we'd cache this too
                    let mut x = cache.layer_inputs.last().unwrap().clone();
                    let last_layer = &self.model.layers[self.model.n_layers - 1];
                    let y = &cache.layer_y_inner.last().unwrap();
                    for t in 0..l {
                        for j in 0..d {
                            let mut s = 0.0f32;
                            for i in 0..di {
                                s += y[t * di + i] * last_layer.out_proj_w[j * di + i];
                            }
                            x[t * d + j] += last_layer.scale * s;
                        }
                    }
                    x
                }
            ), &self.model.final_norm_w, l, d,
        );
        d_x = d_x_pre_norm;

        // === Layers backward (reverse order) ===
        let mut d_layer_params: Vec<Vec<f32>> = Vec::new();

        for li in (0..self.model.n_layers).rev() {
            let layer = &self.model.layers[li];
            let x_in = &cache.layer_inputs[li];
            let y_inner = &cache.layer_y_inner[li];

            // d_x *= scale (residual backward)
            let mut d_residual = d_x.clone();
            let mut d_y_out = vec![0.0f32; l * d];
            for i in 0..l * d {
                d_y_out[i] = d_x[i] * layer.scale;
            }

            // Out-projection backward: out = y_inner @ out_proj_w^T
            // d_y_inner = d_y_out @ out_proj_w  (L,D) × (D,DI) → (L,DI)
            // d_out_proj_w = d_y_out^T @ y_inner  (D,L) × (L,DI) → (D,DI)
            let mut d_y_inner = vec![0.0f32; l * di];
            let mut d_out_proj_w = vec![0.0f32; d * di];

            for t in 0..l {
                for i in 0..di {
                    let mut s = 0.0f32;
                    for j in 0..d {
                        s += d_y_out[t * d + j] * layer.out_proj_w[j * di + i];
                    }
                    d_y_inner[t * di + i] = s;
                }
                for j in 0..d {
                    for i in 0..di {
                        d_out_proj_w[j * di + i] += d_y_out[t * d + j] * y_inner[t * di + i];
                    }
                }
            }

            // === SSM backward with proper adjoint scan ===
            let dip = layer.d_in_proj;
            let nh = self.model.n_heads;
            let hd = self.model.headdim;
            let ds = self.model.d_state;
            let mut d_in_proj_w = vec![0.0f32; dip * d];
            let mut d_proj = vec![0.0f32; l * dip];

            let sc = &cache.layer_scan_caches[li];
            let proj = &cache.layer_projs[li];
            let z_raw = proj_slice(proj, l, dip, 0, di);

            // Gate backward: y = y_pregate * z_silu
            // d_y_pregate = d_y * z_silu
            // d_z_silu = d_y * y_pregate
            let mut d_y_pregate = vec![0.0f32; l * di];
            for t in 0..l {
                for i in 0..di {
                    d_y_pregate[t * di + i] = d_y_inner[t * di + i] * sc.z_silu[t * di + i];
                    // d_z via silu derivative
                    let zs = sc.z_silu[t * di + i];
                    let y_pre = if zs.abs() > 1e-8 { y_inner[t * di + i] / zs } else { 0.0 };
                    let z = z_raw[t * di + i];
                    let s = sigmoid(z);
                    let dsilu_dz = s + z * s * (1.0 - s);
                    d_proj[t * dip + i] = d_y_inner[t * di + i] * y_pre * dsilu_dz;
                }
            }

            // Adjoint scan: propagate d_y_pregate backwards through the SSM recurrence
            // dh[t] = d_y_pregate[t] outer C[t] + decay[t+1] * dh[t+1]
            let mut dh = vec![0.0f32; nh * hd * ds];
            let mut d_scan_inp = vec![0.0f32; l * nh * hd * ds];

            for t in (0..l).rev() {
                for h in 0..nh {
                    // Add output gradient to adjoint state
                    for p in 0..hd {
                        let dy_pre = d_y_pregate[t * di + h * hd + p];
                        for n in 0..ds {
                            dh[(h * hd + p) * ds + n] += dy_pre * sc.cp[t * ds + n];
                        }
                        // d_x from skip connection
                        d_proj[t * dip + di + h * hd + p] += dy_pre * layer.d_param[h];
                    }

                    // d_Cp: += d_y_pregate * h[t]
                    for n in 0..ds {
                        let mut d_cp_n = 0.0f32;
                        for p in 0..hd {
                            d_cp_n += d_y_pregate[t * di + h * hd + p]
                                * sc.states[((t + 1) * nh * hd * ds) + (h * hd + p) * ds + n];
                        }
                        // Cp is broadcast across heads — accumulate
                        d_proj[t * dip + 2 * di + ds + n] += d_cp_n;
                    }

                    // d_inp[t] = dh (the adjoint state IS the gradient of inp)
                    for p in 0..hd {
                        for n in 0..ds {
                            d_scan_inp[((t * nh + h) * hd + p) * ds + n] = dh[(h * hd + p) * ds + n];
                        }
                    }

                    // Propagate adjoint: dh[t-1] = decay[t] * dh[t]
                    let dec = sc.decay[t * nh + h];
                    for p in 0..hd {
                        for n in 0..ds {
                            dh[(h * hd + p) * ds + n] *= dec;
                        }
                    }
                }
            }

            // d_scan_inp → flows back through: inp = (trap * Bx + ...) * dt
            // d_Bx = d_scan_inp * dt * trap (simplified, ignoring prev term)
            // Bx = outer(x, Bp), so:
            // d_x[p] = sum_n(d_Bx[p,n] * Bp[n])
            // d_Bp[n] = sum_p(d_Bx[p,n] * x[p])
            // Get Bp from cache (it was normalized)
            let bp_proj = proj_slice(proj, l, dip, 2 * di, ds);
            // Note: bp was layer-normed during forward. Use the raw projection as approx.

            for t in 0..l {
                for h in 0..nh {
                    let dt_val = softplus(
                        proj_slice(proj, l, dip, 2*di + 2*ds, nh)[t * nh + h] + layer.dt_bias[h]
                    );
                    let tr = sigmoid(
                        proj_slice(proj, l, dip, 2*di + 2*ds + 2*nh, nh)[t * nh + h]
                    );

                    for p in 0..hd {
                        let mut d_x_val = 0.0f32;
                        for n in 0..ds {
                            let d_bx = d_scan_inp[((t * nh + h) * hd + p) * ds + n] / (dt_val * tr + 1e-8);
                            // d_Bp[n] += d_bx * x[p]
                            d_proj[t * dip + 2 * di + n] += d_bx * sc.x_raw[t * di + h * hd + p];
                            // d_x[p] += d_bx * Bp[n]
                            d_x_val += d_bx * bp_proj[t * ds + n];
                        }
                        d_proj[t * dip + di + h * hd + p] += d_x_val;
                    }
                }
            }

            // d_in_proj_w = d_proj^T @ x_in
            for t in 0..l {
                for j in 0..dip {
                    for i in 0..d {
                        d_in_proj_w[j * d + i] += d_proj[t * dip + j] * x_in[t * d + i];
                    }
                }
            }

            // d_x through in_proj: d_x_in = d_proj @ in_proj_w
            let mut d_x_from_layer = vec![0.0f32; l * d];
            for t in 0..l {
                for i in 0..d {
                    let mut s = 0.0f32;
                    for j in 0..dip {
                        s += d_proj[t * dip + j] * layer.in_proj_w[j * d + i];
                    }
                    d_x_from_layer[t * d + i] = s;
                }
            }

            // Residual: d_x = d_residual + scale * d_x_from_layer
            for i in 0..l * d {
                d_x[i] = d_residual[i] + layer.scale * d_x_from_layer[i];
            }

            // Collect layer param grads
            let mut lgrads = Vec::new();
            lgrads.extend_from_slice(&d_in_proj_w);
            lgrads.extend_from_slice(&d_out_proj_w);
            lgrads.extend(vec![0.0f32; layer.dt_bias.len()]);   // dt_bias grad (approx 0)
            lgrads.extend(vec![0.0f32; layer.d_param.len()]);   // D grad
            lgrads.extend(vec![0.0f32; layer.b_norm_w.len()]); // B norm
            lgrads.extend(vec![0.0f32; layer.b_norm_b.len()]);
            lgrads.extend(vec![0.0f32; layer.c_norm_w.len()]); // C norm
            lgrads.extend(vec![0.0f32; layer.c_norm_b.len()]);
            lgrads.extend(vec![0.0f32; layer.layer_norm_w.len()]);
            lgrads.push(0.0); // scale
            d_layer_params.push(lgrads);
        }

        // === Embed norm backward ===
        // (simplified — skip for now, embed gets gradient from head)

        // === Embedding backward (from head, weight-tied) ===
        // Also accumulate from d_x through embedding lookup
        let d_embed_from_input = embedding_backward(&d_x, &cache.tokens, v, d);
        for i in 0..v * d {
            d_embed[i] += d_embed_from_input[i];
        }

        // === Pack gradients in parameter order ===
        grads[g_off..g_off + v * d].copy_from_slice(&d_embed);
        g_off += v * d;
        g_off += d; // embed_norm_w (skip for now)
        g_off += d; // embed_norm_b

        // Layers (were computed in reverse, need to reverse back)
        d_layer_params.reverse();
        for lgrads in &d_layer_params {
            grads[g_off..g_off + lgrads.len()].copy_from_slice(lgrads);
            g_off += lgrads.len();
        }

        // Final norm
        grads[g_off..g_off + d].copy_from_slice(&d_fnorm_w);
        g_off += d;
        grads[g_off..g_off + d].copy_from_slice(&d_fnorm_b);

        grads
    }
}

impl Mamba3Model {
    pub fn param_count(&self) -> usize {
        let mut n = self.embed_w.len() + self.embed_norm_w.len() + self.embed_norm_b.len()
            + self.final_norm_w.len() + self.final_norm_b.len();
        for layer in &self.layers {
            n += layer.in_proj_w.len() + layer.out_proj_w.len()
                + layer.dt_bias.len() + layer.d_param.len()
                + layer.b_norm_w.len() + layer.b_norm_b.len()
                + layer.c_norm_w.len() + layer.c_norm_b.len()
                + layer.layer_norm_w.len() + 1;
        }
        n
    }

    pub fn collect_params(&self) -> Vec<f32> {
        let mut p = Vec::with_capacity(self.param_count());
        p.extend_from_slice(&self.embed_w);
        p.extend_from_slice(&self.embed_norm_w);
        p.extend_from_slice(&self.embed_norm_b);
        for layer in &self.layers {
            p.extend_from_slice(&layer.in_proj_w);
            p.extend_from_slice(&layer.out_proj_w);
            p.extend_from_slice(&layer.dt_bias);
            p.extend_from_slice(&layer.d_param);
            p.extend_from_slice(&layer.b_norm_w);
            p.extend_from_slice(&layer.b_norm_b);
            p.extend_from_slice(&layer.c_norm_w);
            p.extend_from_slice(&layer.c_norm_b);
            p.extend_from_slice(&layer.layer_norm_w);
            p.push(layer.scale);
        }
        p.extend_from_slice(&self.final_norm_w);
        p.extend_from_slice(&self.final_norm_b);
        p
    }

    pub fn scatter_params(&mut self, params: &[f32]) {
        let mut off = 0usize;
        fn cp(dst: &mut [f32], src: &[f32], off: &mut usize) {
            dst.copy_from_slice(&src[*off..*off + dst.len()]);
            *off += dst.len();
        }
        cp(&mut self.embed_w, params, &mut off);
        cp(&mut self.embed_norm_w, params, &mut off);
        cp(&mut self.embed_norm_b, params, &mut off);
        for layer in &mut self.layers {
            cp(&mut layer.in_proj_w, params, &mut off);
            cp(&mut layer.out_proj_w, params, &mut off);
            cp(&mut layer.dt_bias, params, &mut off);
            cp(&mut layer.d_param, params, &mut off);
            cp(&mut layer.b_norm_w, params, &mut off);
            cp(&mut layer.b_norm_b, params, &mut off);
            cp(&mut layer.c_norm_w, params, &mut off);
            cp(&mut layer.c_norm_b, params, &mut off);
            cp(&mut layer.layer_norm_w, params, &mut off);
            layer.scale = params[off];
            off += 1;
        }
        cp(&mut self.final_norm_w, params, &mut off);
        cp(&mut self.final_norm_b, params, &mut off);
    }

    /// Run SSM from already-computed projection (used by both forward and cached forward)
    pub fn run_ssm_from_proj(&self, proj: &[f32], lw: &LayerWeights, l: usize) -> (Vec<f32>, ScanCache) {
        let di = self.d_inner;
        let nh = self.n_heads;
        let hd = self.headdim;
        let ds = self.d_state;
        let dip = lw.d_in_proj;

        // Split and process (same as model.rs mamba3_block, from proj onward)
        let mut off = 0;
        let z_raw = proj_slice(proj, l, dip, off, di); off += di;
        let x_raw = proj_slice(proj, l, dip, off, di); off += di;
        let bp_raw = proj_slice(proj, l, dip, off, ds); off += ds;
        let cp_raw = proj_slice(proj, l, dip, off, ds); off += ds;
        let dd_dt = proj_slice(proj, l, dip, off, nh); off += nh;
        let dd_a = proj_slice(proj, l, dip, off, nh); off += nh;
        let trap_raw = proj_slice(proj, l, dip, off, nh);

        // Norm B, C
        let mut bp = bp_raw;
        layer_norm_inplace(&mut bp, &lw.b_norm_w, &lw.b_norm_b, l, ds);
        let mut cp = cp_raw;
        layer_norm_inplace(&mut cp, &lw.c_norm_w, &lw.c_norm_b, l, ds);

        // DT, decay, trap
        let mut dt = vec![0.0f32; l * nh];
        let mut decay = vec![0.0f32; l * nh];
        let mut trap = vec![0.0f32; l * nh];
        for t in 0..l {
            for h in 0..nh {
                dt[t * nh + h] = softplus(dd_dt[t * nh + h] + lw.dt_bias[h]);
                let a = -softplus(dd_a[t * nh + h]).max(0.001);
                decay[t * nh + h] = (a * dt[t * nh + h]).exp();
                trap[t * nh + h] = sigmoid(trap_raw[t * nh + h]);
            }
        }

        // Compute inp with trapezoidal
        let mut inp = vec![0.0f32; l * nh * hd * ds];
        let mut bx_prev = vec![0.0f32; nh * hd * ds];
        for t in 0..l {
            for h in 0..nh {
                let mut bx_cur = vec![0.0f32; hd * ds];
                for p in 0..hd {
                    let xv = x_raw[t * di + h * hd + p];
                    for n in 0..ds {
                        bx_cur[p * ds + n] = xv * bp[t * ds + n];
                    }
                }
                let tr = trap[t * nh + h];
                let dtv = dt[t * nh + h];
                for p in 0..hd {
                    for n in 0..ds {
                        let idx = ((t * nh + h) * hd + p) * ds + n;
                        inp[idx] = (tr * bx_cur[p * ds + n] + (1.0 - tr) * bx_prev[h * hd * ds + p * ds + n]) * dtv;
                    }
                }
                for i in 0..hd * ds {
                    bx_prev[h * hd * ds + i] = bx_cur[i];
                }
            }
        }

        // Scan + gate — SAVE states for backward
        let state_size = nh * hd * ds;
        let mut states = vec![0.0f32; (l + 1) * state_size]; // states[0] = zeros (init)
        let mut state = vec![0.0f32; state_size];
        let mut y = vec![0.0f32; l * di];
        let mut z_silu = vec![0.0f32; l * di];
        for i in 0..l * di {
            let z = z_raw[i];
            z_silu[i] = z * sigmoid(z);
        }

        for t in 0..l {
            for h in 0..nh {
                let dec = decay[t * nh + h];
                for p in 0..hd {
                    for n in 0..ds {
                        let si = (h * hd + p) * ds + n;
                        state[si] = dec * state[si] + inp[((t * nh + h) * hd + p) * ds + n];
                    }
                }
                for p in 0..hd {
                    let mut sum = 0.0f32;
                    for n in 0..ds {
                        sum += state[(h * hd + p) * ds + n] * cp[t * ds + n];
                    }
                    sum += lw.d_param[h] * x_raw[t * di + h * hd + p];
                    y[t * di + h * hd + p] = sum * z_silu[t * di + h * hd + p];
                }
            }
            // Save state for backward (after this timestep's update)
            states[(t + 1) * state_size..(t + 2) * state_size].copy_from_slice(&state);
        }

        (y, ScanCache { inp, decay, cp, x_raw: x_raw.clone(), z_silu, states, d_param: lw.d_param.clone(), l, nh, hd, ds, di })
    }
}

/// Cached scan intermediates for backward
struct ScanCache {
    inp: Vec<f32>,      // (L, H, hD, dS)
    decay: Vec<f32>,    // (L, H)
    cp: Vec<f32>,       // (L, dS)
    x_raw: Vec<f32>,    // (L, d_inner) — x values for skip
    z_silu: Vec<f32>,   // (L, d_inner)
    states: Vec<f32>,   // (L+1, H, hD, dS)
    d_param: Vec<f32>,  // (H,)
    l: usize,
    nh: usize,
    hd: usize,
    ds: usize,
    di: usize,
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn softplus(x: f32) -> f32 { (1.0 + x.exp()).ln() }

fn layer_norm_inplace(x: &mut [f32], w: &[f32], b: &[f32], l: usize, d: usize) {
    let eps = 1e-5f32;
    for t in 0..l {
        let off = t * d;
        let mean: f32 = (0..d).map(|i| x[off + i]).sum::<f32>() / d as f32;
        let var: f32 = (0..d).map(|i| (x[off + i] - mean).powi(2)).sum::<f32>() / d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            x[off + i] = (x[off + i] - mean) * inv_std * w[i] + b[i];
        }
    }
}

fn matmul_t(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[j * k + p];
            }
            out[i * n + j] = s;
        }
    }
}

fn proj_slice(proj: &[f32], l: usize, total: usize, offset: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; l * dim];
    for t in 0..l {
        for i in 0..dim {
            out[t * dim + i] = proj[t * total + offset + i];
        }
    }
    out
}
