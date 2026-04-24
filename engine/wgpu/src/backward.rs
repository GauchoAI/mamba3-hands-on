//! Backward pass (gradients) for the Mamba-3 model.
//!
//! Each forward op has a corresponding backward function that computes
//! gradients with respect to its inputs and parameters.

/// SSM scan backward — the critical gradient computation.
///
/// Forward: for t in 0..L:
///   h[t] = decay[t] * h[t-1] + inp[t]
///   y[t] = sum(h[t] * C[t]) + D * x[t]
///   y[t] = y[t] * z_silu[t]
///
/// Backward: reverse scan with adjoint state.
///   dh[t] = dy[t] * z_silu[t] ⊗ C[t] + decay[t+1] * dh[t+1]
///   d_inp[t] = dh[t]
///   d_decay[t] = dh[t] · h[t-1]   (dot product)
///   d_C[t] = dy[t] * z_silu[t] · h[t]
///   d_z_silu[t] = dy[t] · (sum(h[t]*C[t]) + D*x[t])
///   d_x[t] = dy[t] * z_silu[t] * D
///   d_D = sum_t(dy[t] * z_silu[t] * x[t])
pub fn ssm_scan_backward(
    // Forward pass saved tensors
    inp: &[f32],     // (L, H, hD, dS)
    decay: &[f32],   // (L, H)
    c: &[f32],       // (L, dS)  -- broadcast across heads
    x: &[f32],       // (L, H, hD)
    z_silu: &[f32],  // (L, H, hD)
    d_param: &[f32], // (H,)
    // States saved during forward
    states: &[f32],  // (L+1, H, hD, dS) — h at each timestep
    // Gradient of output
    dy: &[f32],      // (L, H, hD)
    // Dimensions
    l: usize, h: usize, hd: usize, ds: usize,
) -> ScanGrads {
    let mut d_inp = vec![0.0f32; l * h * hd * ds];
    let mut d_decay = vec![0.0f32; l * h];
    let mut d_c = vec![0.0f32; l * ds];
    let mut d_x = vec![0.0f32; l * h * hd];
    let mut d_z_silu = vec![0.0f32; l * h * hd];
    let mut d_d = vec![0.0f32; h];

    // Adjoint state: dh accumulates backwards
    let mut dh = vec![0.0f32; h * hd * ds];

    for t in (0..l).rev() {
        for hi in 0..h {
            // dy_gated = dy[t] (already gated in forward, but we need pre-gate)
            // Actually: y_pregate = sum(h*C) + D*x, then y = y_pregate * z_silu
            // So: d_y_pregate = dy * z_silu, d_z_silu = dy * y_pregate

            for p in 0..hd {
                let dy_val = dy[(t * h + hi) * hd + p];
                let zs_val = z_silu[(t * h + hi) * hd + p];

                // Compute y_pregate for d_z_silu
                let mut y_pregate = 0.0f32;
                for n in 0..ds {
                    y_pregate += states[((t + 1) * h + hi) * hd * ds + p * ds + n]
                              * c[t * ds + n];
                }
                y_pregate += d_param[hi] * x[(t * h + hi) * hd + p];

                // d_z_silu
                d_z_silu[(t * h + hi) * hd + p] += dy_val * y_pregate;

                // d_y_pregate = dy * z_silu
                let dy_pre = dy_val * zs_val;

                // d_x
                d_x[(t * h + hi) * hd + p] += dy_pre * d_param[hi];

                // d_D
                d_d[hi] += dy_pre * x[(t * h + hi) * hd + p];

                // Accumulate into dh: dh += dy_pre * C (outer product over dS)
                for n in 0..ds {
                    dh[(hi * hd + p) * ds + n] += dy_pre * c[t * ds + n];
                }

                // d_C: += dy_pre * h[t]
                for n in 0..ds {
                    d_c[t * ds + n] += dy_pre
                        * states[((t + 1) * h + hi) * hd * ds + p * ds + n];
                }
            }

            // Now propagate dh through the state update: h[t] = decay*h[t-1] + inp
            // d_inp[t] = dh
            // d_decay[t] += dh · h[t-1]
            // dh_prev = decay[t] * dh (carried to next iteration)

            let dec = decay[t * h + hi];
            let mut d_dec = 0.0f32;

            for p in 0..hd {
                for n in 0..ds {
                    let si = (hi * hd + p) * ds + n;
                    let dh_val = dh[si];

                    // d_inp
                    d_inp[(t * h + hi) * hd * ds + p * ds + n] = dh_val;

                    // d_decay: dh · h[t-1]
                    let h_prev = states[(t * h + hi) * hd * ds + p * ds + n];
                    d_dec += dh_val * h_prev;

                    // Propagate: dh[t-1] += decay * dh[t]
                    dh[si] = dec * dh_val;
                }
            }

            d_decay[t * h + hi] += d_dec;
        }
    }

    ScanGrads { d_inp, d_decay, d_c, d_x, d_z_silu, d_d }
}

pub struct ScanGrads {
    pub d_inp: Vec<f32>,     // (L, H, hD, dS)
    pub d_decay: Vec<f32>,   // (L, H)
    pub d_c: Vec<f32>,       // (L, dS)
    pub d_x: Vec<f32>,       // (L, H, hD)
    pub d_z_silu: Vec<f32>,  // (L, H, hD)
    pub d_d: Vec<f32>,       // (H,)
}

/// Linear layer backward: out = x @ W^T
/// Given d_out, compute d_x = d_out @ W, d_W = d_out^T @ x
pub fn linear_backward(
    d_out: &[f32],  // (m, n)
    x: &[f32],      // (m, k)
    w: &[f32],      // (n, k)
    m: usize, k: usize, n: usize,
) -> (Vec<f32>, Vec<f32>) {
    // d_x = d_out @ W  — (m, n) × (n, k) → (m, k)
    let mut d_x = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut s = 0.0f32;
            for p in 0..n {
                s += d_out[i * n + p] * w[p * k + j];
            }
            d_x[i * k + j] = s;
        }
    }

    // d_W = d_out^T @ x  — (n, m) × (m, k) → (n, k)
    let mut d_w = vec![0.0f32; n * k];
    for i in 0..n {
        for j in 0..k {
            let mut s = 0.0f32;
            for p in 0..m {
                s += d_out[p * n + i] * x[p * k + j];
            }
            d_w[i * k + j] = s;
        }
    }

    (d_x, d_w)
}

/// Layer norm backward
pub fn layer_norm_backward(
    d_out: &[f32],  // (seq_len, d)
    x: &[f32],      // (seq_len, d) — input to layer norm
    w: &[f32],      // (d,)
    seq_len: usize, d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let eps = 1e-5f32;
    let mut d_x = vec![0.0f32; seq_len * d];
    let mut d_w = vec![0.0f32; d];
    let mut d_b = vec![0.0f32; d];

    for t in 0..seq_len {
        let off = t * d;
        let mean: f32 = (0..d).map(|i| x[off + i]).sum::<f32>() / d as f32;
        let var: f32 = (0..d).map(|i| (x[off + i] - mean).powi(2)).sum::<f32>() / d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        // Normalized values
        let x_norm: Vec<f32> = (0..d).map(|i| (x[off + i] - mean) * inv_std).collect();

        // d_b += d_out
        // d_w += d_out * x_norm
        for i in 0..d {
            d_b[i] += d_out[off + i];
            d_w[i] += d_out[off + i] * x_norm[i];
        }

        // d_x_norm = d_out * w
        let d_x_norm: Vec<f32> = (0..d).map(|i| d_out[off + i] * w[i]).collect();

        // d_var and d_mean
        let d_var: f32 = (0..d).map(|i| d_x_norm[i] * (x[off + i] - mean) * -0.5 * inv_std.powi(3)).sum();
        let d_mean: f32 = (0..d).map(|i| -d_x_norm[i] * inv_std).sum::<f32>()
            + d_var * (0..d).map(|i| -2.0 * (x[off + i] - mean)).sum::<f32>() / d as f32;

        for i in 0..d {
            d_x[off + i] = d_x_norm[i] * inv_std
                + d_var * 2.0 * (x[off + i] - mean) / d as f32
                + d_mean / d as f32;
        }
    }

    (d_x, d_w, d_b)
}

/// Embedding backward: accumulate gradients per token
pub fn embedding_backward(
    d_out: &[f32],   // (seq_len, d)
    tokens: &[u32],
    vocab_size: usize, d: usize,
) -> Vec<f32> {
    let mut d_embed = vec![0.0f32; vocab_size * d];
    for (t, &tok) in tokens.iter().enumerate() {
        let tok = tok as usize;
        if tok < vocab_size {
            for i in 0..d {
                d_embed[tok * d + i] += d_out[t * d + i];
            }
        }
    }
    d_embed
}

/// Cross-entropy loss + backward
pub fn cross_entropy_loss(logits: &[f32], targets: &[u32], vocab: usize, seq_len: usize) -> (f32, Vec<f32>) {
    let mut loss = 0.0f32;
    let mut d_logits = vec![0.0f32; seq_len * vocab];

    for t in 0..seq_len {
        let target = targets[t] as usize;
        let off = t * vocab;

        // Softmax
        let max_logit = logits[off..off + vocab].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for i in 0..vocab {
            let e = (logits[off + i] - max_logit).exp();
            d_logits[off + i] = e;
            exp_sum += e;
        }
        for i in 0..vocab {
            d_logits[off + i] /= exp_sum;
        }

        // Loss: -log(softmax[target])
        loss -= (d_logits[off + target]).ln();

        // Gradient: softmax - one_hot
        d_logits[off + target] -= 1.0;
    }

    loss /= seq_len as f32;
    for v in d_logits.iter_mut() {
        *v /= seq_len as f32;
    }

    (loss, d_logits)
}

/// AdamW optimizer step
pub fn adamw_step(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],      // first moment
    v: &mut [f32],      // second moment
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) {
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);

    for i in 0..params.len() {
        // Weight decay (decoupled)
        params[i] *= 1.0 - lr * weight_decay;

        // Moment updates
        m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];

        // Bias-corrected
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;

        // Parameter update
        params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}
