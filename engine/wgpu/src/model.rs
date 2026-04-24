//! Full Mamba-3 model inference in Rust — embedding, SSM scan, linear, output.
//!
//! No PyTorch dependency. Loads weights from a simple binary format.
//! All fp32, explicit arithmetic.

use std::path::Path;

/// Full Mamba-3 model for inference
pub struct Mamba3Model {
    pub d_model: usize,
    pub d_state: usize,
    pub d_inner: usize,
    pub headdim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub vocab_size: usize,

    // Weights
    pub embed: Vec<f32>,            // (vocab_size, d_model)
    pub embed_norm_w: Vec<f32>,     // (d_model,)
    pub embed_norm_b: Vec<f32>,     // (d_model,)
    pub layers: Vec<LayerWeights>,
    pub final_norm_w: Vec<f32>,     // (d_model,)
    pub final_norm_b: Vec<f32>,     // (d_model,)
    pub head: Vec<f32>,             // (vocab_size, d_model) — tied with embed
}

pub struct LayerWeights {
    pub in_proj_w: Vec<f32>,        // (d_in_proj, d_model)
    pub conv1d_w: Vec<f32>,         // (d_inner, 1, conv_width)
    pub conv1d_b: Vec<f32>,         // (d_inner,)
    pub out_proj_w: Vec<f32>,       // (d_model, d_inner)
    pub dt_bias: Vec<f32>,          // (n_heads,)
    pub a_log: Vec<f32>,            // (n_heads,)
    pub d_param: Vec<f32>,          // (n_heads,)
    pub norm_w: Vec<f32>,           // (d_model,)
    pub scale: f32,                 // near-identity scale
}

impl Mamba3Model {
    /// Load from a PyTorch checkpoint exported as raw binary tensors
    pub fn from_bin(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let header_size = 7 * 4; // 7 u32 values
        if data.len() < header_size {
            return Err("File too small".into());
        }

        let header: &[u32] = bytemuck::cast_slice(&data[..header_size]);
        let d_model = header[0] as usize;
        let d_state = header[1] as usize;
        let headdim = header[2] as usize;
        let n_layers = header[3] as usize;
        let vocab_size = header[4] as usize;
        let d_inner = d_model * 2;
        let n_heads = d_inner / headdim;

        let floats: &[f32] = bytemuck::cast_slice(&data[header_size..]);
        let mut offset = 0;

        let read = |off: &mut usize, n: usize| -> Vec<f32> {
            let slice = floats[*off..*off + n].to_vec();
            *off += n;
            slice
        };

        let embed = read(&mut offset, vocab_size * d_model);
        let embed_norm_w = read(&mut offset, d_model);
        let embed_norm_b = read(&mut offset, d_model);

        let mut layers = Vec::new();
        for _ in 0..n_layers {
            let d_in_proj = 2 * d_inner + 2 * n_heads * d_state + 3 * n_heads + d_state / 2;
            layers.push(LayerWeights {
                in_proj_w: read(&mut offset, d_in_proj * d_model),
                conv1d_w: read(&mut offset, d_inner * 4), // conv_width=4
                conv1d_b: read(&mut offset, d_inner),
                out_proj_w: read(&mut offset, d_model * d_inner),
                dt_bias: read(&mut offset, n_heads),
                a_log: read(&mut offset, n_heads),
                d_param: read(&mut offset, n_heads),
                norm_w: read(&mut offset, d_model),
                scale: read(&mut offset, 1)[0],
            });
        }

        let final_norm_w = read(&mut offset, d_model);
        let final_norm_b = read(&mut offset, d_model);
        let head = embed.clone(); // weight-tied

        Ok(Self {
            d_model, d_state, d_inner, headdim, n_heads, n_layers, vocab_size,
            embed, embed_norm_w, embed_norm_b, layers,
            final_norm_w, final_norm_b, head,
        })
    }

    /// Run inference on a token sequence. Returns logits (seq_len, vocab_size).
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();
        let d = self.d_model;

        // Embedding lookup + layer norm
        let mut x = vec![0.0f32; seq_len * d];
        for (t, &tok) in tokens.iter().enumerate() {
            for i in 0..d {
                x[t * d + i] = self.embed[tok as usize * d + i];
            }
        }
        layer_norm_inplace(&mut x, &self.embed_norm_w, &self.embed_norm_b, seq_len, d);

        // SSM layers
        for layer in &self.layers {
            let residual = x.clone();
            x = self.ssm_layer(&x, layer, seq_len);
            // Residual connection with scale
            for i in 0..seq_len * d {
                x[i] = residual[i] + layer.scale * x[i];
            }
        }

        // Final norm
        layer_norm_inplace(&mut x, &self.final_norm_w, &self.final_norm_b, seq_len, d);

        // LM head: (seq_len, d_model) × (d_model, vocab_size)^T → (seq_len, vocab_size)
        let mut logits = vec![0.0f32; seq_len * self.vocab_size];
        for t in 0..seq_len {
            for v in 0..self.vocab_size {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += x[t * d + i] * self.head[v * d + i];
                }
                logits[t * self.vocab_size + v] = sum;
            }
        }

        logits
    }

    fn ssm_layer(&self, x_in: &[f32], layer: &LayerWeights, seq_len: usize) -> Vec<f32> {
        let d = self.d_model;
        let di = self.d_inner;
        let nh = self.n_heads;
        let hd = self.headdim;
        let ds = self.d_state;

        // In-projection: (seq_len, d_model) × (d_in_proj, d_model)^T
        let d_in_proj = layer.in_proj_w.len() / d;
        let mut proj = vec![0.0f32; seq_len * d_in_proj];
        for t in 0..seq_len {
            for j in 0..d_in_proj {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += x_in[t * d + i] * layer.in_proj_w[j * d + i];
                }
                proj[t * d_in_proj + j] = sum;
            }
        }

        // Split projections (simplified — skipping conv1d, RoPE, trapezoidal for now)
        // This is a simplified forward pass for inference verification
        let mut z = vec![0.0f32; seq_len * di];
        let mut x_val = vec![0.0f32; seq_len * di];
        for t in 0..seq_len {
            for i in 0..di {
                z[t * di + i] = proj[t * d_in_proj + i];
                x_val[t * di + i] = proj[t * d_in_proj + di + i];
            }
        }

        // SSM scan (simplified — using x_val as both inp and output)
        // Full implementation would compute B, C, dt from projections
        // For now, output = silu(z) * x_val as a placeholder
        let mut y = vec![0.0f32; seq_len * di];
        for t in 0..seq_len {
            for i in 0..di {
                let z_val = z[t * di + i];
                let silu = z_val * sigmoid(z_val);
                y[t * di + i] = silu * x_val[t * di + i];
            }
        }

        // Out-projection: (seq_len, d_inner) × (d_model, d_inner)^T
        let mut out = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..di {
                    sum += y[t * di + i] * layer.out_proj_w[j * di + i];
                }
                out[t * d + j] = sum;
            }
        }

        out
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn layer_norm_inplace(x: &mut [f32], w: &[f32], b: &[f32], seq_len: usize, d: usize) {
    let eps = 1e-5f32;
    for t in 0..seq_len {
        let slice = &x[t * d..(t + 1) * d];
        let mean: f32 = slice.iter().sum::<f32>() / d as f32;
        let var: f32 = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let std = (var + eps).sqrt();
        for i in 0..d {
            x[t * d + i] = (x[t * d + i] - mean) / std * w[i] + b[i];
        }
    }
}
