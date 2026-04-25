//! PtxTrainer — forward_cached + cross-entropy + backward + AdamW,
//! producing a single f32 loss per train_step.  Matches mamba3_engine::TrainState
//! semantics: same AdamW, same simplified backward (gradients for dt_bias,
//! d_param, layer norm weights, b/c norm weights, scale are all zero).

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use std::error::Error;
use std::sync::Arc;

use crate::model::PtxModel;
use crate::runtime::PtxContext;
use crate::train_scratch::TrainScratch;

pub struct PtxTrainer {
    pub model: PtxModel,
    pub train_scratch: TrainScratch,

    // Optimizer state per trainable weight tensor (one CudaSlice each)
    pub m_embed: CudaSlice<f32>,
    pub v_embed: CudaSlice<f32>,
    pub m_in_proj: Vec<CudaSlice<f32>>,
    pub v_in_proj: Vec<CudaSlice<f32>>,
    pub m_out_proj: Vec<CudaSlice<f32>>,
    pub v_out_proj: Vec<CudaSlice<f32>>,
    pub m_fnorm_w: CudaSlice<f32>,
    pub v_fnorm_w: CudaSlice<f32>,
    pub m_fnorm_b: CudaSlice<f32>,
    pub v_fnorm_b: CudaSlice<f32>,
    pub m_d_param: Vec<CudaSlice<f32>>,
    pub v_d_param: Vec<CudaSlice<f32>>,
    pub m_dt_bias: Vec<CudaSlice<f32>>,
    pub v_dt_bias: Vec<CudaSlice<f32>>,
    pub m_layer_norm_w: Vec<CudaSlice<f32>>,
    pub v_layer_norm_w: Vec<CudaSlice<f32>>,
    pub m_layer_norm_b: Vec<CudaSlice<f32>>,
    pub v_layer_norm_b: Vec<CudaSlice<f32>>,
    pub m_b_norm_w: Vec<CudaSlice<f32>>,
    pub v_b_norm_w: Vec<CudaSlice<f32>>,
    pub m_b_norm_b: Vec<CudaSlice<f32>>,
    pub v_b_norm_b: Vec<CudaSlice<f32>>,
    pub m_c_norm_w: Vec<CudaSlice<f32>>,
    pub v_c_norm_w: Vec<CudaSlice<f32>>,
    pub m_c_norm_b: Vec<CudaSlice<f32>>,
    pub v_c_norm_b: Vec<CudaSlice<f32>>,
    pub m_embed_norm_w: CudaSlice<f32>,
    pub v_embed_norm_w: CudaSlice<f32>,
    pub m_embed_norm_b: CudaSlice<f32>,
    pub v_embed_norm_b: CudaSlice<f32>,

    pub step: u32,
    pub lr: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl PtxTrainer {
    pub fn new(model: PtxModel, lr: f32, weight_decay: f32, max_seq: usize) -> Result<Self, Box<dyn Error>> {
        let ctx = model.ptx.clone();
        let stream = &ctx.stream;

        let d = model.d_model;
        let di = model.d_inner;
        let ds = model.d_state;
        let hd = model.headdim;
        let nh = model.n_heads;
        let v = model.vocab_size;
        let dip = model.d_in_proj;
        let nl = model.n_layers;

        let train_scratch = TrainScratch::new(
            &ctx, max_seq, nl, d, ds, di, nh, hd, dip, v, model.num_rope_angles.max(1),
        )?;

        let m_embed = stream.alloc_zeros::<f32>(v * d)?;
        let v_embed = stream.alloc_zeros::<f32>(v * d)?;
        let mut m_in_proj = Vec::with_capacity(nl);
        let mut v_in_proj = Vec::with_capacity(nl);
        let mut m_out_proj = Vec::with_capacity(nl);
        let mut v_out_proj = Vec::with_capacity(nl);
        let mut m_d_param = Vec::with_capacity(nl);
        let mut v_d_param = Vec::with_capacity(nl);
        let mut m_dt_bias = Vec::with_capacity(nl);
        let mut v_dt_bias = Vec::with_capacity(nl);
        let mut m_layer_norm_w = Vec::with_capacity(nl);
        let mut v_layer_norm_w = Vec::with_capacity(nl);
        let mut m_layer_norm_b = Vec::with_capacity(nl);
        let mut v_layer_norm_b = Vec::with_capacity(nl);
        let mut m_b_norm_w = Vec::with_capacity(nl);
        let mut v_b_norm_w = Vec::with_capacity(nl);
        let mut m_b_norm_b = Vec::with_capacity(nl);
        let mut v_b_norm_b = Vec::with_capacity(nl);
        let mut m_c_norm_w = Vec::with_capacity(nl);
        let mut v_c_norm_w = Vec::with_capacity(nl);
        let mut m_c_norm_b = Vec::with_capacity(nl);
        let mut v_c_norm_b = Vec::with_capacity(nl);
        for _ in 0..nl {
            m_in_proj.push(stream.alloc_zeros::<f32>(dip * d)?);
            v_in_proj.push(stream.alloc_zeros::<f32>(dip * d)?);
            m_out_proj.push(stream.alloc_zeros::<f32>(d * di)?);
            v_out_proj.push(stream.alloc_zeros::<f32>(d * di)?);
            m_d_param.push(stream.alloc_zeros::<f32>(nh)?);
            v_d_param.push(stream.alloc_zeros::<f32>(nh)?);
            m_dt_bias.push(stream.alloc_zeros::<f32>(nh)?);
            v_dt_bias.push(stream.alloc_zeros::<f32>(nh)?);
            m_layer_norm_w.push(stream.alloc_zeros::<f32>(d)?);
            v_layer_norm_w.push(stream.alloc_zeros::<f32>(d)?);
            m_layer_norm_b.push(stream.alloc_zeros::<f32>(d)?);
            v_layer_norm_b.push(stream.alloc_zeros::<f32>(d)?);
            m_b_norm_w.push(stream.alloc_zeros::<f32>(ds)?);
            v_b_norm_w.push(stream.alloc_zeros::<f32>(ds)?);
            m_b_norm_b.push(stream.alloc_zeros::<f32>(ds)?);
            v_b_norm_b.push(stream.alloc_zeros::<f32>(ds)?);
            m_c_norm_w.push(stream.alloc_zeros::<f32>(ds)?);
            v_c_norm_w.push(stream.alloc_zeros::<f32>(ds)?);
            m_c_norm_b.push(stream.alloc_zeros::<f32>(ds)?);
            v_c_norm_b.push(stream.alloc_zeros::<f32>(ds)?);
        }
        let m_fnorm_w = stream.alloc_zeros::<f32>(d)?;
        let v_fnorm_w = stream.alloc_zeros::<f32>(d)?;
        let m_fnorm_b = stream.alloc_zeros::<f32>(d)?;
        let v_fnorm_b = stream.alloc_zeros::<f32>(d)?;
        let m_embed_norm_w = stream.alloc_zeros::<f32>(d)?;
        let v_embed_norm_w = stream.alloc_zeros::<f32>(d)?;
        let m_embed_norm_b = stream.alloc_zeros::<f32>(d)?;
        let v_embed_norm_b = stream.alloc_zeros::<f32>(d)?;

        Ok(Self {
            model,
            train_scratch,
            m_embed, v_embed,
            m_in_proj, v_in_proj,
            m_out_proj, v_out_proj,
            m_fnorm_w, v_fnorm_w,
            m_fnorm_b, v_fnorm_b,
            m_d_param, v_d_param,
            m_dt_bias, v_dt_bias,
            m_layer_norm_w, v_layer_norm_w,
            m_layer_norm_b, v_layer_norm_b,
            m_b_norm_w, v_b_norm_w,
            m_b_norm_b, v_b_norm_b,
            m_c_norm_w, v_c_norm_w,
            m_c_norm_b, v_c_norm_b,
            m_embed_norm_w, v_embed_norm_w,
            m_embed_norm_b, v_embed_norm_b,
            step: 0,
            lr,
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        })
    }

    /// Run one training step. Returns loss.
    pub fn train_step(&mut self, tokens: &[u32], targets: &[u32]) -> Result<f32, Box<dyn Error>> {
        let loss = self.compute_gradients_only(tokens, targets)?;
        self.apply_optimizer_step()?;
        Ok(loss)
    }

    /// Forward + cross-entropy + backward, filling all gradient buffers in
    /// `train_scratch`.  **Does not** advance the AdamW step or mutate weights.
    /// Useful for finite-difference correctness tests and gradient inspection.
    /// Does advance `self.step` so bias-corrected moments are computed for the
    /// subsequent `apply_optimizer_step`.
    pub fn compute_gradients_only(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
    ) -> Result<f32, Box<dyn Error>> {
        self.step += 1;
        let l = tokens.len();
        assert_eq!(targets.len(), l, "targets must have same length as tokens");

        let ptx = self.model.ptx.clone();
        let stream = ptx.stream.clone();
        let d = self.model.d_model;
        let di = self.model.d_inner;
        let ds = self.model.d_state;
        let hd = self.model.headdim;
        let nh = self.model.n_heads;
        let v = self.model.vocab_size;
        let dip = self.model.d_in_proj;

        // -- Upload targets (tokens are uploaded by forward_cached internally)
        let targets_dev: CudaSlice<u32> = stream.memcpy_stod(targets)?;

        // -- Forward with activation cache
        let _logits = self.model.forward_cached(tokens, &mut self.train_scratch)?;

        // -- Zero gradient buffers that accumulate via atomicAdd
        self.zero_gradient_buffers(&stream, l)?;

        // -- Compute loss + d_logits (write into train_scratch)
        {
            // Zero the loss accumulator
            let mut lb = stream.launch_builder(&ptx.k.fill_zero);
            let n_i = 1i32;
            lb.arg(&mut self.train_scratch.loss);
            lb.arg(&n_i);
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
            unsafe { lb.launch(cfg)? };
        }
        {
            let l_i = l as i32;
            let v_i = v as i32;
            let logits = &self.model.scratch.borrow().logits;
            let logits_src: &CudaSlice<f32> = logits;
            let mut lb = stream.launch_builder(&ptx.k.cross_entropy_fwd_bwd);
            lb.arg(logits_src);
            lb.arg(&targets_dev);
            lb.arg(&mut self.train_scratch.d_logits);
            lb.arg(&mut self.train_scratch.loss);
            lb.arg(&l_i);
            lb.arg(&v_i);
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // --- BACKWARD CHAIN ---

        // 1. LM head backward: logits = x_before_head @ embed_w^T
        //    d_x += d_logits @ embed_w       (L, V) × (V, d) → (L, d)
        //    d_embed += d_logits^T @ x_before_head  (V, L) × (L, d) → (V, d)
        launch_matmul_ab(&stream, &ptx,
            &self.train_scratch.d_logits,
            &self.model.embed_w,
            &mut self.train_scratch.d_x,
            l, d, v,
        )?;
        // d_embed accumulates from LM head AND from embed-scatter later.
        launch_matmul_atb(&stream, &ptx,
            &self.train_scratch.d_logits,
            &self.train_scratch.x_before_head,
            &mut self.train_scratch.d_embed,
            v, d, l,
        )?;

        // 2. Final norm backward: d_x flows back through LN
        //    in: d_x (L, d) out: d_x (after LN bwd)
        //    Also accumulates d_fnorm_w and d_fnorm_b
        // We need d_out = current d_x, x = x_before_final_norm, w = final_norm_w
        // Layer-norm bwd writes to a new d_x, so we double-buffer via d_y_out as temp
        {
            let l_i = l as i32;
            let d_i = d as i32;
            let mut lb = stream.launch_builder(&ptx.k.layer_norm_bwd);
            lb.arg(&self.train_scratch.d_x);       // d_out
            lb.arg(&self.train_scratch.x_before_final_norm);  // x
            lb.arg(&self.model.final_norm_w);      // w
            lb.arg(&mut self.train_scratch.d_y_out);  // d_x (reuse d_y_out as temp)
            lb.arg(&mut self.train_scratch.d_fnorm_w);
            lb.arg(&mut self.train_scratch.d_fnorm_b);
            lb.arg(&l_i);
            lb.arg(&d_i);
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }
        // Copy d_y_out → d_x (now d_x holds pre-final-norm gradient)
        launch_copy(&stream, &ptx, &self.train_scratch.d_y_out, &mut self.train_scratch.d_x, l * d)?;

        // 3. Layers backward (reverse order)
        for li in (0..self.model.n_layers).rev() {
            self.layer_backward(&stream, &ptx, li, l)?;
        }

        // 4a. Embed-norm backward: d_x (w.r.t. post-LN residual) flows back
        // through layer_norm to give d_x_pre (w.r.t. pre-LN embedded x) plus
        // accumulated d_embed_norm_w/b. Writes d_x_pre into d_y_out (scratch).
        {
            let l_i = l as i32;
            let d_i = d as i32;
            let mut lb = stream.launch_builder(&ptx.k.layer_norm_bwd);
            lb.arg(&self.train_scratch.d_x);
            lb.arg(&self.train_scratch.x_before_embed_norm);
            lb.arg(&self.model.embed_norm_w);
            lb.arg(&mut self.train_scratch.d_y_out);
            lb.arg(&mut self.train_scratch.d_embed_norm_w);
            lb.arg(&mut self.train_scratch.d_embed_norm_b);
            lb.arg(&l_i);
            lb.arg(&d_i);
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }
        // Copy d_y_out → d_x (d_x now holds pre-embed-norm gradient).
        launch_copy(&stream, &ptx, &self.train_scratch.d_y_out, &mut self.train_scratch.d_x, l * d)?;

        // 4b. Embedding scatter backward: d_embed[token[t], :] += d_x[t, :]
        {
            let tokens_dev = &self.model.scratch.borrow().tokens;
            let l_i = l as i32;
            let d_i = d as i32;
            let v_i = v as i32;
            let mut lb = stream.launch_builder(&ptx.k.embed_scatter_bwd);
            lb.arg(&self.train_scratch.d_x);
            lb.arg(tokens_dev);
            lb.arg(&mut self.train_scratch.d_embed);
            lb.arg(&l_i);
            lb.arg(&d_i);
            lb.arg(&v_i);
            let bd = if d < 256 { d as u32 } else { 256 };
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (bd, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Read back loss (single float) — gradients are now in train_scratch.d_*.
        stream.synchronize()?;
        let loss_host = stream.memcpy_dtov(&self.train_scratch.loss)?;
        let _ = (l, d, di, ds, hd, nh, dip);
        Ok(loss_host[0])
    }

    /// Apply one AdamW step using whatever gradients are currently in
    /// `train_scratch.d_*`.  Called by `train_step` after
    /// `compute_gradients_only`.  Safe to call multiple times for the same
    /// gradients (each call advances moments).  Uses uniform weight decay
    /// across all params (see comment inside).
    pub fn apply_optimizer_step(&mut self) -> Result<(), Box<dyn Error>> {
        let ptx = self.model.ptx.clone();
        let stream = ptx.stream.clone();
        let d = self.model.d_model;
        let di = self.model.d_inner;
        let v = self.model.vocab_size;
        let dip = self.model.d_in_proj;
        let nh = self.model.n_heads;

        let bc1_inv = 1.0f32 / (1.0 - self.beta1.powi(self.step as i32));
        let bc2_inv = 1.0f32 / (1.0 - self.beta2.powi(self.step as i32));

        // Uniform weight decay across all params. PyTorch's "no_decay" idiom
        // requires the FULL gradient chain to be implemented first; with our
        // partial gradients, removing WD on 1-D tensors (d_param especially)
        // lets them grow unboundedly because there's no gradient pullback
        // counterbalancing them. Tested: no_decay + curriculum diverged with
        // loss climbing from 0.48 → 1.75 (parity acc fell to 12%). Until we
        // have the full backward chain, everyone gets the same wd.
        let no_decay_wd = self.weight_decay;

        launch_adamw(&stream, &ptx,
            &mut self.model.embed_w, &self.train_scratch.d_embed,
            &mut self.m_embed, &mut self.v_embed,
            self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, bc1_inv, bc2_inv,
            v * d,
        )?;
        for li in 0..self.model.n_layers {
            let layer = &mut self.model.layers[li];
            launch_adamw(&stream, &ptx,
                &mut layer.in_proj_w, &self.train_scratch.d_in_proj_w[li],
                &mut self.m_in_proj[li], &mut self.v_in_proj[li],
                self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, bc1_inv, bc2_inv,
                dip * d,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.out_proj_w, &self.train_scratch.d_out_proj_w[li],
                &mut self.m_out_proj[li], &mut self.v_out_proj[li],
                self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, bc1_inv, bc2_inv,
                d * di,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.d_param, &self.train_scratch.d_d_param[li],
                &mut self.m_d_param[li], &mut self.v_d_param[li],
                self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
                nh,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.dt_bias, &self.train_scratch.d_dt_bias[li],
                &mut self.m_dt_bias[li], &mut self.v_dt_bias[li],
                self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
                nh,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.layer_norm_w, &self.train_scratch.d_layer_norm_w[li],
                &mut self.m_layer_norm_w[li], &mut self.v_layer_norm_w[li],
                self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
                d,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.layer_norm_b, &self.train_scratch.d_layer_norm_b[li],
                &mut self.m_layer_norm_b[li], &mut self.v_layer_norm_b[li],
                self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
                d,
            )?;
            let ds = self.model.d_state;
            launch_adamw(&stream, &ptx,
                &mut layer.b_norm_w, &self.train_scratch.d_b_norm_w[li],
                &mut self.m_b_norm_w[li], &mut self.v_b_norm_w[li],
                self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
                ds,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.b_norm_b, &self.train_scratch.d_b_norm_b[li],
                &mut self.m_b_norm_b[li], &mut self.v_b_norm_b[li],
                self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
                ds,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.c_norm_w, &self.train_scratch.d_c_norm_w[li],
                &mut self.m_c_norm_w[li], &mut self.v_c_norm_w[li],
                self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
                ds,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.c_norm_b, &self.train_scratch.d_c_norm_b[li],
                &mut self.m_c_norm_b[li], &mut self.v_c_norm_b[li],
                self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
                ds,
            )?;
        }
        launch_adamw(&stream, &ptx,
            &mut self.model.final_norm_w, &self.train_scratch.d_fnorm_w,
            &mut self.m_fnorm_w, &mut self.v_fnorm_w,
            self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
            d,
        )?;
        launch_adamw(&stream, &ptx,
            &mut self.model.final_norm_b, &self.train_scratch.d_fnorm_b,
            &mut self.m_fnorm_b, &mut self.v_fnorm_b,
            self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
            d,
        )?;
        launch_adamw(&stream, &ptx,
            &mut self.model.embed_norm_w, &self.train_scratch.d_embed_norm_w,
            &mut self.m_embed_norm_w, &mut self.v_embed_norm_w,
            self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
            d,
        )?;
        launch_adamw(&stream, &ptx,
            &mut self.model.embed_norm_b, &self.train_scratch.d_embed_norm_b,
            &mut self.m_embed_norm_b, &mut self.v_embed_norm_b,
            self.lr, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv,
            d,
        )?;
        Ok(())
    }

    fn zero_gradient_buffers(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        l: usize,
    ) -> Result<(), Box<dyn Error>> {
        let ptx = self.model.ptx.clone();
        let zero = |buf: &mut CudaSlice<f32>, n: usize| -> Result<(), Box<dyn Error>> {
            let mut lb = stream.launch_builder(&ptx.k.fill_zero);
            lb.arg(buf);
            let n_i = n as i32;
            lb.arg(&n_i);
            let cfg = LaunchConfig {
                grid_dim: ((n as u32 + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
            Ok(())
        };
        let d = self.model.d_model;
        let v = self.model.vocab_size;
        let dip = self.model.d_in_proj;
        let di = self.model.d_inner;

        let nh = self.model.n_heads;
        let ds = self.model.d_state;
        zero(&mut self.train_scratch.d_embed, v * d)?;
        zero(&mut self.train_scratch.d_fnorm_w, d)?;
        zero(&mut self.train_scratch.d_fnorm_b, d)?;
        zero(&mut self.train_scratch.d_embed_norm_w, d)?;
        zero(&mut self.train_scratch.d_embed_norm_b, d)?;
        for li in 0..self.model.n_layers {
            zero(&mut self.train_scratch.d_in_proj_w[li], dip * d)?;
            zero(&mut self.train_scratch.d_out_proj_w[li], d * di)?;
            zero(&mut self.train_scratch.d_d_param[li], nh)?;
            zero(&mut self.train_scratch.d_dt_bias[li], nh)?;
            zero(&mut self.train_scratch.d_layer_norm_w[li], d)?;
            zero(&mut self.train_scratch.d_layer_norm_b[li], d)?;
            zero(&mut self.train_scratch.d_b_norm_w[li], ds)?;
            zero(&mut self.train_scratch.d_b_norm_b[li], ds)?;
            zero(&mut self.train_scratch.d_c_norm_w[li], ds)?;
            zero(&mut self.train_scratch.d_c_norm_b[li], ds)?;
        }
        // d_x, d_y_out, d_y_inner, d_y_pregate, d_scan_inp are overwritten
        // each layer; d_proj needs zeroing because backward atomicAdd's into
        // it. d_bp_post / d_cp_post are also atomic-accumulators but are
        // per-layer scratch, re-zeroed in layer_backward's Step C.
        zero(&mut self.train_scratch.d_proj, l * dip)?;
        let _ = l;
        let _ = ds;
        Ok(())
    }

    /// Backward chain for the b_norm/c_norm + RoPE that sits between
    /// proj[bp/cp_slice] and the post-LN+RoPE bp/cp fed into the SSM scan.
    ///
    /// Inputs (already populated):
    ///   d_bp_post, d_cp_post   — grad w.r.t. post-LN+RoPE bp/cp
    ///   layer_phases[li]       — cached rotation phases
    ///   layer_projs[li]        — holds raw bp/cp slices in-place
    ///
    /// Outputs (accumulated into):
    ///   d_proj[bp_slice], d_proj[cp_slice]   (atomic via scatter_add)
    ///   d_b_norm_w/b[li], d_c_norm_w/b[li]   (atomic via layer_norm_bwd)
    fn bp_cp_norm_bwd(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        ptx: &Arc<PtxContext>,
        li: usize,
        l: usize,
        _d: usize,
        di: usize,
        ds: usize,
        dip: usize,
    ) -> Result<(), Box<dyn Error>> {
        let n_angles = self.model.layers[li].num_rope_angles;
        let l_i = l as i32;
        let ds_i = ds as i32;
        let dip_i = dip as i32;

        // rope_bwd in-place (d_out: post-RoPE → post-LN).
        if n_angles > 0 {
            let phase_off = li * self.train_scratch.layer_phase_stride;
            let phase_len = l * n_angles;
            let phase_view = self.train_scratch.layer_phases.slice(phase_off..phase_off + phase_len);
            let na_i = n_angles as i32;
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (n_angles as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            // bp
            {
                let mut lb = stream.launch_builder(&ptx.k.rope_bwd);
                lb.arg(&mut self.train_scratch.d_bp_post);
                lb.arg(&phase_view);
                lb.arg(&l_i); lb.arg(&ds_i); lb.arg(&na_i);
                unsafe { lb.launch(cfg)? };
            }
            // cp
            {
                let mut lb = stream.launch_builder(&ptx.k.rope_bwd);
                lb.arg(&mut self.train_scratch.d_cp_post);
                lb.arg(&phase_view);
                lb.arg(&l_i); lb.arg(&ds_i); lb.arg(&na_i);
                unsafe { lb.launch(cfg)? };
            }
        }

        // For each of {bp, cp}: gather raw slice from proj → bc_raw_tmp;
        // run layer_norm_bwd(d_out=d_*_post, x=bc_raw_tmp, w=*_norm_w);
        // writes d_x to d_ln_tmp; accumulates d_*_norm_w/b atomically;
        // then scatter-add d_ln_tmp into d_proj[slice_off].
        let proj_off = li * self.train_scratch.layer_proj_stride;
        let proj_view = self.train_scratch.layer_projs.slice(proj_off..proj_off + l * dip);
        let bd_gather = ds.min(256).max(32) as u32;
        let gather_cfg = LaunchConfig {
            grid_dim: (l as u32, 1, 1),
            block_dim: (bd_gather, 1, 1),
            shared_mem_bytes: 0,
        };
        let ln_cfg = LaunchConfig {
            grid_dim: (l as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };

        // ---- bp chain ----
        {
            let bp_off_i = (2 * di) as i32;
            {
                let mut lb = stream.launch_builder(&ptx.k.gather_slice_from_proj);
                lb.arg(&proj_view);
                lb.arg(&mut self.train_scratch.bc_raw_tmp);
                lb.arg(&l_i); lb.arg(&ds_i); lb.arg(&dip_i); lb.arg(&bp_off_i);
                unsafe { lb.launch(gather_cfg)? };
            }
            {
                let mut lb = stream.launch_builder(&ptx.k.layer_norm_bwd);
                lb.arg(&self.train_scratch.d_bp_post);
                lb.arg(&self.train_scratch.bc_raw_tmp);
                lb.arg(&self.model.layers[li].b_norm_w);
                lb.arg(&mut self.train_scratch.d_ln_tmp);
                lb.arg(&mut self.train_scratch.d_b_norm_w[li]);
                lb.arg(&mut self.train_scratch.d_b_norm_b[li]);
                lb.arg(&l_i); lb.arg(&ds_i);
                unsafe { lb.launch(ln_cfg)? };
            }
            {
                let mut lb = stream.launch_builder(&ptx.k.scatter_add_to_proj);
                lb.arg(&self.train_scratch.d_ln_tmp);
                lb.arg(&mut self.train_scratch.d_proj);
                lb.arg(&l_i); lb.arg(&ds_i); lb.arg(&dip_i); lb.arg(&bp_off_i);
                unsafe { lb.launch(gather_cfg)? };
            }
        }

        // ---- cp chain ----
        {
            let cp_off_i = (2 * di + ds) as i32;
            {
                let mut lb = stream.launch_builder(&ptx.k.gather_slice_from_proj);
                lb.arg(&proj_view);
                lb.arg(&mut self.train_scratch.bc_raw_tmp);
                lb.arg(&l_i); lb.arg(&ds_i); lb.arg(&dip_i); lb.arg(&cp_off_i);
                unsafe { lb.launch(gather_cfg)? };
            }
            {
                let mut lb = stream.launch_builder(&ptx.k.layer_norm_bwd);
                lb.arg(&self.train_scratch.d_cp_post);
                lb.arg(&self.train_scratch.bc_raw_tmp);
                lb.arg(&self.model.layers[li].c_norm_w);
                lb.arg(&mut self.train_scratch.d_ln_tmp);
                lb.arg(&mut self.train_scratch.d_c_norm_w[li]);
                lb.arg(&mut self.train_scratch.d_c_norm_b[li]);
                lb.arg(&l_i); lb.arg(&ds_i);
                unsafe { lb.launch(ln_cfg)? };
            }
            {
                let mut lb = stream.launch_builder(&ptx.k.scatter_add_to_proj);
                lb.arg(&self.train_scratch.d_ln_tmp);
                lb.arg(&mut self.train_scratch.d_proj);
                lb.arg(&l_i); lb.arg(&ds_i); lb.arg(&dip_i); lb.arg(&cp_off_i);
                unsafe { lb.launch(gather_cfg)? };
            }
        }

        Ok(())
    }

    fn layer_backward(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        ptx: &Arc<PtxContext>,
        li: usize,
        l: usize,
    ) -> Result<(), Box<dyn Error>> {
        let d = self.model.d_model;
        let di = self.model.d_inner;
        let ds = self.model.d_state;
        let hd = self.model.headdim;
        let nh = self.model.n_heads;
        let dip = self.model.d_in_proj;

        // We need:
        //   d_x (in): gradient of residual stream coming from above
        //   layer.in_proj_w, layer.out_proj_w, layer.d_param: weights
        //   layer.scale: scalar
        //   cached: layer_inputs[li], layer_projs[li], layer_y_inners[li],
        //           layer_cps[li], layer_decays[li], layer_states[li]
        //
        // Produce:
        //   update d_x (new residual gradient, for previous layer)
        //   d_in_proj_w[li], d_out_proj_w[li]

        let scale = self.model.layers[li].scale;

        // Step A: d_y_out = d_x * scale;  save d_residual = d_x (we'll overwrite d_x later)
        // We use d_y_out as the "d_y_out" buffer, d_x as d_residual in place (we'll use it later).
        // For scale multiplication: zero d_y_out, then residual_add(d_y_out, d_x, scale, L*d).
        launch_fill_zero(stream, ptx, &mut self.train_scratch.d_y_out, l * d)?;
        launch_residual_add(stream, ptx, &mut self.train_scratch.d_y_out, &self.train_scratch.d_x, scale, l * d)?;

        // Step B: out_proj backward
        //   d_y_inner = d_y_out @ out_proj_w            (L, d) × (d, di) → (L, di)
        //   d_out_proj_w += d_y_out^T @ y_inner         (d, L) × (L, di) → (d, di)
        let out_proj_w_borrow: &CudaSlice<f32> = &self.model.layers[li].out_proj_w;
        launch_matmul_ab(stream, ptx,
            &self.train_scratch.d_y_out,
            out_proj_w_borrow,
            &mut self.train_scratch.d_y_inner,
            l, di, d,
        )?;
        let y_off = li * self.train_scratch.layer_yinner_stride;
        let y_len = l * di;
        {
            let y_src = self.train_scratch.layer_y_inners.slice(y_off..y_off + y_len);
            let m_i = d as i32;
            let n_i = di as i32;
            let k_i = l as i32;
            let mut lb = stream.launch_builder(&ptx.k.matmul_atb_tiled);
            lb.arg(&self.train_scratch.d_y_out);
            lb.arg(&y_src);
            lb.arg(&mut self.train_scratch.d_out_proj_w[li]);
            lb.arg(&m_i);
            lb.arg(&n_i);
            lb.arg(&k_i);
            let cfg = LaunchConfig {
                grid_dim: ((di as u32 + 15) / 16, (d as u32 + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Step C: Zero per-layer atomic accumulators:
        //   d_proj       (atomic-add'd by gate/scan/bx + scatter from LN bwd)
        //   d_bp_post    (atomic-add'd by bx_bwd)
        //   d_cp_post    (atomic-add'd by ssm_scan_bwd_full)
        launch_fill_zero(stream, ptx, &mut self.train_scratch.d_proj, l * dip)?;
        launch_fill_zero(stream, ptx, &mut self.train_scratch.d_bp_post, l * ds)?;
        launch_fill_zero(stream, ptx, &mut self.train_scratch.d_cp_post, l * ds)?;

        // Step D: gate_bwd writes d_proj[z slice] + d_y_pregate
        {
            let proj_off = li * self.train_scratch.layer_proj_stride;
            let proj_len = l * dip;
            let proj_view = self.train_scratch.layer_projs.slice(proj_off..proj_off + proj_len);
            let y_inner_view = self.train_scratch.layer_y_inners.slice(y_off..y_off + y_len);
            let l_i = l as i32;
            let di_i = di as i32;
            let dip_i = dip as i32;
            let mut lb = stream.launch_builder(&ptx.k.gate_bwd);
            lb.arg(&self.train_scratch.d_y_inner);
            lb.arg(&y_inner_view);
            lb.arg(&proj_view);
            lb.arg(&mut self.train_scratch.d_proj);
            lb.arg(&mut self.train_scratch.d_y_pregate);
            lb.arg(&l_i);
            lb.arg(&di_i);
            lb.arg(&dip_i);
            let total = (l * di) as u32;
            let cfg = LaunchConfig {
                grid_dim: ((total + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Step E: ssm_scan_bwd_full — atomic writes to d_proj[x_skip], d_proj[cp],
        // plus d_scan_inp, d_decay, d_dt_from_inp, atomic d_d_param.
        {
            let proj_off = li * self.train_scratch.layer_proj_stride;
            let cp_off = li * self.train_scratch.layer_cp_stride;
            let dc_off = li * self.train_scratch.layer_decay_stride;
            let st_off = li * self.train_scratch.layer_states_stride;
            let proj_view = self.train_scratch.layer_projs.slice(proj_off..proj_off + l * dip);
            let bp_view = self.train_scratch.layer_bps.slice(cp_off..cp_off + l * ds);
            let cp_view = self.train_scratch.layer_cps.slice(cp_off..cp_off + l * ds);
            let dc_view = self.train_scratch.layer_decays.slice(dc_off..dc_off + l * nh);
            let dt_view = self.train_scratch.layer_dts.slice(dc_off..dc_off + l * nh);
            let st_view = self.train_scratch.layer_states.slice(st_off..st_off + (l + 1) * nh * hd * ds);

            let l_i = l as i32;
            let h_i = nh as i32;
            let hd_i = hd as i32;
            let ds_i = ds as i32;
            let di_i = di as i32;
            let dip_i = dip as i32;
            let d_param_ref: &CudaSlice<f32> = &self.model.layers[li].d_param;
            let mut lb = stream.launch_builder(&ptx.k.ssm_scan_bwd_full);
            lb.arg(&self.train_scratch.d_y_pregate);
            lb.arg(&proj_view);
            lb.arg(&bp_view);
            lb.arg(&cp_view);
            lb.arg(&dc_view);
            lb.arg(&dt_view);
            lb.arg(&st_view);
            lb.arg(d_param_ref);
            lb.arg(&mut self.train_scratch.d_proj);
            lb.arg(&mut self.train_scratch.d_cp_post);
            lb.arg(&mut self.train_scratch.d_scan_inp);
            lb.arg(&mut self.train_scratch.d_decay);
            lb.arg(&mut self.train_scratch.d_d_param[li]);
            lb.arg(&mut self.train_scratch.d_dt_from_inp);
            lb.arg(&l_i); lb.arg(&h_i); lb.arg(&hd_i); lb.arg(&ds_i); lb.arg(&di_i); lb.arg(&dip_i);
            let cfg = LaunchConfig {
                grid_dim: (nh as u32, 1, 1),
                block_dim: ((hd * ds) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Step E2: ssm_param_grads — writes d_proj[dt_off] and d_proj[a_off]
        // (overwrite — currently 0), plus atomic d_dt_bias[h]. Needs
        // d_decay + d_dt_from_inp (both filled by ssm_scan_bwd_full; dt_from_inp
        // is held at 0 by design for stability, so this is the decay-path
        // d_dt only). FD-verified: before re-enable, d_dt_bias analytical was
        // 0.0 while FD showed ~2e-4; after this, dt_bias should PASS the gate.
        {
            let proj_off = li * self.train_scratch.layer_proj_stride;
            let dc_off = li * self.train_scratch.layer_decay_stride;
            let proj_view = self.train_scratch.layer_projs.slice(proj_off..proj_off + l * dip);
            let dt_view = self.train_scratch.layer_dts.slice(dc_off..dc_off + l * nh);
            let dc_view = self.train_scratch.layer_decays.slice(dc_off..dc_off + l * nh);
            let dt_bias_ref: &CudaSlice<f32> = &self.model.layers[li].dt_bias;
            let l_i = l as i32;
            let h_i = nh as i32;
            let dip_i = dip as i32;
            let di_i = di as i32;
            let ds_i = ds as i32;
            let mut lb = stream.launch_builder(&ptx.k.ssm_param_grads);
            lb.arg(&proj_view);
            lb.arg(dt_bias_ref);
            lb.arg(&dt_view);
            lb.arg(&dc_view);
            lb.arg(&self.train_scratch.d_decay);
            lb.arg(&self.train_scratch.d_dt_from_inp);
            lb.arg(&mut self.train_scratch.d_proj);
            lb.arg(&mut self.train_scratch.d_dt_bias[li]);
            lb.arg(&l_i); lb.arg(&h_i); lb.arg(&dip_i); lb.arg(&di_i); lb.arg(&ds_i);
            let block_x = nh.max(32) as u32;
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }
        let _ = (dip, ds, di);

        // Step F: bx_bwd — accumulates d_proj[x_off] (atomic) and d_bp_post
        // (atomic). Uses post-LN+RoPE bp from cache, not raw proj[bp_off].
        {
            let proj_off = li * self.train_scratch.layer_proj_stride;
            let cp_off = li * self.train_scratch.layer_cp_stride;
            let proj_view = self.train_scratch.layer_projs.slice(proj_off..proj_off + l * dip);
            let bp_view = self.train_scratch.layer_bps.slice(cp_off..cp_off + l * ds);
            let l_i = l as i32;
            let h_i = nh as i32;
            let hd_i = hd as i32;
            let ds_i = ds as i32;
            let di_i = di as i32;
            let dip_i = dip as i32;
            let dt_bias_ref: &CudaSlice<f32> = &self.model.layers[li].dt_bias;
            let mut lb = stream.launch_builder(&ptx.k.bx_bwd);
            lb.arg(&self.train_scratch.d_scan_inp);
            lb.arg(&proj_view);
            lb.arg(&bp_view);
            lb.arg(dt_bias_ref);
            lb.arg(&mut self.train_scratch.d_proj);
            lb.arg(&mut self.train_scratch.d_bp_post);
            lb.arg(&l_i); lb.arg(&h_i); lb.arg(&hd_i); lb.arg(&ds_i); lb.arg(&di_i); lb.arg(&dip_i);
            let smem_bytes = (hd * ds) as u32 * 4;
            let cfg = LaunchConfig {
                grid_dim: (nh as u32, 1, 1),
                block_dim: ((hd * ds) as u32, 1, 1),
                shared_mem_bytes: smem_bytes,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Step F2: Chain d_{bp,cp}_post → rope_bwd → layer_norm_bwd →
        // scatter-add into d_proj[{bp,cp}_slice]. Fixes the remaining
        // in_proj_w[cp/bp_row] FAILs. Also accumulates d_{b,c}_norm_w/b
        // (populated but not yet AdamW'd until verified).
        self.bp_cp_norm_bwd(&stream, &ptx, li, l, d, di, ds, dip)?;

        // Step G: in_proj backward
        //   d_in_proj_w[li] = d_proj^T @ x_normed (dip, L) × (L, d) → (dip, d)
        //   d_x_normed    = d_proj @ in_proj_w   (L, dip) × (dip, d) → (L, d)
        // (x_normed = post-pre-LN input; the forward feeds x_normed into in_proj,
        // so the matmul gradient is against x_normed, NOT the raw residual x_in.)
        {
            let input_off = li * self.train_scratch.layer_input_stride;
            let input_len = l * d;
            let x_normed_view = self.train_scratch.layer_x_normed.slice(input_off..input_off + input_len);
            let m_i = dip as i32;
            let n_i = d as i32;
            let k_i = l as i32;
            let mut lb = stream.launch_builder(&ptx.k.matmul_atb_tiled);
            lb.arg(&self.train_scratch.d_proj);
            lb.arg(&x_normed_view);
            lb.arg(&mut self.train_scratch.d_in_proj_w[li]);
            lb.arg(&m_i);
            lb.arg(&n_i);
            lb.arg(&k_i);
            let cfg = LaunchConfig {
                grid_dim: ((d as u32 + 15) / 16, (dip as u32 + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }
        // d_x_normed = d_proj @ in_proj_w (stored in d_y_out as scratch).
        let in_proj_w_ref: &CudaSlice<f32> = &self.model.layers[li].in_proj_w;
        launch_matmul_ab(stream, ptx,
            &self.train_scratch.d_proj,
            in_proj_w_ref,
            &mut self.train_scratch.d_y_out,
            l, d, dip,
        )?;

        // Step G2: pre-layer LN backward — d_x_normed → d_x_input (plus
        // d_layer_norm_w, d_layer_norm_b).  Uses the cached RAW x_input
        // (pre-LN) that the LN fwd consumed.  Writes d_x_input into
        // d_y_inner as a scratch target (anything L*d that's free right
        // now will do; d_y_inner is L*di which holds L*d fine since di>=d).
        {
            let input_off = li * self.train_scratch.layer_input_stride;
            let input_len = l * d;
            let x_in_view = self.train_scratch.layer_inputs.slice(input_off..input_off + input_len);
            let l_i = l as i32;
            let d_i = d as i32;
            let mut lb = stream.launch_builder(&ptx.k.layer_norm_bwd);
            lb.arg(&self.train_scratch.d_y_out);       // d_out (= d_x_normed)
            lb.arg(&x_in_view);                        // x (pre-LN)
            lb.arg(&self.model.layers[li].layer_norm_w);
            lb.arg(&mut self.train_scratch.d_y_inner); // d_x (writes d_x_input)
            lb.arg(&mut self.train_scratch.d_layer_norm_w[li]);
            lb.arg(&mut self.train_scratch.d_layer_norm_b[li]);
            lb.arg(&l_i);
            lb.arg(&d_i);
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Step H: residual combine — d_x += scale * d_x_input
        launch_residual_add(stream, ptx, &mut self.train_scratch.d_x, &self.train_scratch.d_y_inner, scale, l * d)?;

        Ok(())
    }
}

// --- launch helpers ---

fn launch_copy(
    stream: &Arc<cudarc::driver::CudaStream>,
    ptx: &Arc<PtxContext>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), Box<dyn Error>> {
    let n_i = n as i32;
    let mut lb = stream.launch_builder(&ptx.k.copy_f32);
    lb.arg(src);
    lb.arg(dst);
    lb.arg(&n_i);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32 + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg)? };
    Ok(())
}

fn launch_fill_zero(
    stream: &Arc<cudarc::driver::CudaStream>,
    ptx: &Arc<PtxContext>,
    buf: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), Box<dyn Error>> {
    let n_i = n as i32;
    let mut lb = stream.launch_builder(&ptx.k.fill_zero);
    lb.arg(buf);
    lb.arg(&n_i);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32 + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg)? };
    Ok(())
}

fn launch_residual_add(
    stream: &Arc<cudarc::driver::CudaStream>,
    ptx: &Arc<PtxContext>,
    acc: &mut CudaSlice<f32>,
    other: &CudaSlice<f32>,
    scale: f32,
    n: usize,
) -> Result<(), Box<dyn Error>> {
    let n_i = n as i32;
    let mut lb = stream.launch_builder(&ptx.k.residual_add);
    lb.arg(acc);
    lb.arg(other);
    lb.arg(&scale);
    lb.arg(&n_i);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32 + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg)? };
    Ok(())
}

fn launch_matmul_ab(
    stream: &Arc<cudarc::driver::CudaStream>,
    ptx: &Arc<PtxContext>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
    m: usize, n: usize, k: usize,
) -> Result<(), Box<dyn Error>> {
    let m_i = m as i32;
    let n_i = n as i32;
    let k_i = k as i32;
    let mut lb = stream.launch_builder(&ptx.k.matmul_ab_tiled);
    lb.arg(a);
    lb.arg(b);
    lb.arg(c);
    lb.arg(&m_i);
    lb.arg(&n_i);
    lb.arg(&k_i);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg)? };
    Ok(())
}

// atb variants: A(K,M), B(K,N), C(M,N); C overwritten (non-accumulate)
fn launch_matmul_atb(
    stream: &Arc<cudarc::driver::CudaStream>,
    ptx: &Arc<PtxContext>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
    m: usize, n: usize, k: usize,
) -> Result<(), Box<dyn Error>> {
    let m_i = m as i32;
    let n_i = n as i32;
    let k_i = k as i32;
    let mut lb = stream.launch_builder(&ptx.k.matmul_atb_tiled);
    lb.arg(a);
    lb.arg(b);
    lb.arg(c);
    lb.arg(&m_i);
    lb.arg(&n_i);
    lb.arg(&k_i);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg)? };
    Ok(())
}

// atb-accumulate: C += A^T @ B. Implementation: since matmul_atb_tiled overwrites,
// we need an accumulating variant. Simpler: compute into a temp, then add.
// For efficiency, zero C first, compute into C, then the caller knows it's been reset.
// But we USE this for d_in_proj_w and d_out_proj_w which should ACCUMULATE across
// calls within a step... actually they are set to 0 per train_step, then written
// ONCE per layer via this function, so non-accumulating overwrite is fine.

fn launch_adamw(
    stream: &Arc<cudarc::driver::CudaStream>,
    ptx: &Arc<PtxContext>,
    params: &mut CudaSlice<f32>,
    grads: &CudaSlice<f32>,
    m: &mut CudaSlice<f32>,
    v: &mut CudaSlice<f32>,
    lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
    bc1_inv: f32, bc2_inv: f32,
    n: usize,
) -> Result<(), Box<dyn Error>> {
    let n_i = n as i32;
    let mut lb = stream.launch_builder(&ptx.k.adamw_step);
    lb.arg(params);
    lb.arg(grads);
    lb.arg(m);
    lb.arg(v);
    lb.arg(&lr); lb.arg(&beta1); lb.arg(&beta2); lb.arg(&eps); lb.arg(&wd);
    lb.arg(&bc1_inv); lb.arg(&bc2_inv);
    lb.arg(&n_i);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32 + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg)? };
    Ok(())
}
