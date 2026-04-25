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
    // Host-side AdamW state for per-layer scale (tiny scalar — no point on GPU).
    pub m_scale: Vec<f32>,
    pub v_scale: Vec<f32>,

    pub step: u32,
    pub lr: f32,                 // peak / steady-state LR
    pub warmup_steps: u32,       // linear LR ramp from 0 → lr over this many steps
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
            m_scale: vec![0.0; nl],
            v_scale: vec![0.0; nl],
            step: 0,
            lr,
            warmup_steps: 200,   // matches specialist_trainer scale; CLI override below
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

    /// Accumulating variant: forward + backward like compute_gradients_only,
    /// but does NOT zero the gradient buffers first. Use to sum gradients
    /// across a mini-batch of sequences before a single AdamW step.
    /// Caller must invoke `zero_gradients_only()` before the first sample of
    /// a batch and `apply_optimizer_step()` after the last. The returned
    /// loss is the per-sample loss (not averaged).
    ///
    /// This sidesteps the per-sample-AdamW thrashing that hurts mixed-length
    /// training (variable-length parity bounces because each sequence's
    /// gradient drives a separate Adam step; with accumulation, the optimizer
    /// sees the average gradient across the batch — matching PyTorch
    /// semantics).
    pub fn accumulate_gradients(&mut self, tokens: &[u32], targets: &[u32]) -> Result<f32, Box<dyn Error>> {
        self.compute_gradients_with_zero(tokens, targets, /*do_zero=*/false)
    }

    pub fn zero_gradients_only(&mut self) -> Result<(), Box<dyn Error>> {
        let stream = self.model.stream.clone();
        let ptx = self.model.ptx.clone();
        let l = self.train_scratch.max_seq;
        self.zero_gradient_buffers(&stream, l)?;
        // Also zero the GPU-side loss accumulator. The cross_entropy kernel
        // uses atomicAdd, so without this each accumulate_gradients call
        // would see a growing cumulative sum from prior batches.
        let mut lb = stream.launch_builder(&ptx.k.fill_zero);
        let n_i = 1i32;
        lb.arg(&mut self.train_scratch.loss);
        lb.arg(&n_i);
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        unsafe { lb.launch(cfg)? };
        Ok(())
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
        self.compute_gradients_with_zero(tokens, targets, /*do_zero=*/true)
    }

    fn compute_gradients_with_zero(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        do_zero: bool,
    ) -> Result<f32, Box<dyn Error>> {
        // self.step counts AdamW steps, NOT compute_gradients calls — moved
        // into apply_optimizer_step so accumulation across a mini-batch counts
        // as one step. (Old behaviour: step++ here, which double-counted when
        // accumulating across a batch.)
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

        // -- Zero gradient + loss buffers (compute_gradients_only path only).
        //    The accumulate path passes do_zero=false and the caller is
        //    expected to call zero_gradients_only() before the first sample.
        if do_zero {
            self.zero_gradient_buffers(&stream, l)?;
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
            // Count unmasked positions (target != MASK sentinel) so the kernel
            // can normalise by 1/n_active, matching PyTorch's masked CE.
            let n_active = targets.iter().filter(|&&t| t != u32::MAX).count();
            let n_active_inv: f32 = if n_active == 0 { 0.0 } else { 1.0 / n_active as f32 };
            let logits = &self.model.scratch.borrow().logits;
            let logits_src: &CudaSlice<f32> = logits;
            let mut lb = stream.launch_builder(&ptx.k.cross_entropy_fwd_bwd);
            lb.arg(logits_src);
            lb.arg(&targets_dev);
            lb.arg(&mut self.train_scratch.d_logits);
            lb.arg(&mut self.train_scratch.loss);
            lb.arg(&l_i);
            lb.arg(&v_i);
            lb.arg(&n_active_inv);
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
        self.apply_optimizer_step_scaled(1.0)
    }

    /// AdamW step with an extra gradient multiplier folded into the global-norm
    /// clip factor. Use with gradient accumulation: after summing grads across
    /// B samples, pass `1.0 / B` to average — produces the same effective step
    /// as PyTorch's per-batch backward (one AdamW step per batch, not B).
    pub fn apply_optimizer_step_scaled(&mut self, extra_g_mul: f32) -> Result<(), Box<dyn Error>> {
        // step is incremented here (not in compute_gradients_with_zero) so
        // gradient accumulation across a mini-batch counts as ONE step.
        self.step += 1;
        let ptx = self.model.ptx.clone();
        let stream = ptx.stream.clone();
        let d = self.model.d_model;
        let di = self.model.d_inner;
        let v = self.model.vocab_size;
        let dip = self.model.d_in_proj;
        let nh = self.model.n_heads;

        let bc1_inv = 1.0f32 / (1.0 - self.beta1.powi(self.step as i32));
        let bc2_inv = 1.0f32 / (1.0 - self.beta2.powi(self.step as i32));

        // Linear LR warmup: gradient magnitudes are unstable in the first
        // few hundred steps because Adam's `v` moment is still tiny, which
        // amplifies updates. Without warmup, scalars like `scale` overshoot
        // through zero on step 1 and the optimizer can't recover. We linearly
        // ramp lr_eff from 0 → self.lr over `warmup_steps`.
        let lr_eff = if self.step < self.warmup_steps {
            self.lr * (self.step as f32) / (self.warmup_steps as f32)
        } else {
            self.lr
        };

        // Uniform weight decay across all params. PyTorch's "no_decay" idiom
        // requires the FULL gradient chain to be implemented first; with our
        // partial gradients, removing WD on 1-D tensors (d_param especially)
        // lets them grow unboundedly because there's no gradient pullback
        // counterbalancing them. Tested: no_decay + curriculum diverged with
        // loss climbing from 0.48 → 1.75 (parity acc fell to 12%). Until we
        // have the full backward chain, everyone gets the same wd.
        let no_decay_wd = self.weight_decay;

        // -- Global-norm gradient clipping (matches PyTorch's
        //    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)). Without
        //    this, AdamW applies updates element-wise without seeing the total
        //    norm, and small parameters like the per-layer `scale` get over-
        //    corrected toward zero on the very first step (the gradient says
        //    "SSM output is random noise, decrease scale" — the optimizer
        //    obeys aggressively).  PyTorch's global clip implicitly damps each
        //    parameter's update by 1/||all_grads||_2 when the total norm is
        //    above 1.0.
        const MAX_NORM: f32 = 1.0;
        let mut sumsq_dev = stream.alloc_zeros::<f32>(1)?;
        // Accumulate sum of squares across every gradient tensor we'll AdamW.
        let acc_sumsq = |stream: &Arc<cudarc::driver::CudaStream>,
                          ptx: &Arc<PtxContext>,
                          buf: &CudaSlice<f32>,
                          n: usize,
                          out: &mut CudaSlice<f32>|
         -> Result<(), Box<dyn Error>> {
            let mut lb = stream.launch_builder(&ptx.k.reduce_dot_f32);
            let n_i = n as i32;
            lb.arg(buf);
            lb.arg(buf);
            lb.arg(out);
            lb.arg(&n_i);
            let cfg = LaunchConfig {
                grid_dim: ((n as u32 + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
            Ok(())
        };
        acc_sumsq(&stream, &ptx, &self.train_scratch.d_embed, v * d, &mut sumsq_dev)?;
        acc_sumsq(&stream, &ptx, &self.train_scratch.d_fnorm_w, d, &mut sumsq_dev)?;
        acc_sumsq(&stream, &ptx, &self.train_scratch.d_fnorm_b, d, &mut sumsq_dev)?;
        acc_sumsq(&stream, &ptx, &self.train_scratch.d_embed_norm_w, d, &mut sumsq_dev)?;
        acc_sumsq(&stream, &ptx, &self.train_scratch.d_embed_norm_b, d, &mut sumsq_dev)?;
        let ds_local = self.model.d_state;
        for li in 0..self.model.n_layers {
            acc_sumsq(&stream, &ptx, &self.train_scratch.d_in_proj_w[li], dip * d, &mut sumsq_dev)?;
            acc_sumsq(&stream, &ptx, &self.train_scratch.d_out_proj_w[li], d * di, &mut sumsq_dev)?;
            acc_sumsq(&stream, &ptx, &self.train_scratch.d_d_param[li], nh, &mut sumsq_dev)?;
            acc_sumsq(&stream, &ptx, &self.train_scratch.d_dt_bias[li], nh, &mut sumsq_dev)?;
            acc_sumsq(&stream, &ptx, &self.train_scratch.d_layer_norm_w[li], d, &mut sumsq_dev)?;
            acc_sumsq(&stream, &ptx, &self.train_scratch.d_layer_norm_b[li], d, &mut sumsq_dev)?;
            acc_sumsq(&stream, &ptx, &self.train_scratch.d_b_norm_w[li], ds_local, &mut sumsq_dev)?;
            acc_sumsq(&stream, &ptx, &self.train_scratch.d_b_norm_b[li], ds_local, &mut sumsq_dev)?;
            acc_sumsq(&stream, &ptx, &self.train_scratch.d_c_norm_w[li], ds_local, &mut sumsq_dev)?;
            acc_sumsq(&stream, &ptx, &self.train_scratch.d_c_norm_b[li], ds_local, &mut sumsq_dev)?;
            acc_sumsq(&stream, &ptx, &self.train_scratch.d_scale[li], 1, &mut sumsq_dev)?;
        }
        let sumsq = stream.memcpy_dtov(&sumsq_dev)?[0];
        let total_norm = sumsq.max(0.0).sqrt();
        let clip_mul: f32 = if total_norm.is_finite() && total_norm > MAX_NORM {
            MAX_NORM / total_norm
        } else if !total_norm.is_finite() {
            0.0
        } else {
            1.0
        };
        let g_mul = clip_mul * extra_g_mul;

        launch_adamw(&stream, &ptx,
            &mut self.model.embed_w, &self.train_scratch.d_embed,
            &mut self.m_embed, &mut self.v_embed,
            lr_eff, self.beta1, self.beta2, self.eps, self.weight_decay, bc1_inv, bc2_inv, g_mul,
            v * d,
        )?;
        for li in 0..self.model.n_layers {
            let layer = &mut self.model.layers[li];
            launch_adamw(&stream, &ptx,
                &mut layer.in_proj_w, &self.train_scratch.d_in_proj_w[li],
                &mut self.m_in_proj[li], &mut self.v_in_proj[li],
                lr_eff, self.beta1, self.beta2, self.eps, self.weight_decay, bc1_inv, bc2_inv, g_mul,
                dip * d,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.out_proj_w, &self.train_scratch.d_out_proj_w[li],
                &mut self.m_out_proj[li], &mut self.v_out_proj[li],
                lr_eff, self.beta1, self.beta2, self.eps, self.weight_decay, bc1_inv, bc2_inv, g_mul,
                d * di,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.d_param, &self.train_scratch.d_d_param[li],
                &mut self.m_d_param[li], &mut self.v_d_param[li],
                lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
                nh,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.dt_bias, &self.train_scratch.d_dt_bias[li],
                &mut self.m_dt_bias[li], &mut self.v_dt_bias[li],
                lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
                nh,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.layer_norm_w, &self.train_scratch.d_layer_norm_w[li],
                &mut self.m_layer_norm_w[li], &mut self.v_layer_norm_w[li],
                lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
                d,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.layer_norm_b, &self.train_scratch.d_layer_norm_b[li],
                &mut self.m_layer_norm_b[li], &mut self.v_layer_norm_b[li],
                lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
                d,
            )?;
            let ds = self.model.d_state;
            launch_adamw(&stream, &ptx,
                &mut layer.b_norm_w, &self.train_scratch.d_b_norm_w[li],
                &mut self.m_b_norm_w[li], &mut self.v_b_norm_w[li],
                lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
                ds,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.b_norm_b, &self.train_scratch.d_b_norm_b[li],
                &mut self.m_b_norm_b[li], &mut self.v_b_norm_b[li],
                lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
                ds,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.c_norm_w, &self.train_scratch.d_c_norm_w[li],
                &mut self.m_c_norm_w[li], &mut self.v_c_norm_w[li],
                lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
                ds,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.c_norm_b, &self.train_scratch.d_c_norm_b[li],
                &mut self.m_c_norm_b[li], &mut self.v_c_norm_b[li],
                lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
                ds,
            )?;
        }
        launch_adamw(&stream, &ptx,
            &mut self.model.final_norm_w, &self.train_scratch.d_fnorm_w,
            &mut self.m_fnorm_w, &mut self.v_fnorm_w,
            lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
            d,
        )?;
        launch_adamw(&stream, &ptx,
            &mut self.model.final_norm_b, &self.train_scratch.d_fnorm_b,
            &mut self.m_fnorm_b, &mut self.v_fnorm_b,
            lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
            d,
        )?;
        launch_adamw(&stream, &ptx,
            &mut self.model.embed_norm_w, &self.train_scratch.d_embed_norm_w,
            &mut self.m_embed_norm_w, &mut self.v_embed_norm_w,
            lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
            d,
        )?;
        launch_adamw(&stream, &ptx,
            &mut self.model.embed_norm_b, &self.train_scratch.d_embed_norm_b,
            &mut self.m_embed_norm_b, &mut self.v_embed_norm_b,
            lr_eff, self.beta1, self.beta2, self.eps, no_decay_wd, bc1_inv, bc2_inv, g_mul,
            d,
        )?;

        // Host-side AdamW for per-layer scale. Read d_scale (single f32) from
        // device, apply the update in Rust, write back to layer.scale. One
        // D→H copy per layer per step — cheap (8 bytes × n_layers).
        for li in 0..self.model.n_layers {
            let g = stream.memcpy_dtov(&self.train_scratch.d_scale[li])?[0];
            let g = if g.is_finite() { g.clamp(-1.0, 1.0) * g_mul } else { 0.0 };
            let p = self.model.layers[li].scale;
            // no_decay for scalar: wd=0 semantically, but match uniform wd for now
            let p = p * (1.0 - lr_eff * no_decay_wd);
            let mi = self.beta1 * self.m_scale[li] + (1.0 - self.beta1) * g;
            let vi = self.beta2 * self.v_scale[li] + (1.0 - self.beta2) * g * g;
            self.m_scale[li] = mi;
            self.v_scale[li] = vi;
            let m_hat = mi * bc1_inv;
            let v_hat = vi * bc2_inv;
            let p_new = p - lr_eff * m_hat / (v_hat.sqrt() + self.eps);
            self.model.layers[li].scale = p_new;
        }
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

    /// Phase-chain part of the bp/cp backward: rope_bwd in-place on both
    /// d_bp_post and d_cp_post, accumulating d_phase for use by the
    /// reverse-cumsum + phase_step_bwd that follows.  Splits out of the
    /// full bp_cp_norm_bwd so the caller can interleave the phase chain
    /// (which feeds d_dt_mean into ssm_param_grads) before running the
    /// layer-norm half of the chain.
    fn bp_cp_rope_bwd(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        ptx: &Arc<PtxContext>,
        li: usize,
        l: usize,
        ds: usize,
    ) -> Result<(), Box<dyn Error>> {
        let n_angles = self.model.layers[li].num_rope_angles;
        if n_angles == 0 { return Ok(()); }
        let l_i = l as i32;
        let ds_i = ds as i32;
        let na_i = n_angles as i32;
        let phase_off = li * self.train_scratch.layer_phase_stride;
        let phase_len = l * n_angles;
        let phase_view = self.train_scratch.layer_phases.slice(phase_off..phase_off + phase_len);
        let cp_off = li * self.train_scratch.layer_cp_stride;
        let bp_view = self.train_scratch.layer_bps.slice(cp_off..cp_off + l * ds);
        let cp_view = self.train_scratch.layer_cps.slice(cp_off..cp_off + l * ds);
        let cfg = LaunchConfig {
            grid_dim: (l as u32, 1, 1),
            block_dim: (n_angles as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // bp
        {
            let mut lb = stream.launch_builder(&ptx.k.rope_bwd);
            lb.arg(&mut self.train_scratch.d_bp_post);
            lb.arg(&bp_view);
            lb.arg(&phase_view);
            lb.arg(&mut self.train_scratch.d_phase);
            lb.arg(&l_i); lb.arg(&ds_i); lb.arg(&na_i);
            unsafe { lb.launch(cfg)? };
        }
        // cp
        {
            let mut lb = stream.launch_builder(&ptx.k.rope_bwd);
            lb.arg(&mut self.train_scratch.d_cp_post);
            lb.arg(&cp_view);
            lb.arg(&phase_view);
            lb.arg(&mut self.train_scratch.d_phase);
            lb.arg(&l_i); lb.arg(&ds_i); lb.arg(&na_i);
            unsafe { lb.launch(cfg)? };
        }
        Ok(())
    }

    /// LayerNorm half of the bp/cp chain: d_{bp,cp}_post (now post-rope_bwd,
    /// i.e. holding d_normed) → ln_bwd → scatter into d_proj[bp/cp_slice].
    /// Accumulates d_b_norm_w/b and d_c_norm_w/b along the way.
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
        let l_i = l as i32;
        let ds_i = ds as i32;
        let dip_i = dip as i32;
        // rope_bwd half is now run separately by bp_cp_rope_bwd before this
        // function — d_bp_post and d_cp_post already hold post-rope (i.e.
        // pre-RoPE = post-LN-only) gradients.

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

        // Step A0: compute d_scale = <d_x_in_layer, y_post_out_proj>.
        // The forward is x_new = x_in + scale * y_post, so ∂L/∂scale =
        // Σ_ti d_x_new[t,i] * y_post[t,i]. d_x here is d_x_new (gradient
        // flowing in from above; it becomes d_x_in after we propagate through
        // this layer). Zero the accumulator, then reduce_dot.
        launch_fill_zero(stream, ptx, &mut self.train_scratch.d_scale[li], 1)?;
        {
            let yp_off = li * self.train_scratch.layer_input_stride;
            let y_post_view = self.train_scratch.layer_y_post.slice(yp_off..yp_off + l * d);
            let n_i = (l * d) as i32;
            let mut lb = stream.launch_builder(&ptx.k.reduce_dot_f32);
            lb.arg(&self.train_scratch.d_x);
            lb.arg(&y_post_view);
            lb.arg(&mut self.train_scratch.d_scale[li]);
            lb.arg(&n_i);
            let grid_x = (((l * d) as u32) + 255) / 256;
            let cfg = LaunchConfig {
                grid_dim: (grid_x, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

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
        //   d_trap_pre   (atomic-add'd by bx_bwd)
        launch_fill_zero(stream, ptx, &mut self.train_scratch.d_proj, l * dip)?;
        launch_fill_zero(stream, ptx, &mut self.train_scratch.d_bp_post, l * ds)?;
        launch_fill_zero(stream, ptx, &mut self.train_scratch.d_cp_post, l * ds)?;
        launch_fill_zero(stream, ptx, &mut self.train_scratch.d_trap_pre, l * nh)?;
        let n_angles = self.model.layers[li].num_rope_angles;
        if n_angles > 0 {
            launch_fill_zero(stream, ptx, &mut self.train_scratch.d_phase, l * n_angles)?;
        }
        launch_fill_zero(stream, ptx, &mut self.train_scratch.d_dt_mean, l)?;

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

        let _ = (dip, ds, di);
        // ssm_param_grads has moved — it now runs AFTER the phase chain so it
        // can incorporate d_dt_mean's contribution to d_dt[t,h]. See Step E2'
        // below (after F1.5 / F-phase).

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
            lb.arg(&mut self.train_scratch.d_trap_pre);
            lb.arg(&l_i); lb.arg(&h_i); lb.arg(&hd_i); lb.arg(&ds_i); lb.arg(&di_i); lb.arg(&dip_i);
            let smem_bytes = (hd * ds) as u32 * 4;
            let cfg = LaunchConfig {
                grid_dim: (nh as u32, 1, 1),
                block_dim: ((hd * ds) as u32, 1, 1),
                shared_mem_bytes: smem_bytes,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Step F1.5: Convert d_trap_pre (accumulated by bx_bwd) into
        // d_proj[tr_off + h] by applying the sigmoid' factor trap·(1-trap).
        // Without this, in_proj_w[trap_rows] never receives a gradient and
        // the model can't learn how much to mix in bx_prev across timesteps.
        {
            let dc_off = li * self.train_scratch.layer_decay_stride;
            let trap_view = self.train_scratch.layer_traps.slice(dc_off..dc_off + l * nh);
            let l_i = l as i32;
            let h_i = nh as i32;
            let di_i = di as i32;
            let ds_i = ds as i32;
            let dip_i = dip as i32;
            let mut lb = stream.launch_builder(&ptx.k.trap_to_proj_bwd);
            lb.arg(&self.train_scratch.d_trap_pre);
            lb.arg(&trap_view);
            lb.arg(&mut self.train_scratch.d_proj);
            lb.arg(&l_i); lb.arg(&h_i); lb.arg(&di_i); lb.arg(&ds_i); lb.arg(&dip_i);
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (nh.max(32) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Step F-phase-1: rope_bwd half — rotates d_bp_post / d_cp_post by
        // -phase in place AND accumulates d_phase from BOTH chains.  After
        // this, d_bp_post / d_cp_post hold post-LN-only gradients (ready
        // for the LN bwd half), and d_phase has the full phase-side gradient.
        self.bp_cp_rope_bwd(&stream, &ptx, li, l, ds)?;

        // Step F-phase-2: reverse_cumsum d_phase → d_phase_step.
        let n_angles_li = self.model.layers[li].num_rope_angles;
        if n_angles_li > 0 {
            let l_i = l as i32;
            let k_i = n_angles_li as i32;
            let mut lb = stream.launch_builder(&ptx.k.reverse_cumsum_f32);
            lb.arg(&self.train_scratch.d_phase);
            lb.arg(&mut self.train_scratch.d_phase_step);
            lb.arg(&l_i); lb.arg(&k_i);
            let cfg = LaunchConfig {
                grid_dim: (n_angles_li as u32, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Step F-phase-3: phase_step_bwd writes d_proj[angles_off + k] and
        // accumulates d_dt_mean[t] = Σ_k d_phase_step[t,k] · angles[t,k].
        if n_angles_li > 0 {
            let proj_off = li * self.train_scratch.layer_proj_stride;
            let proj_view = self.train_scratch.layer_projs.slice(proj_off..proj_off + l * dip);
            let dtm_off = li * self.train_scratch.max_seq;
            let dtm_view = self.train_scratch.layer_dt_means.slice(dtm_off..dtm_off + l);
            let l_i = l as i32;
            let k_i = n_angles_li as i32;
            let h_i = nh as i32;
            let di_i = di as i32;
            let ds_i = ds as i32;
            let dip_i = dip as i32;
            let mut lb = stream.launch_builder(&ptx.k.phase_step_bwd);
            lb.arg(&self.train_scratch.d_phase_step);
            lb.arg(&proj_view);
            lb.arg(&dtm_view);
            lb.arg(&mut self.train_scratch.d_proj);
            lb.arg(&mut self.train_scratch.d_dt_mean);
            lb.arg(&l_i); lb.arg(&k_i); lb.arg(&h_i); lb.arg(&di_i); lb.arg(&ds_i); lb.arg(&dip_i);
            let block_x = (n_angles_li.max(32)).next_power_of_two().min(64) as u32;
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Step E2 (moved): ssm_param_grads — now sees d_dt_mean from the
        // phase chain so the dt gradient picks up the phase-DT_mean term.
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
            lb.arg(&self.train_scratch.d_dt_mean);
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

        // Step F2: LN-bwd half of the bp/cp chain — d_bp_post / d_cp_post
        // (now holding post-LN-only gradients after rope_bwd) → ln_bwd →
        // scatter into d_proj[bp/cp_slice].
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
    g_mul: f32,
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
    lb.arg(&g_mul);
    lb.arg(&n_i);
    let cfg = LaunchConfig {
        grid_dim: ((n as u32 + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg)? };
    Ok(())
}
