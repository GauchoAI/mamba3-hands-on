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
            &ctx, max_seq, nl, d, ds, di, nh, hd, dip, v,
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
        for _ in 0..nl {
            m_in_proj.push(stream.alloc_zeros::<f32>(dip * d)?);
            v_in_proj.push(stream.alloc_zeros::<f32>(dip * d)?);
            m_out_proj.push(stream.alloc_zeros::<f32>(d * di)?);
            v_out_proj.push(stream.alloc_zeros::<f32>(d * di)?);
            m_d_param.push(stream.alloc_zeros::<f32>(nh)?);
            v_d_param.push(stream.alloc_zeros::<f32>(nh)?);
            m_dt_bias.push(stream.alloc_zeros::<f32>(nh)?);
            v_dt_bias.push(stream.alloc_zeros::<f32>(nh)?);
        }
        let m_fnorm_w = stream.alloc_zeros::<f32>(d)?;
        let v_fnorm_w = stream.alloc_zeros::<f32>(d)?;
        let m_fnorm_b = stream.alloc_zeros::<f32>(d)?;
        let v_fnorm_b = stream.alloc_zeros::<f32>(d)?;

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

        // 4. Embedding scatter backward: d_embed[token[t], :] += d_x[t, :]
        //    Note: embed_norm is skipped in CPU reference — we do the same.
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

        // --- OPTIMIZER ---
        let bc1_inv = 1.0f32 / (1.0 - self.beta1.powi(self.step as i32));
        let bc2_inv = 1.0f32 / (1.0 - self.beta2.powi(self.step as i32));

        // Embed
        launch_adamw(&stream, &ptx,
            &mut self.model.embed_w, &self.train_scratch.d_embed,
            &mut self.m_embed, &mut self.v_embed,
            self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, bc1_inv, bc2_inv,
            v * d,
        )?;
        // Per-layer
        let nh = self.model.n_heads;
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
                self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, bc1_inv, bc2_inv,
                nh,
            )?;
            launch_adamw(&stream, &ptx,
                &mut layer.dt_bias, &self.train_scratch.d_dt_bias[li],
                &mut self.m_dt_bias[li], &mut self.v_dt_bias[li],
                self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, bc1_inv, bc2_inv,
                nh,
            )?;
        }
        // Final norm
        launch_adamw(&stream, &ptx,
            &mut self.model.final_norm_w, &self.train_scratch.d_fnorm_w,
            &mut self.m_fnorm_w, &mut self.v_fnorm_w,
            self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, bc1_inv, bc2_inv,
            d,
        )?;
        launch_adamw(&stream, &ptx,
            &mut self.model.final_norm_b, &self.train_scratch.d_fnorm_b,
            &mut self.m_fnorm_b, &mut self.v_fnorm_b,
            self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, bc1_inv, bc2_inv,
            d,
        )?;

        // Read back loss (single float)
        stream.synchronize()?;
        let loss_host = stream.memcpy_dtov(&self.train_scratch.loss)?;
        let _ = (l, d, di, ds, hd, nh, dip);
        Ok(loss_host[0])
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
        zero(&mut self.train_scratch.d_embed, v * d)?;
        zero(&mut self.train_scratch.d_fnorm_w, d)?;
        zero(&mut self.train_scratch.d_fnorm_b, d)?;
        for li in 0..self.model.n_layers {
            zero(&mut self.train_scratch.d_in_proj_w[li], dip * d)?;
            zero(&mut self.train_scratch.d_out_proj_w[li], d * di)?;
            zero(&mut self.train_scratch.d_d_param[li], nh)?;
            zero(&mut self.train_scratch.d_dt_bias[li], nh)?;
        }
        // d_x, d_y_out, d_y_inner, d_y_pregate, d_scan_inp, d_proj are fully
        // overwritten each layer; only d_proj needs zeroing as backward
        // atomicAdd's into its slices. But we zero the entire used range.
        zero(&mut self.train_scratch.d_proj, l * dip)?;
        let _ = l;
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

        // Step C: Zero d_proj (it accumulates via atomicAdd from gate/scan/bx)
        launch_fill_zero(stream, ptx, &mut self.train_scratch.d_proj, l * dip)?;

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
            lb.arg(&cp_view);
            lb.arg(&dc_view);
            lb.arg(&dt_view);
            lb.arg(&st_view);
            lb.arg(d_param_ref);
            lb.arg(&mut self.train_scratch.d_proj);
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

        // Step E.5: ssm_param_grads — d_dt_bias + d_proj[dt_off/a_off]
        // Uses decay-path d_dt only (inp-path d_dt_from_inp is zeroed by
        // ssm_scan_bwd_full for numerical stability).
        {
            let proj_off = li * self.train_scratch.layer_proj_stride;
            let proj_view = self.train_scratch.layer_projs.slice(proj_off..proj_off + l * dip);
            let dc_off = li * self.train_scratch.layer_decay_stride;
            let dc_view = self.train_scratch.layer_decays.slice(dc_off..dc_off + l * nh);
            let dt_view = self.train_scratch.layer_dts.slice(dc_off..dc_off + l * nh);
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
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (nh.max(32) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Step F: bx_bwd — atomic adds into d_proj[bp] and d_proj[x]
        {
            let proj_off = li * self.train_scratch.layer_proj_stride;
            let proj_view = self.train_scratch.layer_projs.slice(proj_off..proj_off + l * dip);
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
            lb.arg(dt_bias_ref);
            lb.arg(&mut self.train_scratch.d_proj);
            lb.arg(&l_i); lb.arg(&h_i); lb.arg(&hd_i); lb.arg(&ds_i); lb.arg(&di_i); lb.arg(&dip_i);
            let smem_bytes = (hd * ds) as u32 * 4;
            let cfg = LaunchConfig {
                grid_dim: (nh as u32, 1, 1),
                block_dim: ((hd * ds) as u32, 1, 1),
                shared_mem_bytes: smem_bytes,
            };
            unsafe { lb.launch(cfg)? };
        }

        // Step G: in_proj backward
        //   d_in_proj_w[li] = d_proj^T @ x_in   (dip, L) × (L, d) → (dip, d)
        //   d_x_from_layer = d_proj @ in_proj_w (L, dip) × (dip, d) → (L, d)
        {
            let input_off = li * self.train_scratch.layer_input_stride;
            let input_len = l * d;
            let x_in_view = self.train_scratch.layer_inputs.slice(input_off..input_off + input_len);
            let m_i = dip as i32;
            let n_i = d as i32;
            let k_i = l as i32;
            let mut lb = stream.launch_builder(&ptx.k.matmul_atb_tiled);
            lb.arg(&self.train_scratch.d_proj);
            lb.arg(&x_in_view);
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
        // d_x_from_layer into d_y_out (reusing buffer)
        let in_proj_w_ref: &CudaSlice<f32> = &self.model.layers[li].in_proj_w;
        launch_matmul_ab(stream, ptx,
            &self.train_scratch.d_proj,
            in_proj_w_ref,
            &mut self.train_scratch.d_y_out,  // temp
            l, d, dip,
        )?;

        // Step H: residual combine: d_x = d_residual + scale * d_x_from_layer
        // d_x already holds d_residual (the input). Add scale * d_y_out to it.
        launch_residual_add(stream, ptx, &mut self.train_scratch.d_x, &self.train_scratch.d_y_out, scale, l * d)?;

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
