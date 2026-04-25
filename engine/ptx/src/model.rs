//! PtxModel — forward inference using hand-written PTX kernels on NVIDIA GPU.
//!
//! v1: per-op kernels, bit-exact vs CPU reference. Correctness first,
//! creative iteration in v2+.

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaGraph, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use mamba3_engine::model::Mamba3Model;
use std::cell::RefCell;
use std::error::Error;
use std::sync::Arc;

use crate::runtime::PtxContext;
use crate::scratch::PtxScratch;
use crate::train_scratch::TrainScratch;

pub struct PtxLayer {
    pub in_proj_w: CudaSlice<f32>,
    pub out_proj_w: CudaSlice<f32>,
    pub dt_bias: CudaSlice<f32>,
    pub d_param: CudaSlice<f32>,
    pub b_norm_w: CudaSlice<f32>,
    pub b_norm_b: CudaSlice<f32>,
    pub c_norm_w: CudaSlice<f32>,
    pub c_norm_b: CudaSlice<f32>,
    pub layer_norm_w: CudaSlice<f32>,
    pub layer_norm_b: CudaSlice<f32>,
    pub scale: f32,
    pub d_in_proj: usize,
    pub num_rope_angles: usize,
}

pub struct PtxModel {
    pub ptx: Arc<PtxContext>,

    pub d_model: usize,
    pub d_state: usize,
    pub d_inner: usize,
    pub headdim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub num_rope_angles: usize,
    pub d_in_proj: usize,

    pub embed_w: CudaSlice<f32>,
    pub embed_norm_w: CudaSlice<f32>,
    pub embed_norm_b: CudaSlice<f32>,
    pub layers: Vec<PtxLayer>,
    pub final_norm_w: CudaSlice<f32>,
    pub final_norm_b: CudaSlice<f32>,

    // Concatenated layer weights for the persistent kernel path.
    // Assumes uniform config across layers (verified in from_cpu).
    pub l_in_proj_w: CudaSlice<f32>,   // (n_layers * dip * d,)
    pub l_out_proj_w: CudaSlice<f32>,  // (n_layers * d * di,)
    pub l_dt_bias: CudaSlice<f32>,     // (n_layers * H,)
    pub l_d_param: CudaSlice<f32>,     // (n_layers * H,)
    pub l_b_norm_w: CudaSlice<f32>,    // (n_layers * ds,)
    pub l_b_norm_b: CudaSlice<f32>,    // (n_layers * ds,)
    pub l_c_norm_w: CudaSlice<f32>,    // (n_layers * ds,)
    pub l_c_norm_b: CudaSlice<f32>,    // (n_layers * ds,)
    pub l_ln_w: CudaSlice<f32>,        // (n_layers * d,)
    pub l_ln_b: CudaSlice<f32>,        // (n_layers * d,)
    pub l_scale: CudaSlice<f32>,       // (n_layers,)

    pub scratch: RefCell<PtxScratch>,
}

impl PtxModel {
    pub fn from_cpu(
        model: &Mamba3Model,
        ptx: Arc<PtxContext>,
        max_seq: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let stream = ptx.stream.clone();
        let embed_w = stream.memcpy_stod(&model.embed_w)?;
        let embed_norm_w = stream.memcpy_stod(&model.embed_norm_w)?;
        let embed_norm_b = stream.memcpy_stod(&model.embed_norm_b)?;
        let final_norm_w = stream.memcpy_stod(&model.final_norm_w)?;
        let final_norm_b = stream.memcpy_stod(&model.final_norm_b)?;

        let mut layers = Vec::new();
        let mut max_dip = 0usize;
        let mut max_n_angles = 0usize;

        // Build concatenated layer weight vectors on the host, then upload once.
        let first = &model.layers[0];
        let uniform_dip = first.d_in_proj;
        let uniform_n_angles = first.num_rope_angles;
        let mut cat_in_proj_w = Vec::with_capacity(model.n_layers * uniform_dip * model.d_model);
        let mut cat_out_proj_w = Vec::with_capacity(model.n_layers * model.d_model * model.d_inner);
        let mut cat_dt_bias = Vec::with_capacity(model.n_layers * model.n_heads);
        let mut cat_d_param = Vec::with_capacity(model.n_layers * model.n_heads);
        let mut cat_b_norm_w = Vec::with_capacity(model.n_layers * model.d_state);
        let mut cat_b_norm_b = Vec::with_capacity(model.n_layers * model.d_state);
        let mut cat_c_norm_w = Vec::with_capacity(model.n_layers * model.d_state);
        let mut cat_c_norm_b = Vec::with_capacity(model.n_layers * model.d_state);
        let mut cat_ln_w = Vec::with_capacity(model.n_layers * model.d_model);
        let mut cat_ln_b = Vec::with_capacity(model.n_layers * model.d_model);
        let mut cat_scale = Vec::with_capacity(model.n_layers);

        for lw in &model.layers {
            max_dip = max_dip.max(lw.d_in_proj);
            max_n_angles = max_n_angles.max(lw.num_rope_angles);
            // Persistent kernel assumes uniform config across layers
            assert_eq!(lw.d_in_proj, uniform_dip, "persistent kernel: non-uniform d_in_proj");
            assert_eq!(lw.num_rope_angles, uniform_n_angles, "persistent kernel: non-uniform num_rope_angles");

            cat_in_proj_w.extend_from_slice(&lw.in_proj_w);
            cat_out_proj_w.extend_from_slice(&lw.out_proj_w);
            cat_dt_bias.extend_from_slice(&lw.dt_bias);
            cat_d_param.extend_from_slice(&lw.d_param);
            cat_b_norm_w.extend_from_slice(&lw.b_norm_w);
            cat_b_norm_b.extend_from_slice(&lw.b_norm_b);
            cat_c_norm_w.extend_from_slice(&lw.c_norm_w);
            cat_c_norm_b.extend_from_slice(&lw.c_norm_b);
            cat_ln_w.extend_from_slice(&lw.layer_norm_w);
            cat_ln_b.extend_from_slice(&lw.layer_norm_b);
            cat_scale.push(lw.scale);

            layers.push(PtxLayer {
                in_proj_w: stream.memcpy_stod(&lw.in_proj_w)?,
                out_proj_w: stream.memcpy_stod(&lw.out_proj_w)?,
                dt_bias: stream.memcpy_stod(&lw.dt_bias)?,
                d_param: stream.memcpy_stod(&lw.d_param)?,
                b_norm_w: stream.memcpy_stod(&lw.b_norm_w)?,
                b_norm_b: stream.memcpy_stod(&lw.b_norm_b)?,
                c_norm_w: stream.memcpy_stod(&lw.c_norm_w)?,
                c_norm_b: stream.memcpy_stod(&lw.c_norm_b)?,
                layer_norm_w: stream.memcpy_stod(&lw.layer_norm_w)?,
                layer_norm_b: stream.memcpy_stod(&lw.layer_norm_b)?,
                scale: lw.scale,
                d_in_proj: lw.d_in_proj,
                num_rope_angles: lw.num_rope_angles,
            });
        }

        let l_in_proj_w = stream.memcpy_stod(&cat_in_proj_w)?;
        let l_out_proj_w = stream.memcpy_stod(&cat_out_proj_w)?;
        let l_dt_bias = stream.memcpy_stod(&cat_dt_bias)?;
        let l_d_param = stream.memcpy_stod(&cat_d_param)?;
        let l_b_norm_w = stream.memcpy_stod(&cat_b_norm_w)?;
        let l_b_norm_b = stream.memcpy_stod(&cat_b_norm_b)?;
        let l_c_norm_w = stream.memcpy_stod(&cat_c_norm_w)?;
        let l_c_norm_b = stream.memcpy_stod(&cat_c_norm_b)?;
        let l_ln_w = stream.memcpy_stod(&cat_ln_w)?;
        let l_ln_b = stream.memcpy_stod(&cat_ln_b)?;
        let l_scale = stream.memcpy_stod(&cat_scale)?;

        let scratch = PtxScratch::new(
            &ptx,
            max_seq,
            model.d_model,
            model.d_state,
            model.d_inner,
            model.n_heads,
            max_dip,
            max_n_angles,
            model.vocab_size,
        )?;

        Ok(Self {
            ptx,
            d_model: model.d_model,
            d_state: model.d_state,
            d_inner: model.d_inner,
            headdim: model.headdim,
            n_heads: model.n_heads,
            n_layers: model.n_layers,
            vocab_size: model.vocab_size,
            num_rope_angles: uniform_n_angles,
            d_in_proj: uniform_dip,
            embed_w,
            embed_norm_w,
            embed_norm_b,
            layers,
            final_norm_w,
            final_norm_b,
            l_in_proj_w,
            l_out_proj_w,
            l_dt_bias,
            l_d_param,
            l_b_norm_w,
            l_b_norm_b,
            l_c_norm_w,
            l_c_norm_b,
            l_ln_w,
            l_ln_b,
            l_scale,
            scratch: RefCell::new(scratch),
        })
    }

    /// Block size for the persistent kernel. 512 threads → 2 SSM heads in
    /// parallel per iteration for the run_length_next model (hd*ds=256); 4
    /// serial passes for 8 heads. We tried 1024 (4 heads parallel): it builds
    /// with __launch_bounds__(1024,1) but forces registers into local memory
    /// (HBM), making the kernel 2x slower from register spills. 512 is the
    /// sweet spot for this model — full register pressure, no spilling.
    pub fn persistent_block_size(&self) -> u32 {
        512
    }

    /// Shared-memory footprint (in bytes) of the persistent kernel for a given
    /// sequence length.  Must match the SMEM layout in kernels.cu.
    pub fn persistent_smem_bytes(&self, l: usize) -> u32 {
        let d = self.d_model;
        let di = self.d_inner;
        let ds = self.d_state;
        let h = self.n_heads;
        let na = self.num_rope_angles;
        let dip = self.d_in_proj;
        let block = self.persistent_block_size() as usize;
        let floats = l * d               // x
            + l * d                      // x_normed
            + l * dip                    // proj
            + l * h                      // dt
            + l * h                      // decay
            + l * h                      // trap
            + l                          // dt_mean
            + l * na                     // phase
            + l * ds                     // bp
            + l * ds                     // cp
            + l * di                     // y_inner
            + l * d                      // y_out
            + block;                     // reduce_buf sized to block
        (floats * 4) as u32
    }

    /// Forward pass that also persists per-layer activations for backward.
    /// Writes into `train_scratch.layer_*` and `train_scratch.x_before_head`.
    /// Produces logits bit-identical to `forward()`.
    pub fn forward_cached(
        &self,
        tokens: &[u32],
        train_scratch: &mut TrainScratch,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        let l = tokens.len();
        let d = self.d_model;
        let di = self.d_inner;
        let ds = self.d_state;
        let hd = self.headdim;
        let nh = self.n_heads;
        let v = self.vocab_size;

        self.upload_tokens(&stream, tokens)?;

        let mut scratch_ref = self.scratch.borrow_mut();
        let scratch: &mut PtxScratch = &mut *scratch_ref;

        // 1. Embed gather into scratch.x
        {
            let l_i = l as i32;
            let d_i = d as i32;
            let v_i = v as i32;
            let mut lb = stream.launch_builder(&self.ptx.k.embed_gather);
            lb.arg(&scratch.tokens);
            lb.arg(&self.embed_w);
            lb.arg(&mut scratch.x);
            lb.arg(&l_i);
            lb.arg(&d_i);
            lb.arg(&v_i);
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }
        // 1b. Cache pre-embed-norm x (needed by embed_norm backward).
        {
            let n_copy = l * d;
            let src = scratch.x.slice(0..n_copy);
            let mut dst = train_scratch.x_before_embed_norm.slice_mut(0..n_copy);
            let n_i = n_copy as i32;
            let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
            lb.arg(&src);
            lb.arg(&mut dst);
            lb.arg(&n_i);
            let cfg = LaunchConfig {
                grid_dim: ((n_copy as u32 + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }
        // 2. Embed norm in-place
        launch_layer_norm(
            &stream, &self.ptx,
            &mut scratch.x, &self.embed_norm_w, &self.embed_norm_b, l, d,
        )?;

        // 3. Layers
        for (li, layer) in self.layers.iter().enumerate() {
            // 3a. Save x INTO this layer → train_scratch.layer_inputs[li]
            {
                let off = li * train_scratch.layer_input_stride;
                let n_copy = l * d;
                let src = scratch.x.slice(0..n_copy);
                let mut dst = train_scratch.layer_inputs.slice_mut(off..off + n_copy);
                let n_i = n_copy as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
                lb.arg(&src);
                lb.arg(&mut dst);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: ((n_copy as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // 3b. Copy x -> x_normed, pre-norm
            {
                let n_i = (l * d) as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
                lb.arg(&scratch.x);
                lb.arg(&mut scratch.x_normed);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: (((l * d) as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
            launch_layer_norm(
                &stream, &self.ptx,
                &mut scratch.x_normed, &layer.layer_norm_w, &layer.layer_norm_b, l, d,
            )?;

            // 3b.5. Save x_normed — used by d_in_proj_w = d_proj^T @ x_normed
            // and by layer_norm_bwd in the backward pass.
            {
                let off = li * train_scratch.layer_input_stride;
                let n_copy = l * d;
                let src = scratch.x_normed.slice(0..n_copy);
                let mut dst = train_scratch.layer_x_normed.slice_mut(off..off + n_copy);
                let n_i = n_copy as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
                lb.arg(&src);
                lb.arg(&mut dst);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: ((n_copy as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // 3c. in_proj matmul → DIRECTLY into train_scratch.layer_projs[li]
            let dip = layer.d_in_proj;
            let proj_off = li * train_scratch.layer_proj_stride;
            let proj_len = l * dip;
            let mut proj_dst = train_scratch.layer_projs.slice_mut(proj_off..proj_off + proj_len);
            {
                let m_i = l as i32;
                let n_i = dip as i32;
                let k_i = d as i32;
                let func = if d % 16 == 0 { &self.ptx.k.matmul_t_tiled } else { &self.ptx.k.matmul_t };
                let mut lb = stream.launch_builder(func);
                lb.arg(&scratch.x_normed);
                lb.arg(&layer.in_proj_w);
                lb.arg(&mut proj_dst);
                lb.arg(&m_i);
                lb.arg(&n_i);
                lb.arg(&k_i);
                let cfg = LaunchConfig {
                    grid_dim: ((dip as u32 + 15) / 16, (l as u32 + 15) / 16, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
            // From here on we read proj FROM train_scratch.layer_projs[li].
            let proj_view = train_scratch.layer_projs.slice(proj_off..proj_off + proj_len);

            // 3d. Fused SSM params + dt_mean
            {
                let l_i = l as i32;
                let h_i = nh as i32;
                let dip_i = dip as i32;
                let di_i = di as i32;
                let ds_i = ds as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.compute_ssm_params_and_dt_mean);
                lb.arg(&proj_view);
                lb.arg(&layer.dt_bias);
                lb.arg(&mut scratch.dt);
                lb.arg(&mut scratch.decay);
                lb.arg(&mut scratch.trap);
                lb.arg(&mut scratch.dt_mean);
                lb.arg(&l_i); lb.arg(&h_i); lb.arg(&dip_i); lb.arg(&di_i); lb.arg(&ds_i);
                let cfg = LaunchConfig {
                    grid_dim: (l as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // 3e. phase (sequential)
            let n_angles = layer.num_rope_angles;
            {
                let l_i = l as i32;
                let na_i = n_angles as i32;
                let dip_i = dip as i32;
                let di_i = di as i32;
                let ds_i = ds as i32;
                let h_i = nh as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.compute_phase);
                lb.arg(&proj_view);
                lb.arg(&scratch.dt_mean);
                lb.arg(&mut scratch.phase);
                lb.arg(&l_i); lb.arg(&na_i); lb.arg(&dip_i); lb.arg(&di_i); lb.arg(&ds_i); lb.arg(&h_i);
                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (n_angles.max(1) as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
            // 3e.5 — cache phase per layer for rope_bwd.
            if n_angles > 0 {
                let p_off = li * train_scratch.layer_phase_stride;
                let p_len = l * n_angles;
                let src = scratch.phase.slice(0..p_len);
                let mut dst = train_scratch.layer_phases.slice_mut(p_off..p_off + p_len);
                let n_i = p_len as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
                lb.arg(&src);
                lb.arg(&mut dst);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: ((p_len as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // 3f. extract bp, cp from proj_view
            {
                let l_i = l as i32;
                let ds_i = ds as i32;
                let di_i = di as i32;
                let dip_i = dip as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.extract_bp_cp);
                lb.arg(&proj_view);
                lb.arg(&mut scratch.bp);
                lb.arg(&mut scratch.cp);
                lb.arg(&l_i); lb.arg(&ds_i); lb.arg(&di_i); lb.arg(&dip_i);
                let cfg = LaunchConfig {
                    grid_dim: (l as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
            launch_layer_norm(&stream, &self.ptx, &mut scratch.bp, &layer.b_norm_w, &layer.b_norm_b, l, ds)?;
            launch_layer_norm(&stream, &self.ptx, &mut scratch.cp, &layer.c_norm_w, &layer.c_norm_b, l, ds)?;
            launch_apply_rope(&stream, &self.ptx, &mut scratch.bp, &scratch.phase, l, ds, n_angles)?;
            launch_apply_rope(&stream, &self.ptx, &mut scratch.cp, &scratch.phase, l, ds, n_angles)?;

            // 3g. Save bp, cp (post LN+RoPE) and decay for backward.
            // bp is needed so the adjoint scan can reconstruct `blended`
            // without dividing by dt (avoids the dt≈0.05-at-init blowup).
            {
                let bp_off = li * train_scratch.layer_cp_stride;
                let bp_len = l * ds;
                let mut dst = train_scratch.layer_bps.slice_mut(bp_off..bp_off + bp_len);
                let src = scratch.bp.slice(0..bp_len);
                let n_i = bp_len as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
                lb.arg(&src);
                lb.arg(&mut dst);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: ((bp_len as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
            {
                let cp_off = li * train_scratch.layer_cp_stride;
                let cp_len = l * ds;
                let mut dst = train_scratch.layer_cps.slice_mut(cp_off..cp_off + cp_len);
                let src = scratch.cp.slice(0..cp_len);
                let n_i = cp_len as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
                lb.arg(&src);
                lb.arg(&mut dst);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: ((cp_len as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
            {
                let d_off = li * train_scratch.layer_decay_stride;
                let d_len = l * nh;
                let mut dst = train_scratch.layer_decays.slice_mut(d_off..d_off + d_len);
                let src = scratch.decay.slice(0..d_len);
                let n_i = d_len as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
                lb.arg(&src);
                lb.arg(&mut dst);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: ((d_len as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
            // Also save dt per layer (needed by ssm_scan_bwd_full / ssm_param_grads)
            {
                let d_off = li * train_scratch.layer_decay_stride;
                let d_len = l * nh;
                let mut dst = train_scratch.layer_dts.slice_mut(d_off..d_off + d_len);
                let src = scratch.dt.slice(0..d_len);
                let n_i = d_len as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
                lb.arg(&src);
                lb.arg(&mut dst);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: ((d_len as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // 3h. ssm_scan_cached — writes y_inner AND states to per-layer buffers
            let y_off = li * train_scratch.layer_yinner_stride;
            let y_len = l * nh * hd; // == l * di
            let states_off = li * train_scratch.layer_states_stride;
            let states_len = (l + 1) * nh * hd * ds;
            let mut y_dst = train_scratch.layer_y_inners.slice_mut(y_off..y_off + y_len);
            let mut states_dst = train_scratch.layer_states.slice_mut(states_off..states_off + states_len);
            {
                let l_i = l as i32;
                let h_i = nh as i32;
                let hd_i = hd as i32;
                let ds_i = ds as i32;
                let di_i = di as i32;
                let dip_i = dip as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.ssm_scan_cached);
                lb.arg(&proj_view);
                lb.arg(&scratch.bp);
                lb.arg(&scratch.cp);
                lb.arg(&scratch.dt);
                lb.arg(&scratch.decay);
                lb.arg(&scratch.trap);
                lb.arg(&layer.d_param);
                lb.arg(&mut y_dst);
                lb.arg(&mut states_dst);
                lb.arg(&l_i); lb.arg(&h_i); lb.arg(&hd_i); lb.arg(&ds_i); lb.arg(&di_i); lb.arg(&dip_i);
                let smem_bytes = ((hd * ds) + hd) as u32 * 4;
                let cfg = LaunchConfig {
                    grid_dim: (nh as u32, 1, 1),
                    block_dim: ((hd * ds) as u32, 1, 1),
                    shared_mem_bytes: smem_bytes,
                };
                unsafe { lb.launch(cfg)? };
            }

            // 3i. out_proj matmul: reads layer_y_inners[li], writes scratch.y_out
            {
                let m_i = l as i32;
                let n_i = d as i32;
                let k_i = di as i32;
                let func = if di % 16 == 0 { &self.ptx.k.matmul_t_tiled } else { &self.ptx.k.matmul_t };
                let y_src = train_scratch.layer_y_inners.slice(y_off..y_off + y_len);
                let mut lb = stream.launch_builder(func);
                lb.arg(&y_src);
                lb.arg(&layer.out_proj_w);
                lb.arg(&mut scratch.y_out);
                lb.arg(&m_i);
                lb.arg(&n_i);
                lb.arg(&k_i);
                let cfg = LaunchConfig {
                    grid_dim: ((d as u32 + 15) / 16, (l as u32 + 15) / 16, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // 3i.5 — cache y_post_out_proj per layer for d_scale = <d_x_in, y>.
            {
                let yp_off = li * train_scratch.layer_input_stride;
                let n_copy = l * d;
                let src = scratch.y_out.slice(0..n_copy);
                let mut dst = train_scratch.layer_y_post.slice_mut(yp_off..yp_off + n_copy);
                let n_i = n_copy as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
                lb.arg(&src);
                lb.arg(&mut dst);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: ((n_copy as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
            // 3j. residual: scratch.x += scale * y_out
            {
                let n_i = (l * d) as i32;
                let scale_f = layer.scale;
                let mut lb = stream.launch_builder(&self.ptx.k.residual_add);
                lb.arg(&mut scratch.x);
                lb.arg(&scratch.y_out);
                lb.arg(&scale_f);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: (((l * d) as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
        }

        // 4. Save x BEFORE final norm (for backward)
        {
            let n_copy = l * d;
            let src = scratch.x.slice(0..n_copy);
            let mut dst = train_scratch.x_before_final_norm.slice_mut(0..n_copy);
            let n_i = n_copy as i32;
            let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
            lb.arg(&src);
            lb.arg(&mut dst);
            lb.arg(&n_i);
            let cfg = LaunchConfig {
                grid_dim: ((n_copy as u32 + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // 4b. Final norm (in-place on scratch.x)
        launch_layer_norm(
            &stream, &self.ptx,
            &mut scratch.x, &self.final_norm_w, &self.final_norm_b, l, d,
        )?;

        // 4c. Save x AFTER final norm (for LM head backward)
        {
            let n_copy = l * d;
            let src = scratch.x.slice(0..n_copy);
            let mut dst = train_scratch.x_before_head.slice_mut(0..n_copy);
            let n_i = n_copy as i32;
            let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
            lb.arg(&src);
            lb.arg(&mut dst);
            lb.arg(&n_i);
            let cfg = LaunchConfig {
                grid_dim: ((n_copy as u32 + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // 5. LM head
        launch_matmul_t(
            &stream, &self.ptx,
            &scratch.x, &self.embed_w, &mut scratch.logits, l, v, d,
        )?;

        stream.synchronize()?;
        let logits_slice = scratch.logits.slice(0..(l * v));
        Ok(stream.memcpy_dtov(&logits_slice)?)
    }

    /// Forward pass via the cooperative-groups multi-block persistent kernel.
    /// ONE kernel launch, COOP_BLOCKS blocks (default 64), grid-sync between
    /// phases. HBM scratch buffers for cross-block communication. This is the
    /// design that wins under GPU contention: one scheduling event instead of
    /// 45.
    pub fn forward_coop(&self, tokens: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        self.upload_tokens(&stream, tokens)?;
        self.launch_coop(&stream, tokens.len())?;
        stream.synchronize()?;

        let l = tokens.len();
        let scratch = self.scratch.borrow();
        let logits_slice = scratch.logits.slice(0..(l * self.vocab_size));
        let host = stream.memcpy_dtov(&logits_slice)?;
        Ok(host)
    }

    /// Launch the cooperative kernel. No upload, no sync, no readback —
    /// suitable for graph capture.
    pub fn launch_coop(
        &self,
        stream: &Arc<CudaStream>,
        l: usize,
    ) -> Result<(), Box<dyn Error>> {
        let mut scratch_ref = self.scratch.borrow_mut();
        let scratch: &mut PtxScratch = &mut *scratch_ref;

        let l_i = l as i32;
        let n_layers_i = self.n_layers as i32;
        let d_i = self.d_model as i32;
        let di_i = self.d_inner as i32;
        let ds_i = self.d_state as i32;
        let h_i = self.n_heads as i32;
        let hd_i = self.headdim as i32;
        let na_i = self.num_rope_angles as i32;
        let v_i = self.vocab_size as i32;
        let dip_i = self.d_in_proj as i32;

        let mut lb = stream.launch_builder(&self.ptx.k.mamba3_forward_coop);
        lb.arg(&scratch.tokens);
        lb.arg(&self.embed_w);
        lb.arg(&self.embed_norm_w);
        lb.arg(&self.embed_norm_b);
        lb.arg(&self.l_in_proj_w);
        lb.arg(&self.l_out_proj_w);
        lb.arg(&self.l_dt_bias);
        lb.arg(&self.l_d_param);
        lb.arg(&self.l_b_norm_w);
        lb.arg(&self.l_b_norm_b);
        lb.arg(&self.l_c_norm_w);
        lb.arg(&self.l_c_norm_b);
        lb.arg(&self.l_ln_w);
        lb.arg(&self.l_ln_b);
        lb.arg(&self.l_scale);
        lb.arg(&self.final_norm_w);
        lb.arg(&self.final_norm_b);
        lb.arg(&mut scratch.x);
        lb.arg(&mut scratch.x_normed);
        lb.arg(&mut scratch.proj);
        lb.arg(&mut scratch.dt);
        lb.arg(&mut scratch.decay);
        lb.arg(&mut scratch.trap);
        lb.arg(&mut scratch.dt_mean);
        lb.arg(&mut scratch.phase);
        lb.arg(&mut scratch.bp);
        lb.arg(&mut scratch.cp);
        lb.arg(&mut scratch.y_inner);
        lb.arg(&mut scratch.y_out);
        lb.arg(&mut scratch.logits);
        lb.arg(&l_i);
        lb.arg(&n_layers_i);
        lb.arg(&d_i);
        lb.arg(&di_i);
        lb.arg(&ds_i);
        lb.arg(&h_i);
        lb.arg(&hd_i);
        lb.arg(&na_i);
        lb.arg(&v_i);
        lb.arg(&dip_i);

        // 16 blocks × 256 threads. For our tiny model (L=7), 16 blocks is
        // plenty: in_proj matmul has 2240 outputs (140 per block), out_proj
        // has 448 (28 per block), SSM scan needs 8 blocks (one per head),
        // norms need 7. More blocks mean more SMs to wait for at each
        // grid.sync(), so on a contested GPU fewer blocks sync faster.
        let cfg = LaunchConfig {
            grid_dim: (16, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch_cooperative(cfg)? };
        Ok(())
    }

    /// Capture cooperative kernel launch as CUDA Graph.
    pub fn capture_graph_coop(&self, l: usize) -> Result<CudaGraph, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        let dummy: Vec<u32> = vec![0; l];
        self.upload_tokens(&stream, &dummy)?;
        self.launch_coop(&stream, l)?;
        stream.synchronize()?;

        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let r = self.launch_coop(&stream, l);
        let g = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        r?;
        let graph = g?.ok_or("Graph capture returned None")?;
        Ok(graph)
    }

    pub fn forward_graph_coop(
        &self,
        tokens: &[u32],
        graph: &CudaGraph,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        self.upload_tokens(&stream, tokens)?;
        graph.launch()?;
        stream.synchronize()?;

        let l = tokens.len();
        let scratch = self.scratch.borrow();
        let logits_slice = scratch.logits.slice(0..(l * self.vocab_size));
        let host = stream.memcpy_dtov(&logits_slice)?;
        Ok(host)
    }

    /// Forward pass via the persistent single-kernel launch. Entire forward in
    /// ONE CUDA kernel, one dispatch. Logits land in scratch.logits, copied
    /// back to host.
    pub fn forward_persistent(&self, tokens: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        self.upload_tokens(&stream, tokens)?;
        self.launch_persistent(&stream, tokens.len())?;
        stream.synchronize()?;

        let l = tokens.len();
        let scratch = self.scratch.borrow();
        let logits_slice = scratch.logits.slice(0..(l * self.vocab_size));
        let host = stream.memcpy_dtov(&logits_slice)?;
        Ok(host)
    }

    /// Launch the persistent kernel (no upload, no sync, no copy). Used both
    /// by forward_persistent and graph capture.
    pub fn launch_persistent(
        &self,
        stream: &Arc<CudaStream>,
        l: usize,
    ) -> Result<(), Box<dyn Error>> {
        let mut scratch_ref = self.scratch.borrow_mut();
        let scratch: &mut PtxScratch = &mut *scratch_ref;
        let smem = self.persistent_smem_bytes(l);

        let l_i = l as i32;
        let n_layers_i = self.n_layers as i32;
        let d_i = self.d_model as i32;
        let di_i = self.d_inner as i32;
        let ds_i = self.d_state as i32;
        let h_i = self.n_heads as i32;
        let hd_i = self.headdim as i32;
        let na_i = self.num_rope_angles as i32;
        let v_i = self.vocab_size as i32;
        let dip_i = self.d_in_proj as i32;

        let mut lb = stream.launch_builder(&self.ptx.k.mamba3_forward_persistent);
        lb.arg(&scratch.tokens);
        lb.arg(&self.embed_w);
        lb.arg(&self.embed_norm_w);
        lb.arg(&self.embed_norm_b);
        lb.arg(&self.l_in_proj_w);
        lb.arg(&self.l_out_proj_w);
        lb.arg(&self.l_dt_bias);
        lb.arg(&self.l_d_param);
        lb.arg(&self.l_b_norm_w);
        lb.arg(&self.l_b_norm_b);
        lb.arg(&self.l_c_norm_w);
        lb.arg(&self.l_c_norm_b);
        lb.arg(&self.l_ln_w);
        lb.arg(&self.l_ln_b);
        lb.arg(&self.l_scale);
        lb.arg(&self.final_norm_w);
        lb.arg(&self.final_norm_b);
        lb.arg(&mut scratch.logits);
        lb.arg(&l_i);
        lb.arg(&n_layers_i);
        lb.arg(&d_i);
        lb.arg(&di_i);
        lb.arg(&ds_i);
        lb.arg(&h_i);
        lb.arg(&hd_i);
        lb.arg(&na_i);
        lb.arg(&v_i);
        lb.arg(&dip_i);

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (self.persistent_block_size(), 1, 1),
            shared_mem_bytes: smem,
        };
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// Capture the persistent kernel launch as a CUDA graph for replay.
    pub fn capture_graph_persistent(&self, l: usize) -> Result<CudaGraph, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        // Warm-up: resolves JIT, SMEM carveout.
        let dummy: Vec<u32> = vec![0; l];
        self.upload_tokens(&stream, &dummy)?;
        self.launch_persistent(&stream, l)?;
        stream.synchronize()?;

        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let r = self.launch_persistent(&stream, l);
        let g = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        r?;
        let graph = g?.ok_or("Graph capture returned None")?;
        Ok(graph)
    }

    pub fn forward_graph_persistent(
        &self,
        tokens: &[u32],
        graph: &CudaGraph,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        self.upload_tokens(&stream, tokens)?;
        graph.launch()?;
        stream.synchronize()?;

        let l = tokens.len();
        let scratch = self.scratch.borrow();
        let logits_slice = scratch.logits.slice(0..(l * self.vocab_size));
        let host = stream.memcpy_dtov(&logits_slice)?;
        Ok(host)
    }

    /// Forward pass using persistent scratch buffers (no per-call allocation).
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        self.upload_tokens(&stream, tokens)?;
        self.record_compute(&stream, tokens.len())?;
        stream.synchronize()?;

        let l = tokens.len();
        let scratch = self.scratch.borrow();
        let logits_slice = scratch.logits.slice(0..(l * self.vocab_size));
        let host = stream.memcpy_dtov(&logits_slice)?;
        Ok(host)
    }

    /// Forward pass via CUDA Graph, returning only argmax predictions
    /// (L × u32) instead of full logits (L × V × f32). Reduces host readback
    /// from 7.3 KB to 28 bytes for this model. Graph includes the argmax
    /// kernel at the tail.
    pub fn forward_graph_argmax(
        &self,
        tokens: &[u32],
        graph: &CudaGraph,
    ) -> Result<Vec<u32>, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        self.upload_tokens(&stream, tokens)?;
        graph.launch()?;
        // No explicit stream.synchronize() — memcpy_dtov is synchronous and
        // waits for the stream to drain. One sync boundary instead of two.
        let l = tokens.len();
        let scratch = self.scratch.borrow();
        let slice = scratch.preds.slice(0..l);
        let host = stream.memcpy_dtov(&slice)?;
        Ok(host)
    }

    /// Capture the per-op forward + argmax as a CUDA graph. Argmax is the
    /// last node; host reads only L × u32 from scratch.preds after replay.
    pub fn capture_graph_argmax(&self, l: usize) -> Result<CudaGraph, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        // Warmup
        let dummy: Vec<u32> = vec![0; l];
        self.upload_tokens(&stream, &dummy)?;
        self.record_compute(&stream, l)?;
        self.record_argmax(&stream, l)?;
        stream.synchronize()?;

        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let r1 = self.record_compute(&stream, l);
        let r2 = self.record_argmax(&stream, l);
        let g = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        r1?;
        r2?;
        let graph = g?.ok_or("Graph capture returned None")?;
        Ok(graph)
    }

    fn record_argmax(
        &self,
        stream: &Arc<CudaStream>,
        l: usize,
    ) -> Result<(), Box<dyn Error>> {
        let mut scratch_ref = self.scratch.borrow_mut();
        let scratch: &mut PtxScratch = &mut *scratch_ref;
        let l_i = l as i32;
        let v_i = self.vocab_size as i32;
        let mut lb = stream.launch_builder(&self.ptx.k.argmax_f32);
        lb.arg(&scratch.logits);
        lb.arg(&mut scratch.preds);
        lb.arg(&l_i);
        lb.arg(&v_i);
        let cfg = LaunchConfig {
            grid_dim: (l as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    /// Forward pass via a pre-captured CUDA Graph. Eliminates per-kernel
    /// launch overhead — all kernel launches are fused into one graph replay.
    pub fn forward_graph(
        &self,
        tokens: &[u32],
        graph: &CudaGraph,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        self.upload_tokens(&stream, tokens)?;
        graph.launch()?;
        stream.synchronize()?;

        let l = tokens.len();
        let scratch = self.scratch.borrow();
        let logits_slice = scratch.logits.slice(0..(l * self.vocab_size));
        let host = stream.memcpy_dtov(&logits_slice)?;
        Ok(host)
    }

    /// Same as `forward_graph` but returns per-phase timings for diagnostics.
    pub fn forward_graph_diag(
        &self,
        tokens: &[u32],
        graph: &CudaGraph,
    ) -> Result<
        (
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
        ),
        Box<dyn Error>,
    > {
        let stream = self.ptx.stream.clone();
        let t0 = std::time::Instant::now();
        self.upload_tokens(&stream, tokens)?;
        let t1 = std::time::Instant::now();
        graph.launch()?;
        let t2 = std::time::Instant::now();
        stream.synchronize()?;
        let t3 = std::time::Instant::now();

        let l = tokens.len();
        let scratch = self.scratch.borrow();
        let logits_slice = scratch.logits.slice(0..(l * self.vocab_size));
        let _host = stream.memcpy_dtov(&logits_slice)?;
        let t4 = std::time::Instant::now();

        Ok((t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    }

    /// Capture the compute pipeline (all kernel launches for a forward of
    /// length `l`) as a replayable CUDA graph. Tokens must be uploaded to
    /// scratch.tokens via `forward_graph` on each replay — the graph only
    /// contains kernel launches, not the memcpy.
    pub fn capture_graph(&self, l: usize) -> Result<CudaGraph, Box<dyn Error>> {
        let stream = self.ptx.stream.clone();
        // Warm JIT once (kernels compiled / resolved on first launch).
        let dummy: Vec<u32> = vec![0; l];
        self.upload_tokens(&stream, &dummy)?;
        self.record_compute(&stream, l)?;
        stream.synchronize()?;

        // Capture. THREAD_LOCAL mode ensures our stream is isolated; RELAXED
        // permits the widest set of kernel launches without sync ops.
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let capture_result = self.record_compute(&stream, l);
        let graph_result = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        capture_result?;
        let graph = graph_result?.ok_or("Graph capture returned None")?;
        Ok(graph)
    }

    fn upload_tokens(
        &self,
        stream: &Arc<CudaStream>,
        tokens: &[u32],
    ) -> Result<(), Box<dyn Error>> {
        let l = tokens.len();
        assert!(l <= self.scratch.borrow().max_seq, "sequence exceeds max_seq");
        let mut scratch_ref = self.scratch.borrow_mut();
        let scratch: &mut PtxScratch = &mut *scratch_ref;
        let mut dst = scratch.tokens.slice_mut(0..l);
        stream.memcpy_htod(tokens, &mut dst)?;
        Ok(())
    }

    /// Issue all kernel launches for one forward pass of length `l`. Tokens
    /// must already be in scratch.tokens. Results land in scratch.logits.
    /// Purely asynchronous — no synchronization, no memcpy.
    pub fn record_compute(
        &self,
        stream: &Arc<CudaStream>,
        l: usize,
    ) -> Result<(), Box<dyn Error>> {
        let d = self.d_model;
        let di = self.d_inner;
        let ds = self.d_state;
        let hd = self.headdim;
        let nh = self.n_heads;
        let v = self.vocab_size;

        let mut scratch_ref = self.scratch.borrow_mut();
        let scratch: &mut PtxScratch = &mut *scratch_ref;

        // 1. Embedding gather
        {
            let l_i = l as i32;
            let d_i = d as i32;
            let v_i = v as i32;
            let mut lb = stream.launch_builder(&self.ptx.k.embed_gather);
            lb.arg(&scratch.tokens);
            lb.arg(&self.embed_w);
            lb.arg(&mut scratch.x);
            lb.arg(&l_i);
            lb.arg(&d_i);
            lb.arg(&v_i);
            let cfg = LaunchConfig {
                grid_dim: (l as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { lb.launch(cfg)? };
        }

        // 2. Embed norm
        launch_layer_norm(
            stream,
            &self.ptx,
            &mut scratch.x,
            &self.embed_norm_w,
            &self.embed_norm_b,
            l,
            d,
        )?;

        // 3. Layers
        for layer in &self.layers {
            // Copy x → x_normed, then pre-norm
            {
                let n_i = (l * d) as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
                lb.arg(&scratch.x);
                lb.arg(&mut scratch.x_normed);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: (((l * d) as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
            launch_layer_norm(
                stream,
                &self.ptx,
                &mut scratch.x_normed,
                &layer.layer_norm_w,
                &layer.layer_norm_b,
                l,
                d,
            )?;

            // in_proj: proj = x_normed @ in_proj_w.T
            let dip = layer.d_in_proj;
            launch_matmul_t(
                stream,
                &self.ptx,
                &scratch.x_normed,
                &layer.in_proj_w,
                &mut scratch.proj,
                l,
                dip,
                d,
            )?;

            // FUSED: SSM params + dt_mean in one kernel (2 → 1 dispatches)
            {
                let l_i = l as i32;
                let h_i = nh as i32;
                let dip_i = dip as i32;
                let di_i = di as i32;
                let ds_i = ds as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.compute_ssm_params_and_dt_mean);
                lb.arg(&scratch.proj);
                lb.arg(&layer.dt_bias);
                lb.arg(&mut scratch.dt);
                lb.arg(&mut scratch.decay);
                lb.arg(&mut scratch.trap);
                lb.arg(&mut scratch.dt_mean);
                lb.arg(&l_i);
                lb.arg(&h_i);
                lb.arg(&dip_i);
                lb.arg(&di_i);
                lb.arg(&ds_i);
                let cfg = LaunchConfig {
                    grid_dim: (l as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // phase
            let n_angles = layer.num_rope_angles;
            {
                let l_i = l as i32;
                let na_i = n_angles as i32;
                let dip_i = dip as i32;
                let di_i = di as i32;
                let ds_i = ds as i32;
                let h_i = nh as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.compute_phase);
                lb.arg(&scratch.proj);
                lb.arg(&scratch.dt_mean);
                lb.arg(&mut scratch.phase);
                lb.arg(&l_i);
                lb.arg(&na_i);
                lb.arg(&dip_i);
                lb.arg(&di_i);
                lb.arg(&ds_i);
                lb.arg(&h_i);
                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (n_angles.max(1) as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // extract bp, cp
            {
                let l_i = l as i32;
                let ds_i = ds as i32;
                let di_i = di as i32;
                let dip_i = dip as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.extract_bp_cp);
                lb.arg(&scratch.proj);
                lb.arg(&mut scratch.bp);
                lb.arg(&mut scratch.cp);
                lb.arg(&l_i);
                lb.arg(&ds_i);
                lb.arg(&di_i);
                lb.arg(&dip_i);
                let cfg = LaunchConfig {
                    grid_dim: (l as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            launch_layer_norm(
                stream,
                &self.ptx,
                &mut scratch.bp,
                &layer.b_norm_w,
                &layer.b_norm_b,
                l,
                ds,
            )?;
            launch_layer_norm(
                stream,
                &self.ptx,
                &mut scratch.cp,
                &layer.c_norm_w,
                &layer.c_norm_b,
                l,
                ds,
            )?;

            launch_apply_rope(
                stream,
                &self.ptx,
                &mut scratch.bp,
                &scratch.phase,
                l,
                ds,
                n_angles,
            )?;
            launch_apply_rope(
                stream,
                &self.ptx,
                &mut scratch.cp,
                &scratch.phase,
                l,
                ds,
                n_angles,
            )?;

            // SSM scan (z_silu now inlined, no separate compute_z_silu kernel)
            {
                let l_i = l as i32;
                let h_i = nh as i32;
                let hd_i = hd as i32;
                let ds_i = ds as i32;
                let di_i = di as i32;
                let dip_i = dip as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.ssm_scan_sequential);
                lb.arg(&scratch.proj);
                lb.arg(&scratch.bp);
                lb.arg(&scratch.cp);
                lb.arg(&scratch.dt);
                lb.arg(&scratch.decay);
                lb.arg(&scratch.trap);
                lb.arg(&layer.d_param);
                lb.arg(&mut scratch.y_inner);
                lb.arg(&l_i);
                lb.arg(&h_i);
                lb.arg(&hd_i);
                lb.arg(&ds_i);
                lb.arg(&di_i);
                lb.arg(&dip_i);
                let smem_bytes = ((hd * ds) + hd) as u32 * 4;
                let cfg = LaunchConfig {
                    grid_dim: (nh as u32, 1, 1),
                    block_dim: ((hd * ds) as u32, 1, 1),
                    shared_mem_bytes: smem_bytes,
                };
                unsafe { lb.launch(cfg)? };
            }

            // out_proj
            launch_matmul_t(
                stream,
                &self.ptx,
                &scratch.y_inner,
                &layer.out_proj_w,
                &mut scratch.y_out,
                l,
                d,
                di,
            )?;

            // residual: x += scale * y_out
            {
                let n_i = (l * d) as i32;
                let scale_f = layer.scale;
                let mut lb = stream.launch_builder(&self.ptx.k.residual_add);
                lb.arg(&mut scratch.x);
                lb.arg(&scratch.y_out);
                lb.arg(&scale_f);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: (((l * d) as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
        }

        // Final norm
        launch_layer_norm(
            stream,
            &self.ptx,
            &mut scratch.x,
            &self.final_norm_w,
            &self.final_norm_b,
            l,
            d,
        )?;

        // LM head
        launch_matmul_t(
            stream,
            &self.ptx,
            &scratch.x,
            &self.embed_w,
            &mut scratch.logits,
            l,
            v,
            d,
        )?;

        Ok(())
    }
}

// ------ free-function launch helpers (don't borrow PtxModel, only PtxContext) ------

fn launch_layer_norm(
    stream: &Arc<CudaStream>,
    ptx: &PtxContext,
    x: &mut CudaSlice<f32>,
    w: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    l: usize,
    d: usize,
) -> Result<(), Box<dyn Error>> {
    let l_i = l as i32;
    let d_i = d as i32;
    let mut lb = stream.launch_builder(&ptx.k.layer_norm);
    lb.arg(x);
    lb.arg(w);
    lb.arg(b);
    lb.arg(&l_i);
    lb.arg(&d_i);
    let cfg = LaunchConfig {
        grid_dim: (l as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg)? };
    Ok(())
}

fn launch_matmul_t(
    stream: &Arc<CudaStream>,
    ptx: &PtxContext,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn Error>> {
    let m_i = m as i32;
    let n_i = n as i32;
    let k_i = k as i32;
    // Tiled variant is preferred when K is a multiple of 16 (true for our
    // d_model=64, d_inner=128 shapes). Falls back to naive for pathological K.
    let func = if k % 16 == 0 {
        &ptx.k.matmul_t_tiled
    } else {
        &ptx.k.matmul_t
    };
    let mut lb = stream.launch_builder(func);
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

fn launch_apply_rope(
    stream: &Arc<CudaStream>,
    ptx: &PtxContext,
    v: &mut CudaSlice<f32>,
    phase: &CudaSlice<f32>,
    l: usize,
    ds: usize,
    n_angles: usize,
) -> Result<(), Box<dyn Error>> {
    let l_i = l as i32;
    let ds_i = ds as i32;
    let na_i = n_angles as i32;
    let mut lb = stream.launch_builder(&ptx.k.apply_rope);
    lb.arg(v);
    lb.arg(phase);
    lb.arg(&l_i);
    lb.arg(&ds_i);
    lb.arg(&na_i);
    let cfg = LaunchConfig {
        grid_dim: (l as u32, 1, 1),
        block_dim: (n_angles.max(1) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg)? };
    Ok(())
}
