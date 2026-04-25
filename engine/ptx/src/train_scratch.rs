//! Per-layer activation cache buffers for training (backward pass).
//!
//! Separate from PtxScratch so the inference path isn't bloated — a pure-
//! inference PtxModel doesn't allocate these.
//!
//! For an (n_layers, L, ...) tensor, layer l occupies the slice
//! `[l * stride .. (l+1) * stride]` of the corresponding CudaSlice.

use cudarc::driver::CudaSlice;
use std::error::Error;
use std::sync::Arc;

use crate::runtime::PtxContext;

pub struct TrainScratch {
    pub max_seq: usize,
    pub n_layers: usize,

    // Per-layer strides (number of f32 elements per layer)
    pub layer_input_stride: usize,  // max_seq * d_model
    pub layer_proj_stride: usize,   // max_seq * max_dip
    pub layer_yinner_stride: usize, // max_seq * d_inner
    pub layer_cp_stride: usize,     // max_seq * d_state
    pub layer_decay_stride: usize,  // max_seq * n_heads
    pub layer_states_stride: usize, // (max_seq + 1) * n_heads * hd * d_state

    // x going INTO each layer (after pre-norm has NOT been applied; this is
    // the residual stream state at layer start). Shape per layer: (L, d_model)
    pub layer_inputs: CudaSlice<f32>,
    // x AFTER the per-layer pre-norm — this is what feeds in_proj, so the
    // correct operand for the d_in_proj_w matmul. Shape per layer: (L, d_model)
    pub layer_x_normed: CudaSlice<f32>,
    // In-proj output per layer. Shape: (L, dip)
    pub layer_projs: CudaSlice<f32>,
    // SSM output before out-proj. Shape: (L, d_inner)
    pub layer_y_inners: CudaSlice<f32>,
    // Per-layer y AFTER out_proj (what's multiplied by `scale` and added to
    // the residual). Needed to compute d_scale = <d_x_in_layer, y_post>.
    pub layer_y_post: CudaSlice<f32>,  // (n_layers, L, d_model)
    // Layer-normed + RoPE'd bp per layer (needed to reconstruct blended in
    // the adjoint scan without dividing by dt). (L, ds)
    pub layer_bps: CudaSlice<f32>,
    // Layer-normed + RoPE'd cp per layer (needed by adjoint scan). (L, ds)
    pub layer_cps: CudaSlice<f32>,
    // decay per layer (needed by adjoint scan). (L, n_heads)
    pub layer_decays: CudaSlice<f32>,
    // dt per layer (needed for parameter-grad kernels). (L, n_heads)
    pub layer_dts: CudaSlice<f32>,
    // trap (= sigmoid(tr_raw)) per layer — needed for trap_to_proj_bwd to
    // apply the sigmoid' factor when converting d_trap → d_proj[tr_off].
    // (n_layers, L, n_heads)  same stride as layer_dts/layer_decays.
    pub layer_traps: CudaSlice<f32>,
    // Full scan state sequence per layer. Shape: (L+1, n_heads, hd, ds)
    // states[0] is zero.
    pub layer_states: CudaSlice<f32>,

    // x after the final norm — for LM-head backward.  (L, d_model)
    pub x_before_head: CudaSlice<f32>,
    // x BEFORE the final norm — for final-norm backward. (L, d_model)
    pub x_before_final_norm: CudaSlice<f32>,
    // x BEFORE the embed norm — for embed-norm backward. (L, d_model)
    pub x_before_embed_norm: CudaSlice<f32>,
    // Single-float accumulator for cross-entropy loss.
    pub loss: CudaSlice<f32>,
    // Gradient of logits.  (L, vocab)
    pub d_logits: CudaSlice<f32>,
    // Gradient of x (residual stream), flowing through layers backward
    pub d_x: CudaSlice<f32>,      // (L, d)
    pub d_y_out: CudaSlice<f32>,  // (L, d), per layer (overwritten)
    pub d_y_inner: CudaSlice<f32>,// (L, d_inner)
    pub d_proj: CudaSlice<f32>,   // (L, max_dip)
    pub d_y_pregate: CudaSlice<f32>,// (L, d_inner)
    pub d_scan_inp: CudaSlice<f32>, // (L, H, hd, ds)
    // Gradient tensors for weights (accumulated by backward)
    pub d_embed: CudaSlice<f32>,    // (vocab, d)
    pub d_fnorm_w: CudaSlice<f32>,  // (d,)
    pub d_fnorm_b: CudaSlice<f32>,  // (d,)
    pub d_embed_norm_w: CudaSlice<f32>,  // (d,)
    pub d_embed_norm_b: CudaSlice<f32>,  // (d,)
    pub d_in_proj_w: Vec<CudaSlice<f32>>,   // per layer, (dip, d)
    pub d_out_proj_w: Vec<CudaSlice<f32>>,  // per layer, (d, di)
    pub d_d_param: Vec<CudaSlice<f32>>,     // per layer, (H,)
    pub d_dt_bias: Vec<CudaSlice<f32>>,     // per layer, (H,)
    pub d_layer_norm_w: Vec<CudaSlice<f32>>, // per layer, (d,)
    pub d_layer_norm_b: Vec<CudaSlice<f32>>, // per layer, (d,)
    // Intermediate buffers (shared across layers, overwritten each layer)
    pub d_decay: CudaSlice<f32>,            // (L, H)
    pub d_dt_from_inp: CudaSlice<f32>,      // (L, H)
    // d_trap_pre = d w.r.t. trap (post-sigmoid value). bx_bwd accumulates here.
    // trap_to_proj_bwd then multiplies by trap*(1-trap) to get d_tr_raw and
    // writes the result into d_proj[tr_off + h]. (L, H)
    pub d_trap_pre: CudaSlice<f32>,
    // Post-LN+RoPE gradient slots for the bp/cp chain. Zero-init at step
    // start; ssm_scan_bwd_full and bx_bwd atomicAdd into these; then rope_bwd
    // + layer_norm_bwd produce the pre-LN raw gradient, which gets
    // scatter-added into d_proj[bp_slice]/d_proj[cp_slice].
    pub d_bp_post: CudaSlice<f32>,          // (L, ds)
    pub d_cp_post: CudaSlice<f32>,          // (L, ds)
    // Scratch for the bp/cp norm-bwd chain (each pass reuses both).
    // bc_raw_tmp holds a contiguous (L, ds) gather of proj[bp_slice] or
    // proj[cp_slice] as the `x` input to layer_norm_bwd.
    pub bc_raw_tmp: CudaSlice<f32>,         // (L, ds)
    // d_ln_tmp is the `d_x` output of layer_norm_bwd (the raw-side grad).
    pub d_ln_tmp: CudaSlice<f32>,           // (L, ds)
    // Per-layer phases cached in forward (needed for rope_bwd).
    pub layer_phase_stride: usize,          // max_seq * max_n_angles
    pub layer_phases: CudaSlice<f32>,       // (n_layers, L, n_angles)
    // LN-param grads on bp/cp chain — computed but NOT applied until we're
    // confident the chain is bit-clean.
    pub d_b_norm_w: Vec<CudaSlice<f32>>,
    pub d_b_norm_b: Vec<CudaSlice<f32>>,
    pub d_c_norm_w: Vec<CudaSlice<f32>>,
    pub d_c_norm_b: Vec<CudaSlice<f32>>,
    // Per-layer d_scale scalar (computed as <d_x_in_layer, y_post>).
    // Applied via host-side AdamW in apply_optimizer_step.
    pub d_scale: Vec<CudaSlice<f32>>,       // each len 1
}

impl TrainScratch {
    pub fn new(
        ctx: &Arc<PtxContext>,
        max_seq: usize,
        n_layers: usize,
        d_model: usize,
        d_state: usize,
        d_inner: usize,
        n_heads: usize,
        headdim: usize,
        max_dip: usize,
        vocab_size: usize,
        max_n_angles: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let stream = &ctx.stream;
        let li = max_seq * d_model;
        let lp = max_seq * max_dip;
        let ly = max_seq * d_inner;
        let lc = max_seq * d_state;
        let ld = max_seq * n_heads;
        let ls = (max_seq + 1) * n_heads * headdim * d_state;

        let mut d_in_proj_w = Vec::with_capacity(n_layers);
        let mut d_out_proj_w = Vec::with_capacity(n_layers);
        let mut d_d_param = Vec::with_capacity(n_layers);
        let mut d_dt_bias = Vec::with_capacity(n_layers);
        let mut d_layer_norm_w = Vec::with_capacity(n_layers);
        let mut d_layer_norm_b = Vec::with_capacity(n_layers);
        let mut d_b_norm_w = Vec::with_capacity(n_layers);
        let mut d_b_norm_b = Vec::with_capacity(n_layers);
        let mut d_c_norm_w = Vec::with_capacity(n_layers);
        let mut d_c_norm_b = Vec::with_capacity(n_layers);
        let mut d_scale = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            d_in_proj_w.push(stream.alloc_zeros::<f32>(max_dip * d_model)?);
            d_out_proj_w.push(stream.alloc_zeros::<f32>(d_model * d_inner)?);
            d_d_param.push(stream.alloc_zeros::<f32>(n_heads)?);
            d_dt_bias.push(stream.alloc_zeros::<f32>(n_heads)?);
            d_layer_norm_w.push(stream.alloc_zeros::<f32>(d_model)?);
            d_layer_norm_b.push(stream.alloc_zeros::<f32>(d_model)?);
            d_b_norm_w.push(stream.alloc_zeros::<f32>(d_state)?);
            d_b_norm_b.push(stream.alloc_zeros::<f32>(d_state)?);
            d_c_norm_w.push(stream.alloc_zeros::<f32>(d_state)?);
            d_c_norm_b.push(stream.alloc_zeros::<f32>(d_state)?);
            d_scale.push(stream.alloc_zeros::<f32>(1)?);
        }
        let lphase = max_seq * max_n_angles.max(1);

        Ok(Self {
            max_seq,
            n_layers,
            layer_input_stride: li,
            layer_proj_stride: lp,
            layer_yinner_stride: ly,
            layer_cp_stride: lc,
            layer_decay_stride: ld,
            layer_states_stride: ls,
            layer_inputs: stream.alloc_zeros::<f32>(n_layers * li)?,
            layer_x_normed: stream.alloc_zeros::<f32>(n_layers * li)?,
            layer_projs: stream.alloc_zeros::<f32>(n_layers * lp)?,
            layer_y_inners: stream.alloc_zeros::<f32>(n_layers * ly)?,
            layer_y_post: stream.alloc_zeros::<f32>(n_layers * li)?,
            layer_bps: stream.alloc_zeros::<f32>(n_layers * lc)?,
            layer_cps: stream.alloc_zeros::<f32>(n_layers * lc)?,
            layer_decays: stream.alloc_zeros::<f32>(n_layers * ld)?,
            layer_dts: stream.alloc_zeros::<f32>(n_layers * ld)?,
            layer_traps: stream.alloc_zeros::<f32>(n_layers * ld)?,
            layer_states: stream.alloc_zeros::<f32>(n_layers * ls)?,
            x_before_head: stream.alloc_zeros::<f32>(max_seq * d_model)?,
            x_before_final_norm: stream.alloc_zeros::<f32>(max_seq * d_model)?,
            x_before_embed_norm: stream.alloc_zeros::<f32>(max_seq * d_model)?,
            loss: stream.alloc_zeros::<f32>(1)?,
            d_logits: stream.alloc_zeros::<f32>(max_seq * vocab_size)?,
            d_x: stream.alloc_zeros::<f32>(max_seq * d_model)?,
            d_y_out: stream.alloc_zeros::<f32>(max_seq * d_model)?,
            d_y_inner: stream.alloc_zeros::<f32>(max_seq * d_inner)?,
            d_proj: stream.alloc_zeros::<f32>(max_seq * max_dip)?,
            d_y_pregate: stream.alloc_zeros::<f32>(max_seq * d_inner)?,
            d_scan_inp: stream.alloc_zeros::<f32>(max_seq * n_heads * headdim * d_state)?,
            d_embed: stream.alloc_zeros::<f32>(vocab_size * d_model)?,
            d_fnorm_w: stream.alloc_zeros::<f32>(d_model)?,
            d_fnorm_b: stream.alloc_zeros::<f32>(d_model)?,
            d_embed_norm_w: stream.alloc_zeros::<f32>(d_model)?,
            d_embed_norm_b: stream.alloc_zeros::<f32>(d_model)?,
            d_in_proj_w,
            d_out_proj_w,
            d_d_param,
            d_dt_bias,
            d_layer_norm_w,
            d_layer_norm_b,
            d_decay: stream.alloc_zeros::<f32>(max_seq * n_heads)?,
            d_dt_from_inp: stream.alloc_zeros::<f32>(max_seq * n_heads)?,
            d_trap_pre: stream.alloc_zeros::<f32>(max_seq * n_heads)?,
            d_bp_post: stream.alloc_zeros::<f32>(lc)?,
            d_cp_post: stream.alloc_zeros::<f32>(lc)?,
            bc_raw_tmp: stream.alloc_zeros::<f32>(lc)?,
            d_ln_tmp: stream.alloc_zeros::<f32>(lc)?,
            layer_phase_stride: lphase,
            layer_phases: stream.alloc_zeros::<f32>(n_layers * lphase)?,
            d_b_norm_w,
            d_b_norm_b,
            d_c_norm_w,
            d_c_norm_b,
            d_scale,
        })
    }
}
