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
    // In-proj output per layer. Shape: (L, dip)
    pub layer_projs: CudaSlice<f32>,
    // SSM output before out-proj. Shape: (L, d_inner)
    pub layer_y_inners: CudaSlice<f32>,
    // Layer-normed + RoPE'd cp per layer (needed by adjoint scan). (L, ds)
    pub layer_cps: CudaSlice<f32>,
    // decay per layer (needed by adjoint scan). (L, n_heads)
    pub layer_decays: CudaSlice<f32>,
    // Full scan state sequence per layer. Shape: (L+1, n_heads, hd, ds)
    // states[0] is zero.
    pub layer_states: CudaSlice<f32>,

    // x after the final norm — for LM-head backward.  (L, d_model)
    pub x_before_head: CudaSlice<f32>,
    // Single-float accumulator for cross-entropy loss.
    pub loss: CudaSlice<f32>,
    // Gradient of logits.  (L, vocab)
    pub d_logits: CudaSlice<f32>,
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
    ) -> Result<Self, Box<dyn Error>> {
        let stream = &ctx.stream;
        let li = max_seq * d_model;
        let lp = max_seq * max_dip;
        let ly = max_seq * d_inner;
        let lc = max_seq * d_state;
        let ld = max_seq * n_heads;
        let ls = (max_seq + 1) * n_heads * headdim * d_state;

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
            layer_projs: stream.alloc_zeros::<f32>(n_layers * lp)?,
            layer_y_inners: stream.alloc_zeros::<f32>(n_layers * ly)?,
            layer_cps: stream.alloc_zeros::<f32>(n_layers * lc)?,
            layer_decays: stream.alloc_zeros::<f32>(n_layers * ld)?,
            layer_states: stream.alloc_zeros::<f32>(n_layers * ls)?,
            x_before_head: stream.alloc_zeros::<f32>(max_seq * d_model)?,
            loss: stream.alloc_zeros::<f32>(1)?,
            d_logits: stream.alloc_zeros::<f32>(max_seq * vocab_size)?,
        })
    }
}
