//! Persistent scratch buffers for PtxModel. Allocated once at construction
//! sized for `max_seq`; reused across every forward call. No per-call
//! allocation. Prerequisite for CUDA graph capture.

use cudarc::driver::CudaSlice;
use std::error::Error;
use std::sync::Arc;

use crate::runtime::PtxContext;

pub struct PtxScratch {
    pub max_seq: usize,
    pub tokens: CudaSlice<u32>,
    pub x: CudaSlice<f32>,
    pub x_normed: CudaSlice<f32>,
    pub proj: CudaSlice<f32>,
    pub dt: CudaSlice<f32>,
    pub decay: CudaSlice<f32>,
    pub trap: CudaSlice<f32>,
    pub dt_mean: CudaSlice<f32>,
    pub phase: CudaSlice<f32>,
    pub bp: CudaSlice<f32>,
    pub cp: CudaSlice<f32>,
    pub z_silu: CudaSlice<f32>,
    pub y_inner: CudaSlice<f32>,
    pub y_out: CudaSlice<f32>,
    pub logits: CudaSlice<f32>,
}

impl PtxScratch {
    pub fn new(
        ctx: &Arc<PtxContext>,
        max_seq: usize,
        d_model: usize,
        d_state: usize,
        d_inner: usize,
        n_heads: usize,
        max_dip: usize,
        max_n_angles: usize,
        vocab_size: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let stream = ctx.ctx.default_stream();
        Ok(Self {
            max_seq,
            tokens: stream.alloc_zeros::<u32>(max_seq)?,
            x: stream.alloc_zeros::<f32>(max_seq * d_model)?,
            x_normed: stream.alloc_zeros::<f32>(max_seq * d_model)?,
            proj: stream.alloc_zeros::<f32>(max_seq * max_dip)?,
            dt: stream.alloc_zeros::<f32>(max_seq * n_heads)?,
            decay: stream.alloc_zeros::<f32>(max_seq * n_heads)?,
            trap: stream.alloc_zeros::<f32>(max_seq * n_heads)?,
            dt_mean: stream.alloc_zeros::<f32>(max_seq)?,
            phase: stream.alloc_zeros::<f32>(max_seq * max_n_angles.max(1))?,
            bp: stream.alloc_zeros::<f32>(max_seq * d_state)?,
            cp: stream.alloc_zeros::<f32>(max_seq * d_state)?,
            z_silu: stream.alloc_zeros::<f32>(max_seq * d_inner)?,
            y_inner: stream.alloc_zeros::<f32>(max_seq * d_inner)?,
            y_out: stream.alloc_zeros::<f32>(max_seq * d_model)?,
            logits: stream.alloc_zeros::<f32>(max_seq * vocab_size)?,
        })
    }
}
