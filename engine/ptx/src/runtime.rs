//! CUDA runtime + kernel loading. Compiles kernels.cu → PTX at startup via
//! NVRTC with strict FP32 flags.

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::error::Error;
use std::sync::Arc;

pub struct Kernels {
    pub copy_f32: CudaFunction,
    pub embed_gather: CudaFunction,
    pub layer_norm: CudaFunction,
    pub matmul_t: CudaFunction,
    pub matmul_t_tiled: CudaFunction,
    pub compute_ssm_params: CudaFunction,
    pub compute_ssm_params_and_dt_mean: CudaFunction,
    pub compute_dt_mean: CudaFunction,
    pub compute_phase: CudaFunction,
    pub extract_bp_cp: CudaFunction,
    pub apply_rope: CudaFunction,
    pub prepare_bc: CudaFunction,
    pub compute_z_silu: CudaFunction,
    pub ssm_scan_sequential: CudaFunction,
    pub ssm_scan_cached: CudaFunction,
    pub residual_add: CudaFunction,
    pub argmax_f32: CudaFunction,
    pub mamba3_forward_persistent: CudaFunction,
    pub mamba3_forward_coop: CudaFunction,
    pub adamw_step: CudaFunction,
    pub cross_entropy_fwd_bwd: CudaFunction,
    pub fill_zero: CudaFunction,
    pub matmul_ab_tiled: CudaFunction,
    pub matmul_atb_tiled: CudaFunction,
    pub layer_norm_bwd: CudaFunction,
    pub gate_bwd: CudaFunction,
    pub ssm_scan_bwd: CudaFunction,
    pub ssm_scan_bwd_full: CudaFunction,
    pub ssm_param_grads: CudaFunction,
    pub bx_bwd: CudaFunction,
    pub embed_scatter_bwd: CudaFunction,
    pub rope_bwd: CudaFunction,
    pub scatter_add_to_proj: CudaFunction,
    pub gather_slice_from_proj: CudaFunction,
    pub reduce_dot_f32: CudaFunction,
}

pub struct PtxContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub _module: Arc<CudaModule>,
    pub k: Kernels,
}

const KERNELS_CU: &str = include_str!("ptx/kernels.cu");

impl PtxContext {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        // Disable cudarc's event tracking BEFORE allocating any slices.
        // Rationale: event tracking records wait/record CUDA events on each
        // slice across launches for cross-stream synchronization. Those events
        // then leak into captured graphs as cross-stream deps, triggering
        // CUDA_ERROR_STREAM_CAPTURE_ISOLATION. We use a single stream so no
        // cross-stream sync is needed; ordering is preserved by stream FIFO.
        unsafe { ctx.disable_event_tracking() };

        // Use a dedicated non-default stream so we can capture CUDA graphs
        // (default/null stream is not capturable).
        let stream = ctx.new_stream()?;

        // Strict FP32 flags. We explicitly enable FMA because our CPU reference
        // uses IEEE-style FMA (fma.rn), we want prec-div and prec-sqrt, we do
        // NOT want fast-math or FTZ.
        let opts = CompileOptions {
            arch: Some("sm_90"),
            options: vec![
                "--fmad=true".into(),
                "--ftz=false".into(),
                "--prec-div=true".into(),
                "--prec-sqrt=true".into(),
                "-lineinfo".into(),
                // Needed for #include <cooperative_groups.h>. NVRTC doesn't
                // search CUDA toolkit headers by default.
                "--include-path=/usr/local/cuda/include".into(),
                "--include-path=/usr/local/cuda-12.8/targets/x86_64-linux/include".into(),
            ],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNELS_CU, opts)
            .map_err(|e| format!("NVRTC compile failed: {:?}", e))?;

        let module = ctx.load_module(ptx)?;

        let k = Kernels {
            copy_f32: module.load_function("copy_f32")?,
            embed_gather: module.load_function("embed_gather")?,
            layer_norm: module.load_function("layer_norm")?,
            matmul_t: module.load_function("matmul_t")?,
            matmul_t_tiled: module.load_function("matmul_t_tiled")?,
            compute_ssm_params: module.load_function("compute_ssm_params")?,
            compute_ssm_params_and_dt_mean: module.load_function("compute_ssm_params_and_dt_mean")?,
            compute_dt_mean: module.load_function("compute_dt_mean")?,
            compute_phase: module.load_function("compute_phase")?,
            extract_bp_cp: module.load_function("extract_bp_cp")?,
            apply_rope: module.load_function("apply_rope")?,
            prepare_bc: module.load_function("prepare_bc")?,
            compute_z_silu: module.load_function("compute_z_silu")?,
            ssm_scan_sequential: module.load_function("ssm_scan_sequential")?,
            ssm_scan_cached: module.load_function("ssm_scan_cached")?,
            residual_add: module.load_function("residual_add")?,
            argmax_f32: module.load_function("argmax_f32")?,
            mamba3_forward_persistent: module.load_function("mamba3_forward_persistent")?,
            mamba3_forward_coop: module.load_function("mamba3_forward_coop")?,
            adamw_step: module.load_function("adamw_step")?,
            cross_entropy_fwd_bwd: module.load_function("cross_entropy_fwd_bwd")?,
            fill_zero: module.load_function("fill_zero")?,
            matmul_ab_tiled: module.load_function("matmul_ab_tiled")?,
            matmul_atb_tiled: module.load_function("matmul_atb_tiled")?,
            layer_norm_bwd: module.load_function("layer_norm_bwd")?,
            gate_bwd: module.load_function("gate_bwd")?,
            ssm_scan_bwd: module.load_function("ssm_scan_bwd")?,
            ssm_scan_bwd_full: module.load_function("ssm_scan_bwd_full")?,
            ssm_param_grads: module.load_function("ssm_param_grads")?,
            bx_bwd: module.load_function("bx_bwd")?,
            embed_scatter_bwd: module.load_function("embed_scatter_bwd")?,
            rope_bwd: module.load_function("rope_bwd")?,
            scatter_add_to_proj: module.load_function("scatter_add_to_proj")?,
            gather_slice_from_proj: module.load_function("gather_slice_from_proj")?,
            reduce_dot_f32: module.load_function("reduce_dot_f32")?,
        };

        Ok(Self {
            ctx,
            stream,
            _module: module,
            k,
        })
    }
}
