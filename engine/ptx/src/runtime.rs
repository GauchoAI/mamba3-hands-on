//! CUDA runtime + kernel loading. Compiles kernels.cu → PTX at startup via
//! NVRTC with strict FP32 flags.

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::error::Error;
use std::sync::Arc;

pub struct Kernels {
    pub copy_f32: CudaFunction,
    pub embed_gather: CudaFunction,
    pub layer_norm: CudaFunction,
    pub matmul_t: CudaFunction,
    pub compute_ssm_params: CudaFunction,
    pub compute_dt_mean: CudaFunction,
    pub compute_phase: CudaFunction,
    pub extract_bp_cp: CudaFunction,
    pub apply_rope: CudaFunction,
    pub compute_z_silu: CudaFunction,
    pub ssm_scan_sequential: CudaFunction,
    pub residual_add: CudaFunction,
}

pub struct PtxContext {
    pub ctx: Arc<CudaContext>,
    pub _module: Arc<CudaModule>,
    pub k: Kernels,
}

const KERNELS_CU: &str = include_str!("ptx/kernels.cu");

impl PtxContext {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;

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
            compute_ssm_params: module.load_function("compute_ssm_params")?,
            compute_dt_mean: module.load_function("compute_dt_mean")?,
            compute_phase: module.load_function("compute_phase")?,
            extract_bp_cp: module.load_function("extract_bp_cp")?,
            apply_rope: module.load_function("apply_rope")?,
            compute_z_silu: module.load_function("compute_z_silu")?,
            ssm_scan_sequential: module.load_function("ssm_scan_sequential")?,
            residual_add: module.load_function("residual_add")?,
        };

        Ok(Self {
            ctx,
            _module: module,
            k,
        })
    }
}
