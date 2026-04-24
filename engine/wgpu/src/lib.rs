//! Mamba-3 Engine — wgpu-based SSM scan with explicit fp32 control.
//!
//! Exposes a Python module via PyO3 that runs the SSM scan on any GPU
//! backend (Vulkan, Metal, DX12) without PyTorch dependency.
//!
//! Usage from Python:
//! ```python
//! import mamba3_engine
//! y = mamba3_engine.ssm_scan(inp, decay, C, x, z_silu, D, B, L, H, hD, dS)
//! ```

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use std::sync::OnceLock;

mod scan;

static GPU: OnceLock<scan::GpuContext> = OnceLock::new();

fn get_gpu() -> &'static scan::GpuContext {
    GPU.get_or_init(|| {
        pollster::block_on(scan::GpuContext::new())
            .expect("Failed to initialize GPU")
    })
}

/// Run SSM scan on GPU via wgpu compute shader.
/// All inputs are flat f32 arrays. Returns flat f32 array of shape (B*L*H*hD,).
#[pyfunction]
fn ssm_scan(
    inp: PyReadonlyArray1<f32>,
    decay: PyReadonlyArray1<f32>,
    c: PyReadonlyArray1<f32>,
    x: PyReadonlyArray1<f32>,
    z_silu: PyReadonlyArray1<f32>,
    d: PyReadonlyArray1<f32>,
    b: u32, l: u32, h: u32, hd: u32, ds: u32,
) -> PyResult<Py<PyArray1<f32>>> {
    let gpu = get_gpu();
    let result = pollster::block_on(gpu.run_scan(
        inp.as_slice()?,
        decay.as_slice()?,
        c.as_slice()?,
        x.as_slice()?,
        z_silu.as_slice()?,
        d.as_slice()?,
        b, l, h, hd, ds,
    )).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

    Python::with_gil(|py| {
        Ok(PyArray1::from_vec(py, result).into())
    })
}

#[pymodule]
fn mamba3_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ssm_scan, m)?)?;
    Ok(())
}
