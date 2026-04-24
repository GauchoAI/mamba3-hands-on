//! Mamba-3 Engine — wgpu-based SSM scan + full model inference.
//!
//! Runs on any GPU via wgpu (Vulkan, Metal, DX12).
//! Explicit fp32 arithmetic — no FMA, no TF32.
//! No PyTorch dependency.

pub mod scan;
pub mod model;
