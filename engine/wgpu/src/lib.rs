//! Mamba-3 Engine — wgpu-based SSM scan.
//!
//! Runs the SSM scan on any GPU via wgpu (Vulkan, Metal, DX12).
//! Explicit fp32 arithmetic — no FMA, no TF32.

pub mod scan;
