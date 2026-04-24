//! Unit tests for training kernels: adamw_step, cross_entropy_fwd_bwd.
//! Compares against CPU reference in mamba3_engine::backward.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use mamba3_engine::backward;
use ptx_engine::PtxContext;
use std::error::Error;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== PTX training kernel tests ===");
    println!("Compiling kernels...");
    let t0 = std::time::Instant::now();
    let ptx = Arc::new(PtxContext::new()?);
    println!("  compiled in {:.2}s", t0.elapsed().as_secs_f64());
    println!();

    test_adamw(&ptx)?;
    test_cross_entropy(&ptx)?;

    println!("\nAll training kernel tests PASSED.");
    Ok(())
}

fn test_adamw(ptx: &Arc<PtxContext>) -> Result<(), Box<dyn Error>> {
    println!("--- adamw_step ---");
    let stream = ptx.stream.clone();

    let n = 1024usize;
    let mut rng = 0x12345678u64;
    let lcg = |s: &mut u64| -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*s >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let params0: Vec<f32> = (0..n).map(|_| lcg(&mut rng)).collect();
    let grads: Vec<f32> = (0..n).map(|_| lcg(&mut rng) * 0.01).collect();
    let m0: Vec<f32> = vec![0.0; n];
    let v0: Vec<f32> = vec![0.0; n];

    let lr = 1e-3f32;
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps = 1e-8f32;
    let wd = 0.1f32;

    // Run 3 optimizer steps, verify after each one
    let mut cpu_params = params0.clone();
    let mut cpu_m = m0.clone();
    let mut cpu_v = v0.clone();

    let mut gpu_params = stream.memcpy_stod(&params0)?;
    let gpu_grads = stream.memcpy_stod(&grads)?;
    let mut gpu_m = stream.memcpy_stod(&m0)?;
    let mut gpu_v = stream.memcpy_stod(&v0)?;

    for step in 1..=3u32 {
        // CPU
        backward::adamw_step(
            &mut cpu_params, &grads, &mut cpu_m, &mut cpu_v,
            lr, beta1, beta2, eps, wd, step,
        );

        // GPU
        let bc1_inv = 1.0f32 / (1.0 - beta1.powi(step as i32));
        let bc2_inv = 1.0f32 / (1.0 - beta2.powi(step as i32));
        let n_i = n as i32;
        let mut lb = stream.launch_builder(&ptx.k.adamw_step);
        lb.arg(&mut gpu_params);
        lb.arg(&gpu_grads);
        lb.arg(&mut gpu_m);
        lb.arg(&mut gpu_v);
        lb.arg(&lr);
        lb.arg(&beta1);
        lb.arg(&beta2);
        lb.arg(&eps);
        lb.arg(&wd);
        lb.arg(&bc1_inv);
        lb.arg(&bc2_inv);
        lb.arg(&n_i);
        let cfg = LaunchConfig {
            grid_dim: (((n as u32) + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg)? };
        stream.synchronize()?;

        let gpu_p_host = stream.memcpy_dtov(&gpu_params)?;
        let max_diff: f32 = cpu_params.iter().zip(gpu_p_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let status = if max_diff < 1e-6 { "PASS" } else { "FAIL" };
        println!("  step {}: max |CPU-GPU| params diff = {:.3e}   {}", step, max_diff, status);
        if max_diff >= 1e-6 {
            // Show first few mismatches
            for (i, (c, g)) in cpu_params.iter().zip(gpu_p_host.iter()).enumerate().take(5) {
                println!("    [{}] CPU={} GPU={} diff={}", i, c, g, (c - g).abs());
            }
            return Err("adamw mismatch".into());
        }
    }
    Ok(())
}

fn test_cross_entropy(ptx: &Arc<PtxContext>) -> Result<(), Box<dyn Error>> {
    println!("--- cross_entropy_fwd_bwd ---");
    let stream = ptx.stream.clone();

    let l = 7usize;
    let vocab = 260usize;
    let mut rng = 0xabcdef00u64;
    let lcg = |s: &mut u64| -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*s >> 33) as u32 as f32 / u32::MAX as f32) * 4.0 - 2.0
    };

    let logits: Vec<f32> = (0..l * vocab).map(|_| lcg(&mut rng)).collect();
    let targets: Vec<u32> = vec![0, 48, 32, 49, 32, 48, 50];

    // CPU reference
    let (cpu_loss, cpu_d_logits) = backward::cross_entropy_loss(&logits, &targets, vocab, l);

    // GPU
    let gpu_logits = stream.memcpy_stod(&logits)?;
    let gpu_targets = stream.memcpy_stod(&targets)?;
    let mut gpu_d_logits = stream.alloc_zeros::<f32>(l * vocab)?;
    let mut gpu_loss = stream.memcpy_stod(&vec![0.0f32])?;

    let l_i = l as i32;
    let v_i = vocab as i32;
    let mut lb = stream.launch_builder(&ptx.k.cross_entropy_fwd_bwd);
    lb.arg(&gpu_logits);
    lb.arg(&gpu_targets);
    lb.arg(&mut gpu_d_logits);
    lb.arg(&mut gpu_loss);
    lb.arg(&l_i);
    lb.arg(&v_i);
    let cfg = LaunchConfig {
        grid_dim: (l as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { lb.launch(cfg)? };
    stream.synchronize()?;

    let gpu_d_host = stream.memcpy_dtov(&gpu_d_logits)?;
    let gpu_loss_host = stream.memcpy_dtov(&gpu_loss)?;

    let loss_diff = (cpu_loss - gpu_loss_host[0]).abs();
    let d_diff: f32 = cpu_d_logits.iter().zip(gpu_d_host.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let status = if loss_diff < 1e-5 && d_diff < 1e-5 { "PASS" } else { "FAIL" };
    println!(
        "  loss: CPU={:.6}  GPU={:.6}  diff={:.3e}",
        cpu_loss, gpu_loss_host[0], loss_diff
    );
    println!("  d_logits: max diff = {:.3e}   {}", d_diff, status);

    if loss_diff >= 1e-5 || d_diff >= 1e-5 {
        return Err("cross_entropy mismatch".into());
    }
    Ok(())
}
