//! PTX matmul micro-benchmark.
//!
//! Loads a hand-written PTX kernel (fma.rn.f32, no fast-math) on an H100,
//! verifies correctness against a CPU reference, then benchmarks throughput
//! at several matrix sizes representative of Mamba-3 projections.
//!
//!   cargo run --release
//!

use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::time::Instant;

const PTX_SRC: &str = include_str!("matmul.ptx");

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut s = 0.0f32;
            for t in 0..k {
                // mul_add = IEEE fused multiply-add, matches PTX fma.rn.f32
                s = a[r * k + t].mul_add(b[t * n + col], s);
            }
            c[r * n + col] = s;
        }
    }
    c
}

fn main() -> Result<(), DriverError> {
    let dev = CudaDevice::new(0)?;
    println!("Device: {}", dev.name()?);
    println!("Kernel: hand-written PTX, naive matmul, fma.rn.f32 (IEEE round-to-nearest, no fast-math)");
    println!();

    dev.load_ptx(Ptx::from_src(PTX_SRC), "mm", &["matmul_naive_f32"])?;
    let f = dev.get_func("mm", "matmul_naive_f32").unwrap();

    // --- Correctness check at 32x32x32 ---
    {
        let m = 32usize;
        let n = 32usize;
        let k = 32usize;
        let a_host: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let b_host: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.02).cos()).collect();
        let c_ref = cpu_matmul(&a_host, &b_host, m, n, k);

        let a = dev.htod_sync_copy(&a_host)?;
        let b = dev.htod_sync_copy(&b_host)?;
        let mut c = dev.alloc_zeros::<f32>(m * n)?;

        let bx = 16u32;
        let by = 16u32;
        let cfg = LaunchConfig {
            grid_dim: (
                (n as u32 + bx - 1) / bx,
                (m as u32 + by - 1) / by,
                1,
            ),
            block_dim: (bx, by, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            f.clone().launch(
                cfg,
                (&a, &b, &mut c, m as u32, n as u32, k as u32),
            )?
        };
        dev.synchronize()?;
        let c_host = dev.dtoh_sync_copy(&c)?;
        let max_diff = c_ref
            .iter()
            .zip(c_host.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "Correctness (32x32x32):  max |GPU - CPU| = {:.3e}   {}",
            max_diff,
            if max_diff < 1e-5 { "PASS" } else { "FAIL" }
        );
        println!();
    }

    // --- Benchmark sweep ---
    //
    // Mamba-3 deployed configs use d_model in [64..256].  Projections are
    // (L, d_model) x (d_model, 2*d_inner), inner matmuls topping out around
    // 512x512.  We sweep 128 -> 1024 to show how throughput scales with
    // arithmetic intensity on this hardware.
    println!("Per-size benchmark (~1.25 s each, ~5 s total):");
    println!(
        "{:>10}  {:>14}  {:>10}  {:>11}  {:>12}",
        "size", "matmuls/sec", "GFLOPS", "µs / matmul", "total launches"
    );

    let per_size_secs = 1.25f64;
    for &size in &[128u32, 256, 512, 1024] {
        let m = size;
        let n = size;
        let k = size;
        let a_host: Vec<f32> = (0..(m * k) as usize)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect();
        let b_host: Vec<f32> = (0..(k * n) as usize)
            .map(|i| ((i as f32) * 0.002).cos())
            .collect();

        let a = dev.htod_sync_copy(&a_host)?;
        let b = dev.htod_sync_copy(&b_host)?;
        let mut c = dev.alloc_zeros::<f32>((m * n) as usize)?;

        let bx = 16u32;
        let by = 16u32;
        let cfg = LaunchConfig {
            grid_dim: ((n + bx - 1) / bx, (m + by - 1) / by, 1),
            block_dim: (bx, by, 1),
            shared_mem_bytes: 0,
        };

        // Warmup
        for _ in 0..10 {
            unsafe {
                f.clone()
                    .launch(cfg, (&a, &b, &mut c, m, n, k))?
            };
        }
        dev.synchronize()?;

        // Time: submit launches continuously for `per_size_secs`.  Periodic
        // sync keeps the launch queue from backing up arbitrarily.
        let start = Instant::now();
        let mut count = 0u64;
        while start.elapsed().as_secs_f64() < per_size_secs {
            unsafe {
                f.clone()
                    .launch(cfg, (&a, &b, &mut c, m, n, k))?
            };
            count += 1;
            if count % 2048 == 0 {
                dev.synchronize()?;
            }
        }
        dev.synchronize()?;
        let secs = start.elapsed().as_secs_f64();

        let flops_per_mm = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops_per_mm * count as f64 / secs / 1e9;
        let us_per_mm = secs * 1e6 / count as f64;
        let mm_per_sec = count as f64 / secs;

        println!(
            "{:>4}x{:<4}  {:>14.0}  {:>10.1}  {:>11.2}  {:>12}",
            m, n, mm_per_sec, gflops, us_per_mm, count
        );
    }

    println!();
    println!("Note: naive kernel, no shared memory tiling, no tensor cores.");
    println!("      H100 FP32 peak ~67 TFLOPS — this kernel is memory-bandwidth bound.");
    Ok(())
}
