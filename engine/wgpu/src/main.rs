//! Benchmark + correctness test: SSM scan via wgpu vs CPU reference.

use mamba3_engine::scan::GpuContext;
use std::time::Instant;

/// CPU reference implementation — matches PyTorch JIT exactly
fn ssm_scan_cpu(
    inp: &[f32], decay: &[f32], c: &[f32],
    x: &[f32], z_silu: &[f32], d: &[f32],
    b: usize, l: usize, h: usize, hd: usize, ds: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; b * l * h * hd];

    for bi in 0..b {
        for hi in 0..h {
            // State: (hd, ds)
            let mut state = vec![0.0f32; hd * ds];

            for t in 0..l {
                let dec = decay[bi * l * h + t * h + hi];

                // State update: state = decay * state + inp
                for p in 0..hd {
                    for n in 0..ds {
                        let si = p * ds + n;
                        let inp_val = inp[((bi * l + t) * h + hi) * hd * ds + p * ds + n];
                        state[si] = dec * state[si] + inp_val;
                    }
                }

                // Output projection + skip + gate
                for p in 0..hd {
                    let mut sum = 0.0f32;
                    for n in 0..ds {
                        let c_val = c[((bi * l + t) * h + hi) * ds + n];
                        sum += state[p * ds + n] * c_val;
                    }
                    let x_val = x[((bi * l + t) * h + hi) * hd + p];
                    sum += d[hi] * x_val;
                    let gate = z_silu[((bi * l + t) * h + hi) * hd + p];
                    y[((bi * l + t) * h + hi) * hd + p] = sum * gate;
                }
            }
        }
    }
    y
}

fn main() {
    let gpu = pollster::block_on(GpuContext::new()).expect("Failed to init GPU");

    let b: u32 = 4;
    let l: u32 = 16;
    let h: u32 = 8;
    let hd: u32 = 16;
    let ds: u32 = 16;

    let inp_size = (b * l * h * hd * ds) as usize;
    let decay_size = (b * l * h) as usize;
    let c_size = (b * l * h * ds) as usize;
    let x_size = (b * l * h * hd) as usize;
    let d_size = h as usize;

    // Deterministic test data
    let inp: Vec<f32> = (0..inp_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let decay: Vec<f32> = (0..decay_size).map(|i| 0.5 + 0.4 * (i as f32 * 0.01).sin()).collect();
    let c: Vec<f32> = (0..c_size).map(|i| (i as f32 * 0.002).cos()).collect();
    let x: Vec<f32> = (0..x_size).map(|i| (i as f32 * 0.003).sin()).collect();
    let z_silu: Vec<f32> = (0..x_size).map(|i| {
        let z = (i as f32 * 0.004).sin();
        z * (1.0 / (1.0 + (-z).exp()))
    }).collect();
    let d: Vec<f32> = (0..d_size).map(|i| 0.1 * i as f32).collect();

    // === Correctness Test ===
    println!("=== Correctness: wgpu vs CPU reference ===");
    let cpu_result = ssm_scan_cpu(&inp, &decay, &c, &x, &z_silu, &d,
        b as usize, l as usize, h as usize, hd as usize, ds as usize);
    let gpu_result = pollster::block_on(gpu.run_scan(&inp, &decay, &c, &x, &z_silu, &d, b, l, h, hd, ds))
        .expect("GPU scan failed");

    let max_diff: f32 = cpu_result.iter().zip(gpu_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let avg_diff: f32 = cpu_result.iter().zip(gpu_result.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / cpu_result.len() as f32;

    println!("  Max diff:  {:.2e}", max_diff);
    println!("  Avg diff:  {:.2e}", avg_diff);
    println!("  Status:    {}", if max_diff < 1e-4 { "PASS" } else { "FAIL" });

    // Print comparison of first few values
    println!("\n  {:>12} {:>12} {:>12}", "CPU", "GPU", "Diff");
    for i in 0..8.min(cpu_result.len()) {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        println!("  {:>12.6} {:>12.6} {:>12.2e}", cpu_result[i], gpu_result[i], diff);
    }

    // === Benchmark ===
    println!("\n=== Benchmark ===");
    // Larger batch for perf test
    let b_bench: u32 = 32;
    let l_bench: u32 = 64;
    let inp_b: Vec<f32> = vec![0.01; (b_bench * l_bench * h * hd * ds) as usize];
    let decay_b: Vec<f32> = vec![0.9; (b_bench * l_bench * h) as usize];
    let c_b: Vec<f32> = vec![0.1; (b_bench * l_bench * h * ds) as usize];
    let x_b: Vec<f32> = vec![0.05; (b_bench * l_bench * h * hd) as usize];
    let z_b: Vec<f32> = vec![0.5; (b_bench * l_bench * h * hd) as usize];

    // Warmup
    for _ in 0..5 {
        let _ = pollster::block_on(gpu.run_scan(&inp_b, &decay_b, &c_b, &x_b, &z_b, &d,
            b_bench, l_bench, h, hd, ds));
    }

    let n_iters = 100;
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = pollster::block_on(gpu.run_scan(&inp_b, &decay_b, &c_b, &x_b, &z_b, &d,
            b_bench, l_bench, h, hd, ds));
    }
    let elapsed = start.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0 / n_iters as f64;
    let tps = (b_bench * l_bench) as f64 / (ms / 1000.0);

    println!("  wgpu Metal: {:.2}ms/scan, {:.0} tokens/sec", ms, tps);

    // CPU benchmark
    let start_cpu = Instant::now();
    for _ in 0..10 {
        let _ = ssm_scan_cpu(&inp_b, &decay_b, &c_b, &x_b, &z_b, &d,
            b_bench as usize, l_bench as usize, h as usize, hd as usize, ds as usize);
    }
    let cpu_ms = start_cpu.elapsed().as_secs_f64() * 1000.0 / 10.0;
    let cpu_tps = (b_bench * l_bench) as f64 / (cpu_ms / 1000.0);

    println!("  Rust CPU:   {:.2}ms/scan, {:.0} tokens/sec", cpu_ms, cpu_tps);
    println!("  Speedup:    {:.1}x (GPU vs CPU)", cpu_ms / ms);
}
