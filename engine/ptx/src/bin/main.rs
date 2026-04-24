//! PTX engine benchmark + correctness check.
//!
//!   cargo run --release --bin ptx-engine
//!   cargo run --release --bin ptx-engine -- --model /tmp/run_length_next.bin

use mamba3_engine::model::Mamba3Model;
use ptx_engine::{PtxContext, PtxModel};
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    println!("=== PTX Mamba-3 Engine ===");
    println!("Compiling kernels.cu (NVRTC, strict FP32)...");
    let t0 = Instant::now();
    let ptx = Arc::new(PtxContext::new()?);
    println!("  compiled in {:.2}s", t0.elapsed().as_secs_f64());
    println!();

    let cpu_model = if let Some(pos) = args.iter().position(|a| a == "--model") {
        let p = &args[pos + 1];
        println!("Loading model from {}", p);
        Mamba3Model::from_bin(Path::new(p))?
    } else {
        println!("No --model given. Using random d=32 L=1 tiny model.");
        Mamba3Model::new_random(32, 16, 16, 1, 260)
    };
    println!(
        "Model: d={}, L={}, d_state={}, headdim={}, vocab={}",
        cpu_model.d_model,
        cpu_model.n_layers,
        cpu_model.d_state,
        cpu_model.headdim,
        cpu_model.vocab_size
    );
    println!(
        "  d_inner={}, n_heads={}",
        cpu_model.d_inner, cpu_model.n_heads
    );
    println!();

    println!("Uploading weights to device...");
    let t0 = Instant::now();
    let gpu_model = PtxModel::from_cpu(&cpu_model, ptx.clone(), 64)?;
    println!("  uploaded in {:.2}s", t0.elapsed().as_secs_f64());
    println!();

    // Test tokens: BOS + "0 1 0" + SEP
    let tokens: Vec<u32> = vec![256, 48, 32, 49, 32, 48, 258];
    println!("Tokens: {:?}", tokens);

    // CPU reference
    let cpu_logits = cpu_model.forward(&tokens);

    // PTX forward (warmup)
    let _ = gpu_model.forward(&tokens)?;
    let ptx_logits = gpu_model.forward(&tokens)?;

    // Correctness
    let max_diff = cpu_logits
        .iter()
        .zip(ptx_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let status = if max_diff < 1e-3 { "PASS" } else { "FAIL" };
    println!(
        "Correctness (full forward):  max |PTX - CPU| = {:.3e}   {}",
        max_diff, status
    );
    if max_diff >= 1e-3 {
        // Diagnostic: show first few mismatches
        let mut shown = 0;
        for (i, (c, p)) in cpu_logits.iter().zip(ptx_logits.iter()).enumerate() {
            if (c - p).abs() > 1e-3 && shown < 10 {
                println!("  logit[{}]: CPU={:.6} PTX={:.6} diff={:.3e}", i, c, p, (c - p).abs());
                shown += 1;
            }
        }
    }
    println!();

    // Predictions
    let v = cpu_model.vocab_size;
    let l = tokens.len();
    println!("Predictions (top-1 per position):");
    for t in 0..l {
        let mut cpu_best = (0usize, f32::NEG_INFINITY);
        let mut ptx_best = (0usize, f32::NEG_INFINITY);
        for i in 0..v {
            if cpu_logits[t * v + i] > cpu_best.1 {
                cpu_best = (i, cpu_logits[t * v + i]);
            }
            if ptx_logits[t * v + i] > ptx_best.1 {
                ptx_best = (i, ptx_logits[t * v + i]);
            }
        }
        let ch = |idx: usize| {
            if idx < 256 {
                format!("'{}'", idx as u8 as char)
            } else {
                format!("special({})", idx)
            }
        };
        println!(
            "  pos {}:  CPU={} ({:.3})   PTX={} ({:.3})",
            t,
            ch(cpu_best.0),
            cpu_best.1,
            ch(ptx_best.0),
            ptx_best.1
        );
    }
    println!();

    // Benchmark
    println!("Benchmarks (each loops for ~2.5s):");
    println!(
        "{:>24}  {:>14}  {:>14}",
        "backend", "ms/inference", "tokens/sec"
    );

    // CPU
    for _ in 0..3 {
        let _ = cpu_model.forward(&tokens);
    }
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed().as_secs_f64() < 2.5 {
        let _ = cpu_model.forward(&tokens);
        count += 1;
    }
    let cpu_ms = start.elapsed().as_secs_f64() * 1000.0 / count as f64;
    let cpu_tps = tokens.len() as f64 / (cpu_ms / 1000.0);
    println!(
        "{:>24}  {:>14.3}  {:>14.0}",
        "CPU (mamba3-engine)", cpu_ms, cpu_tps
    );

    // PTX
    for _ in 0..5 {
        let _ = gpu_model.forward(&tokens)?;
    }
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed().as_secs_f64() < 2.5 {
        let _ = gpu_model.forward(&tokens)?;
        count += 1;
    }
    let ptx_ms = start.elapsed().as_secs_f64() * 1000.0 / count as f64;
    let ptx_tps = tokens.len() as f64 / (ptx_ms / 1000.0);
    println!(
        "{:>24}  {:>14.3}  {:>14.0}",
        "PTX (this engine)", ptx_ms, ptx_tps
    );

    println!();
    println!("Speedup PTX vs CPU: {:.2}x", cpu_ms / ptx_ms);

    Ok(())
}
