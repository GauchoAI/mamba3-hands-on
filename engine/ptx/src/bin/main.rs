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
    let has_nan = ptx_logits.iter().any(|v| !v.is_finite());
    let max_diff = cpu_logits
        .iter()
        .zip(ptx_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |acc, d| if d.is_nan() { f32::INFINITY } else { acc.max(d) });
    let status = if !has_nan && max_diff < 1e-3 { "PASS" } else { "FAIL" };
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

    // PTX (per-call launches)
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
        "PTX (per-op launches)", ptx_ms, ptx_tps
    );

    // PTX with CUDA Graph
    println!("Capturing CUDA Graph (L={})...", tokens.len());
    let graph = gpu_model.capture_graph(tokens.len())?;

    // Correctness check: graph replay must match per-op path
    let graph_logits = gpu_model.forward_graph(&tokens, &graph)?;
    let graph_diff: f32 = graph_logits
        .iter()
        .zip(ptx_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "Graph vs per-op PTX: max diff = {:.3e}   {}",
        graph_diff,
        if graph_diff < 1e-5 { "PASS" } else { "FAIL" }
    );

    for _ in 0..5 {
        let _ = gpu_model.forward_graph(&tokens, &graph)?;
    }
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed().as_secs_f64() < 2.5 {
        let _ = gpu_model.forward_graph(&tokens, &graph)?;
        count += 1;
    }
    let graph_ms = start.elapsed().as_secs_f64() * 1000.0 / count as f64;
    let graph_tps = tokens.len() as f64 / (graph_ms / 1000.0);
    println!(
        "{:>24}  {:>14.3}  {:>14.0}",
        "PTX + CUDA Graph", graph_ms, graph_tps
    );

    // --- v2.2 per-op + graph + argmax (28-byte readback, no explicit sync) ---
    println!();
    println!("Testing argmax path (per-op + graph + argmax)...");
    let graph_am = gpu_model.capture_graph_argmax(tokens.len())?;
    let am_preds = gpu_model.forward_graph_argmax(&tokens, &graph_am)?;
    // Correctness vs CPU argmax
    let cpu_preds = cpu_model.predict(&tokens);
    let am_ok = am_preds == cpu_preds;
    println!("Argmax vs CPU predictions: {}  ({:?})",
        if am_ok { "PASS" } else { "FAIL" }, am_preds);

    for _ in 0..5 {
        let _ = gpu_model.forward_graph_argmax(&tokens, &graph_am)?;
    }
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed().as_secs_f64() < 2.5 {
        let _ = gpu_model.forward_graph_argmax(&tokens, &graph_am)?;
        count += 1;
    }
    let am_ms = start.elapsed().as_secs_f64() * 1000.0 / count as f64;
    let am_tps = tokens.len() as f64 / (am_ms / 1000.0);
    println!("{:>24}  {:>14.3}  {:>14.0}", "PTX + graph + argmax", am_ms, am_tps);
    println!(
        "  vs CPU ({:.3}ms): {:.2}x",
        cpu_ms, cpu_ms / am_ms
    );

    // --- v2.7 Persistent single-kernel forward ---
    println!();
    println!("Testing persistent single-kernel forward...");

    // Correctness
    let pk_logits = gpu_model.forward_persistent(&tokens)?;
    let pk_diff: f32 = cpu_logits
        .iter()
        .zip(pk_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |acc, d| if d.is_nan() { f32::INFINITY } else { acc.max(d) });
    println!(
        "Persistent vs CPU: max diff = {:.3e}   {}",
        pk_diff,
        if pk_diff < 1e-3 && pk_logits.iter().all(|v| v.is_finite()) { "PASS" } else { "FAIL" }
    );

    // Graph capture for persistent
    let graph_pk = gpu_model.capture_graph_persistent(tokens.len())?;
    let pkg_logits = gpu_model.forward_graph_persistent(&tokens, &graph_pk)?;
    let pkg_diff: f32 = cpu_logits
        .iter()
        .zip(pkg_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |acc, d| if d.is_nan() { f32::INFINITY } else { acc.max(d) });
    println!(
        "Persistent + Graph vs CPU: max diff = {:.3e}   {}",
        pkg_diff,
        if pkg_diff < 1e-3 && pkg_logits.iter().all(|v| v.is_finite()) { "PASS" } else { "FAIL" }
    );

    for _ in 0..5 {
        let _ = gpu_model.forward_persistent(&tokens)?;
    }
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed().as_secs_f64() < 2.5 {
        let _ = gpu_model.forward_persistent(&tokens)?;
        count += 1;
    }
    let pk_ms = start.elapsed().as_secs_f64() * 1000.0 / count as f64;
    let pk_tps = tokens.len() as f64 / (pk_ms / 1000.0);
    println!("{:>24}  {:>14.3}  {:>14.0}", "PTX persistent", pk_ms, pk_tps);

    for _ in 0..5 {
        let _ = gpu_model.forward_graph_persistent(&tokens, &graph_pk)?;
    }
    let start = Instant::now();
    let mut count = 0;
    while start.elapsed().as_secs_f64() < 2.5 {
        let _ = gpu_model.forward_graph_persistent(&tokens, &graph_pk)?;
        count += 1;
    }
    let pkg_ms = start.elapsed().as_secs_f64() * 1000.0 / count as f64;
    let pkg_tps = tokens.len() as f64 / (pkg_ms / 1000.0);
    println!("{:>24}  {:>14.3}  {:>14.0}", "PTX persistent + graph", pkg_ms, pkg_tps);

    println!();
    println!(
        "PTX persistent vs CPU: {:.2}x   (target > 1.00x to win)",
        cpu_ms / pk_ms
    );
    println!(
        "PTX persistent+graph vs CPU: {:.2}x   (target > 1.00x to win)",
        cpu_ms / pkg_ms
    );

    // --- v2.1 diagnostic: breakdown of forward_graph phases ---
    println!();
    println!("Phase breakdown (average of 200 calls):");
    let n_diag = 200;

    let mut t_upload = std::time::Duration::ZERO;
    let mut t_launch = std::time::Duration::ZERO;
    let mut t_sync = std::time::Duration::ZERO;
    let mut t_readback = std::time::Duration::ZERO;

    // Warmup.
    for _ in 0..5 {
        let _ = gpu_model.forward_graph_diag(&tokens, &graph)?;
    }
    for _ in 0..n_diag {
        let (du, dl, ds, dr) = gpu_model.forward_graph_diag(&tokens, &graph)?;
        t_upload += du;
        t_launch += dl;
        t_sync += ds;
        t_readback += dr;
    }
    let avg_us = |d: std::time::Duration| d.as_secs_f64() * 1e6 / n_diag as f64;
    let tot_us = avg_us(t_upload) + avg_us(t_launch) + avg_us(t_sync) + avg_us(t_readback);
    println!("{:>24}  {:>12.1} µs", "upload_tokens (htod)", avg_us(t_upload));
    println!("{:>24}  {:>12.1} µs", "graph.launch()", avg_us(t_launch));
    println!("{:>24}  {:>12.1} µs", "stream.synchronize()", avg_us(t_sync));
    println!("{:>24}  {:>12.1} µs", "memcpy_dtov (logits)", avg_us(t_readback));
    println!("{:>24}  {:>12.1} µs", "sum", tot_us);

    println!();
    println!(
        "PTX per-op    vs CPU: {:.2}x      vs wgpu-fused (21.2 ms): {:.1}x",
        cpu_ms / ptx_ms,
        21.2 / ptx_ms
    );
    println!(
        "PTX + graph   vs CPU: {:.2}x      vs wgpu-fused (21.2 ms): {:.1}x",
        cpu_ms / graph_ms,
        21.2 / graph_ms
    );

    Ok(())
}
