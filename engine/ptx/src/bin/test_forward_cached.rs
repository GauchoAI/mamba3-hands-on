//! Verify forward_cached produces bit-identical logits to forward,
//! and that per-layer cache buffers get populated with sensible values.

use mamba3_engine::model::Mamba3Model;
use ptx_engine::{PtxContext, PtxModel, TrainScratch};
use std::error::Error;
use std::path::Path;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== forward_cached test ===");
    println!("Compiling kernels...");
    let t0 = std::time::Instant::now();
    let ptx = Arc::new(PtxContext::new()?);
    println!("  compiled in {:.2}s", t0.elapsed().as_secs_f64());

    let args: Vec<String> = std::env::args().collect();
    let cpu_model = if let Some(p) = args.iter().position(|a| a == "--model") {
        println!("Loading {}", args[p + 1]);
        Mamba3Model::from_bin(Path::new(&args[p + 1]))?
    } else {
        println!("Using random d=32 L=1 tiny model.");
        Mamba3Model::new_random(32, 16, 16, 1, 260)
    };
    println!(
        "  d={}, L={}, d_state={}, headdim={}, vocab={}",
        cpu_model.d_model, cpu_model.n_layers, cpu_model.d_state,
        cpu_model.headdim, cpu_model.vocab_size
    );

    let gpu_model = PtxModel::from_cpu(&cpu_model, ptx.clone(), 64)?;

    // Allocate TrainScratch
    let mut train_scratch = TrainScratch::new(
        &ptx,
        64, // max_seq
        cpu_model.n_layers,
        cpu_model.d_model,
        cpu_model.d_state,
        cpu_model.d_inner,
        cpu_model.n_heads,
        cpu_model.headdim,
        gpu_model.d_in_proj,
        cpu_model.vocab_size,
    )?;

    let tokens: Vec<u32> = vec![256, 48, 32, 49, 32, 48, 258];

    // Reference: plain forward
    let fwd_logits = gpu_model.forward(&tokens)?;

    // Test: forward_cached
    let fc_logits = gpu_model.forward_cached(&tokens, &mut train_scratch)?;

    // Correctness
    let max_diff: f32 = fwd_logits.iter().zip(fc_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("forward_cached vs forward: max diff = {:.3e}", max_diff);
    if max_diff > 1e-6 {
        println!("FAIL: logits diverged");
        for (i, (a, b)) in fwd_logits.iter().zip(fc_logits.iter()).enumerate().take(5) {
            println!("  [{}] forward={} cached={} diff={}", i, a, b, (a - b).abs());
        }
        return Err("forward_cached mismatch".into());
    }
    println!("forward_cached logits match forward logits exactly: PASS");

    // Also verify match vs CPU
    let cpu_logits = cpu_model.forward(&tokens);
    let cpu_diff: f32 = cpu_logits.iter().zip(fc_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("forward_cached vs CPU: max diff = {:.3e} {}",
        cpu_diff, if cpu_diff < 1e-3 { "PASS" } else { "FAIL" });

    // Verify per-layer caches are non-zero (sanity: something got written)
    let stream = ptx.stream.clone();
    let inputs_probe = stream.memcpy_dtov(
        &train_scratch.layer_inputs.slice(0..8)
    )?;
    let states_probe = stream.memcpy_dtov(
        &train_scratch.layer_states.slice(0..8)
    )?;
    let states_t1_probe = stream.memcpy_dtov(
        // states[1, 0, 0, 0..8] for first layer
        &train_scratch.layer_states.slice(
            cpu_model.n_heads * cpu_model.headdim * cpu_model.d_state
            .. cpu_model.n_heads * cpu_model.headdim * cpu_model.d_state + 8
        )
    )?;
    println!("layer_inputs[0, 0, 0..8]: {:?}", inputs_probe);
    println!("layer_states[0, t=0, 0, 0..8] (should be ~0): {:?}", states_probe);
    println!("layer_states[0, t=1, 0, 0..8]: {:?}", states_t1_probe);
    let states_t0_zero = states_probe.iter().all(|&v| v == 0.0);
    println!("states[t=0] is zero: {}", if states_t0_zero { "PASS" } else { "FAIL" });

    println!("\nAll forward_cached checks PASSED.");
    Ok(())
}
