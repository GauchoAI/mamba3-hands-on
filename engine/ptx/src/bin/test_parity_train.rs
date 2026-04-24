//! Parity-training end-to-end test for PtxTrainer.
//! Matches the `run_parity_training` reference in engine/wgpu/src/main.rs.
//! Target: ≥95% accuracy after 5000 steps on 4-bit parity.

use mamba3_engine::model::Mamba3Model;
use ptx_engine::{PtxContext, PtxModel, PtxTrainer};
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== PTX Parity Training ===");
    let t0 = Instant::now();
    let ptx = Arc::new(PtxContext::new()?);
    println!("kernels compiled in {:.2}s", t0.elapsed().as_secs_f64());

    // Same config as run_parity_training in mamba3-engine.
    let cpu_model = Mamba3Model::new_random(32, 16, 16, 1, 260);
    println!("Model: {} params (d=32, L=1)", cpu_model.param_count());
    let gpu_model = PtxModel::from_cpu(&cpu_model, ptx.clone(), 16)?;

    let mut trainer = PtxTrainer::new(gpu_model, 1e-3, 0.1, 16)?;

    let mut rng_state: u64 = 12345;
    let mut rng = || -> u32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) & 1) as u32
    };

    let total_steps = 5000;
    let start = Instant::now();
    let mut best_acc = 0.0f32;

    for step in 0..total_steps {
        let n_bits = 4;
        let mut total_loss = 0.0f32;
        let batch = 16;

        for _ in 0..batch {
            let mut bits = Vec::new();
            let mut parity = 0u32;
            for _ in 0..n_bits {
                let b = rng();
                bits.push(b);
                parity ^= b;
            }
            let mut tokens: Vec<u32> = vec![256];
            for &b in &bits {
                tokens.push(48 + b);
            }
            tokens.push(258);
            let answer = if parity == 0 { 83 } else { 68 };
            tokens.push(answer);
            tokens.push(257);

            let mut targets = tokens[1..].to_vec();
            targets.push(257);

            total_loss += trainer.train_step(&tokens, &targets)?;
        }
        let loss = total_loss / batch as f32;

        if (step + 1) % 200 == 0 {
            let mut correct = 0usize;
            let mut total = 0usize;
            for _ in 0..200 {
                let mut test_bits = Vec::new();
                let mut test_parity = 0u32;
                for _ in 0..n_bits {
                    let b = rng();
                    test_bits.push(b);
                    test_parity ^= b;
                }
                let mut test_tokens: Vec<u32> = vec![256];
                for &b in &test_bits {
                    test_tokens.push(48 + b);
                }
                test_tokens.push(258);

                let logits = trainer.model.forward(&test_tokens)?;
                let v = trainer.model.vocab_size;
                let last = test_tokens.len() - 1;
                let mut best = (0usize, f32::NEG_INFINITY);
                for i in 0..v {
                    if logits[last * v + i] > best.1 {
                        best = (i, logits[last * v + i]);
                    }
                }
                let expected = if test_parity == 0 { 83 } else { 68 };
                if best.0 == expected { correct += 1; }
                total += 1;
            }

            let acc = correct as f32 / total as f32;
            best_acc = best_acc.max(acc);
            let elapsed = start.elapsed().as_secs_f64();
            println!("  step {:>4}: loss={:.4} acc={:.0}% best={:.0}% ({:.1}s)",
                step + 1, loss, acc * 100.0, best_acc * 100.0, elapsed);
        }
    }

    let total_time = start.elapsed();
    let ms_per_step = total_time.as_secs_f64() * 1000.0 / total_steps as f64;
    println!("\n  Final: best_acc={:.0}%  {:.1}ms/step  {:.0} steps/sec",
        best_acc * 100.0, ms_per_step, 1000.0 / ms_per_step);
    println!("  Status: {}", if best_acc > 0.95 { "PASS" } else if best_acc > 0.7 { "LEARNING" } else { "NEEDS TUNING" });

    Ok(())
}
