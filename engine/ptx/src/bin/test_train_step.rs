//! One PTX train_step vs CPU TrainState::train_step. Compares loss and
//! post-step params.

use mamba3_engine::model::Mamba3Model;
use mamba3_engine::train::TrainState;
use ptx_engine::{PtxContext, PtxModel, PtxTrainer};
use std::error::Error;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== PTX train_step vs CPU TrainState ===");
    let t0 = std::time::Instant::now();
    let ptx = Arc::new(PtxContext::new()?);
    println!("kernels compiled in {:.2}s", t0.elapsed().as_secs_f64());

    // Build a fresh random model (deterministic via LCG seeded at 42 inside Mamba3Model::new_random)
    let cpu_model = Mamba3Model::new_random(32, 16, 16, 1, 260);
    println!("model: d={}, L={}, d_state={}, headdim={}, vocab={}, di={}, H={}",
        cpu_model.d_model, cpu_model.n_layers, cpu_model.d_state,
        cpu_model.headdim, cpu_model.vocab_size, cpu_model.d_inner, cpu_model.n_heads);

    // Deep-copy for CPU training (the thread-local rng makes this tricky; just
    // build TWO identical models by re-seeding the LCG — since it's thread-local
    // to this process, building twice in sequence will give DIFFERENT weights.)
    // Workaround: export weights from model #1 and build model #2 with same weights.
    // We'll accomplish this by running Mamba3Model::new_random once and then
    // duplicating via collect_params / scatter_params.
    let mut cpu_model_b = Mamba3Model::new_random(32, 16, 16, 1, 260);
    let params_a = cpu_model.collect_params();
    cpu_model_b.scatter_params(&params_a);

    // Upload GPU model from cpu_model (not cpu_model_b)
    let gpu_model = PtxModel::from_cpu(&cpu_model_b, ptx.clone(), 64)?;

    // Sanity: initial params match between cpu_model and gpu_model? We can't easily
    // read back all GPU params; just trust from_cpu.

    let mut gpu_trainer = PtxTrainer::new(gpu_model, 1e-3, 0.1, 64)?;
    let mut cpu_trainer = TrainState::new(cpu_model, 1e-3, 0.1);

    let tokens: Vec<u32> = vec![256, 48, 32, 49, 32, 48, 258];
    let targets: Vec<u32> = vec![48, 32, 49, 32, 48, 258, 257];

    // Step 1: both run one train_step
    let cpu_loss = cpu_trainer.train_step(&tokens, &targets);
    let gpu_loss = gpu_trainer.train_step(&tokens, &targets)?;

    println!();
    println!("loss:  CPU={:.6}  GPU={:.6}  diff={:.3e}",
        cpu_loss, gpu_loss, (cpu_loss - gpu_loss).abs());

    // Compare logits after update by running one more forward
    let cpu_logits_after = cpu_trainer.model.forward(&tokens);
    let gpu_logits_after = gpu_trainer.model.forward(&tokens)?;
    let max_diff: f32 = cpu_logits_after.iter().zip(gpu_logits_after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("post-step logits max |CPU-GPU| = {:.3e}", max_diff);

    // Run more steps
    println!("\nRunning 10 steps total:");
    for step in 2..=10 {
        let cpu_l = cpu_trainer.train_step(&tokens, &targets);
        let gpu_l = gpu_trainer.train_step(&tokens, &targets)?;
        println!("  step {:>2}: CPU loss={:.4}  GPU loss={:.4}  diff={:.3e}",
            step, cpu_l, gpu_l, (cpu_l - gpu_l).abs());
    }

    Ok(())
}
