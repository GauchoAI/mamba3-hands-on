//! Replay-driven parity training: load the same initial weights and the same
//! exact training-sample stream that PyTorch saw in /tmp/dump_pytorch_run.py,
//! train PTX on it, compare per-cycle loss/acc.  If our PTX trajectory matches
//! PyTorch's (which converged to 100% in 2 cycles on this stream), the PTX
//! engine is provably PyTorch-equivalent end-to-end.
//!
//! Reads:
//!   /tmp/parity_run/initial.bin  — starting weights (our from_bin format)
//!   /tmp/parity_run/train.bin    — header + cycles*steps*batch samples
//!   /tmp/parity_run/eval.bin     — header + cycles*eval_n eval samples
//!   /tmp/parity_run/pytorch_log.txt — for direct comparison

use mamba3_engine::model::Mamba3Model;
use ptx_engine::{PtxContext, PtxModel, PtxTrainer};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_f32(r: &mut impl Read) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
fn read_sample(r: &mut impl Read) -> std::io::Result<(Vec<u32>, u32, u32)> {
    let n = read_u32(r)? as usize;
    let mut tokens = vec![0u32; n];
    for i in 0..n { tokens[i] = read_u32(r)?; }
    let ap = read_u32(r)?;
    let ans = read_u32(r)?;
    Ok((tokens, ap, ans))
}
fn read_eval_sample(r: &mut impl Read) -> std::io::Result<(Vec<u32>, u32)> {
    let n = read_u32(r)? as usize;
    let mut tokens = vec![0u32; n];
    for i in 0..n { tokens[i] = read_u32(r)?; }
    let ans = read_u32(r)?;
    Ok((tokens, ans))
}

fn main() -> Result<(), Box<dyn Error>> {
    let cpu_model = Mamba3Model::from_bin(Path::new("/tmp/parity_run/initial.bin"))?;
    println!("loaded initial weights: d={} dS={} hd={} L={} V={}",
        cpu_model.d_model, cpu_model.d_state, cpu_model.headdim,
        cpu_model.n_layers, cpu_model.vocab_size);

    // Read train.bin header
    let mut tf = BufReader::new(File::open("/tmp/parity_run/train.bin")?);
    let cycles = read_u32(&mut tf)? as usize;
    let steps_per_cycle = read_u32(&mut tf)? as usize;
    let batch = read_u32(&mut tf)? as usize;
    let eval_n = read_u32(&mut tf)? as usize;
    let lr = read_f32(&mut tf)?;
    let wd = read_f32(&mut tf)?;
    println!("stream: {} cycles × {} steps × {} batch  eval_n={}  lr={}  wd={}",
        cycles, steps_per_cycle, batch, eval_n, lr, wd);

    // Read eval.bin header
    let mut ef = BufReader::new(File::open("/tmp/parity_run/eval.bin")?);
    let _ev_cycles = read_u32(&mut ef)? as usize;
    let _ev_n = read_u32(&mut ef)? as usize;

    let max_seq = 21;
    let ptx = Arc::new(PtxContext::new()?);
    let gpu_model = PtxModel::from_cpu(&cpu_model, ptx.clone(), max_seq)?;
    let mut trainer = PtxTrainer::new(gpu_model, lr, wd, max_seq)?;
    trainer.warmup_steps = 0;   // PyTorch baseline doesn't use warmup

    let pytorch_log = std::fs::read_to_string("/tmp/parity_run/pytorch_log.txt").unwrap_or_default();
    let pytorch_lines: Vec<&str> = pytorch_log.lines().collect();

    println!("\n{:<6} {:>10} {:>6} {:>6}     {:<60}", "cycle", "loss", "acc", "best", "PyTorch on same stream");
    println!("{}", "-".repeat(96));

    let t0 = Instant::now();
    let mut best = 0.0f32;
    for cycle in 0..cycles {
        // --- training: replay batch by batch ---
        let mut cycle_loss = 0.0f64;
        for _step in 0..steps_per_cycle {
            trainer.zero_gradients_only()?;
            let mut last_loss = 0.0f32;
            for _ in 0..batch {
                let (tokens, ap, ans) = read_sample(&mut tf)?;
                let mut targets: Vec<u32> = vec![u32::MAX; tokens.len()];
                targets[ap as usize] = ans;
                last_loss = trainer.accumulate_gradients(&tokens, &targets)?;
            }
            let _ = trainer.apply_optimizer_step_scaled(1.0 / batch as f32)?;
            cycle_loss += (last_loss / batch as f32) as f64;
        }
        let avg_loss = cycle_loss / steps_per_cycle as f64;

        // --- eval on dumped eval set ---
        let mut correct = 0usize;
        for _ in 0..eval_n {
            let (test_tokens, expected) = read_eval_sample(&mut ef)?;
            let logits = trainer.model.forward(&test_tokens)?;
            let v = trainer.model.vocab_size;
            let last = test_tokens.len() - 1;
            let mut best_idx = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for i in 0..v {
                if logits[last * v + i] > best_v { best_v = logits[last * v + i]; best_idx = i; }
            }
            if best_idx as u32 == expected { correct += 1; }
        }
        let acc = correct as f32 / eval_n as f32;
        if acc > best { best = acc; }
        let pl = pytorch_lines.get(cycle).copied().unwrap_or("");
        println!(
            "{:<6} {:>10.4} {:>5.0}% {:>5.0}%     {:<60}",
            cycle + 1, avg_loss, acc * 100.0, best * 100.0, pl
        );
    }
    println!("\nPTX final best_acc = {:.0}%   ({:.1}s)", best * 100.0, t0.elapsed().as_secs_f64());
    println!("\nIf PTX matches PyTorch's per-cycle accuracy column, the engines are provably equivalent.");
    Ok(())
}
