//! ptxd — PTX training daemon.
//!
//! Reads JSON job specifications from stdin (one per line), trains each job
//! sequentially on GPU via PTX, writes JSON results to stdout.  Owns the
//! CUDA context for its lifetime.  No oversubscription: jobs queue and run
//! one at a time, each with the full GPU. This is the minimum scheduler —
//! replaces the current `three_populations.py → spawn_worker` pattern which
//! oversubscribes the machine by launching many `specialist_trainer.py`
//! subprocesses simultaneously.
//!
//! JSON job shape:
//! ```json
//! {
//!   "id": "parity_001",
//!   "task": "parity",               // currently only "parity" supported
//!   "d_model": 32,
//!   "d_state": 16,
//!   "headdim": 16,
//!   "n_layers": 1,
//!   "vocab_size": 260,
//!   "lr": 0.001,
//!   "weight_decay": 0.1,
//!   "steps": 5000,
//!   "batch_size": 16,
//!   "n_bits": 4,
//!   "target_acc": 0.95,
//!   "seed": 12345
//! }
//! ```
//! Response JSON:
//! ```json
//! { "id": "parity_001", "final_loss": 0.72, "best_acc": 0.61,
//!   "ms_per_step": 3.1, "steps_executed": 5000, "status": "done" }
//! ```

use mamba3_engine::model::Mamba3Model;
use ptx_engine::{PtxContext, PtxModel, PtxTrainer};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::time::Instant;

#[derive(Deserialize, Debug)]
struct Job {
    id: String,
    task: String,
    #[serde(default = "default_d_model")]
    d_model: usize,
    #[serde(default = "default_d_state")]
    d_state: usize,
    #[serde(default = "default_headdim")]
    headdim: usize,
    #[serde(default = "default_n_layers")]
    n_layers: usize,
    #[serde(default = "default_vocab")]
    vocab_size: usize,
    #[serde(default = "default_lr")]
    lr: f32,
    #[serde(default = "default_wd")]
    weight_decay: f32,
    #[serde(default = "default_steps")]
    steps: usize,
    #[serde(default = "default_batch")]
    batch_size: usize,
    #[serde(default = "default_nbits")]
    n_bits: usize,
    #[serde(default = "default_target")]
    target_acc: f32,
    #[serde(default = "default_seed")]
    seed: u64,
}
fn default_d_model() -> usize { 32 }
fn default_d_state() -> usize { 16 }
fn default_headdim() -> usize { 16 }
fn default_n_layers() -> usize { 1 }
fn default_vocab() -> usize { 260 }
fn default_lr() -> f32 { 1e-3 }
fn default_wd() -> f32 { 0.1 }
fn default_steps() -> usize { 5000 }
fn default_batch() -> usize { 16 }
fn default_nbits() -> usize { 4 }
fn default_target() -> f32 { 0.95 }
fn default_seed() -> u64 { 12345 }

#[derive(Serialize, Debug)]
struct JobResult {
    id: String,
    status: String,
    #[serde(rename = "final_loss")]
    final_loss: f32,
    best_acc: f32,
    ms_per_step: f64,
    steps_executed: usize,
    wall_ms: f64,
}

fn run_job(ptx: &Arc<PtxContext>, job: &Job) -> Result<JobResult, Box<dyn Error>> {
    // Build model with the given config
    let cpu_model = Mamba3Model::new_random(
        job.d_model, job.d_state, job.headdim, job.n_layers, job.vocab_size,
    );
    let max_seq = 3 + job.n_bits + 2; // BOS + bits + SEP + answer + EOS
    let gpu_model = PtxModel::from_cpu(&cpu_model, ptx.clone(), max_seq.max(16))?;
    let mut trainer = PtxTrainer::new(gpu_model, job.lr, job.weight_decay, max_seq.max(16))?;

    match job.task.as_str() {
        "parity" => run_parity(&mut trainer, job),
        other => Err(format!("unsupported task: {}", other).into()),
    }
}

fn run_parity(trainer: &mut PtxTrainer, job: &Job) -> Result<JobResult, Box<dyn Error>> {
    let mut rng_state = job.seed;
    let mut rng = || -> u32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) & 1) as u32
    };

    let start = Instant::now();
    let mut best_acc = 0.0f32;
    let mut final_loss = 0.0f32;

    for step in 0..job.steps {
        let mut total_loss = 0.0f32;
        for _ in 0..job.batch_size {
            let mut bits = Vec::new();
            let mut parity = 0u32;
            for _ in 0..job.n_bits {
                let b = rng();
                bits.push(b);
                parity ^= b;
            }
            let mut tokens: Vec<u32> = vec![256];
            for &b in &bits { tokens.push(48 + b); }
            tokens.push(258);
            let answer = if parity == 0 { 83 } else { 68 };
            tokens.push(answer);
            tokens.push(257);

            let mut targets = tokens[1..].to_vec();
            targets.push(257);

            total_loss += trainer.train_step(&tokens, &targets)?;
        }
        let loss = total_loss / job.batch_size as f32;
        final_loss = loss;

        // Early-stop eval every 200 steps
        if (step + 1) % 200 == 0 {
            let mut correct = 0usize;
            for _ in 0..200 {
                let mut test_bits = Vec::new();
                let mut test_parity = 0u32;
                for _ in 0..job.n_bits {
                    let b = rng();
                    test_bits.push(b);
                    test_parity ^= b;
                }
                let mut test_tokens: Vec<u32> = vec![256];
                for &b in &test_bits { test_tokens.push(48 + b); }
                test_tokens.push(258);
                let logits = trainer.model.forward(&test_tokens)?;
                let v = trainer.model.vocab_size;
                let last = test_tokens.len() - 1;
                let mut best = (0usize, f32::NEG_INFINITY);
                for i in 0..v {
                    if logits[last * v + i] > best.1 { best = (i, logits[last * v + i]); }
                }
                let expected = if test_parity == 0 { 83 } else { 68 };
                if best.0 == expected { correct += 1; }
            }
            let acc = correct as f32 / 200.0;
            best_acc = best_acc.max(acc);
            if best_acc >= job.target_acc {
                let wall = start.elapsed().as_secs_f64() * 1000.0;
                return Ok(JobResult {
                    id: job.id.clone(),
                    status: "converged".into(),
                    final_loss,
                    best_acc,
                    ms_per_step: wall / (step + 1) as f64,
                    steps_executed: step + 1,
                    wall_ms: wall,
                });
            }
        }
    }
    let wall = start.elapsed().as_secs_f64() * 1000.0;
    Ok(JobResult {
        id: job.id.clone(),
        status: if best_acc >= 0.7 { "learning" } else { "needs_tuning" }.into(),
        final_loss,
        best_acc,
        ms_per_step: wall / job.steps as f64,
        steps_executed: job.steps,
        wall_ms: wall,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    eprintln!("[ptxd] compiling PTX kernels...");
    let t0 = Instant::now();
    let ptx = Arc::new(PtxContext::new()?);
    eprintln!("[ptxd] ready in {:.2}s, awaiting jobs on stdin (one JSON per line)",
        t0.elapsed().as_secs_f64());

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => { eprintln!("[ptxd] stdin error: {}", e); break; }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let job: Job = match serde_json::from_str(trimmed) {
            Ok(j) => j,
            Err(e) => {
                let err = serde_json::json!({
                    "status": "parse_error", "error": e.to_string()
                });
                writeln!(out, "{}", err)?;
                out.flush()?;
                continue;
            }
        };
        eprintln!("[ptxd] job {} starting ({} task, d={}, L={}, {} steps)",
            job.id, job.task, job.d_model, job.n_layers, job.steps);

        let result = match run_job(&ptx, &job) {
            Ok(r) => serde_json::to_value(&r)?,
            Err(e) => {
                serde_json::json!({
                    "id": job.id, "status": "error", "error": e.to_string()
                })
            }
        };
        writeln!(out, "{}", result)?;
        out.flush()?;
        eprintln!("[ptxd] job {} done", job.id);
    }
    Ok(())
}
