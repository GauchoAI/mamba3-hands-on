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

#[derive(Deserialize, Debug, Clone)]
struct Stage {
    min_len: usize,
    max_len: usize,
    advance_at: f32,
}

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
    /// Optional curriculum: list of (min_len, max_len, advance_at) stages.
    /// If present, ptxd advances through stages as training accuracy crosses
    /// each stage's advance_at threshold; n_bits per sample is sampled
    /// uniformly in [min_len, max_len]. Mirrors problems/parity/problem.yaml.
    /// Without this field, ptxd uses fixed-length parity at `n_bits`.
    #[serde(default)]
    stages: Option<Vec<Stage>>,
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
    #[serde(rename = "type")]
    kind: &'static str,
    id: String,
    status: String,
    #[serde(rename = "final_loss")]
    final_loss: f32,
    best_acc: f32,
    ms_per_step: f64,
    steps_executed: usize,
    wall_ms: f64,
}

/// Per-eval streaming row, emitted every 200 steps so an orchestrator can
/// monitor convergence in real time (matches MetricsWriter.log_cycle's
/// shape: cycle, loss, fresh_acc, best_fresh).
#[derive(Serialize, Debug)]
struct CycleRow {
    #[serde(rename = "type")]
    kind: &'static str,
    id: String,
    cycle: usize,
    step: usize,
    loss: f32,
    fresh_acc: f32,
    best_fresh: f32,
    stage: usize,
    elapsed_s: f64,
}

fn run_job(ptx: &Arc<PtxContext>, job: &Job) -> Result<JobResult, Box<dyn Error>> {
    // Build the model and apply the PyTorch-matching init recipe (Entry 35
    // proved this is required to converge in the small number of cycles
    // the GA expects). dt_bias log-uniform, embed N(0,1), Linear kaiming-
    // uniform, scale=0.1.
    let mut cpu_model = Mamba3Model::new_random(
        job.d_model, job.d_state, job.headdim, job.n_layers, job.vocab_size,
    );
    apply_pytorch_init(&mut cpu_model, job.seed);

    // Sequence layout (matching test_parity_train, which converges on
    // parity-replay): [BOS, bit, SPC, bit, SPC, ..., SEP, ANSWER, EOS].
    // Budget for the largest stage in the curriculum (or fixed n_bits when
    // no curriculum is supplied).
    let max_n_bits = match &job.stages {
        Some(s) if !s.is_empty() => s.iter().map(|st| st.max_len).max().unwrap_or(job.n_bits),
        _ => job.n_bits,
    };
    let max_seq = 1 + (2 * max_n_bits - 1).max(1) + 3;

    let gpu_model = PtxModel::from_cpu(&cpu_model, ptx.clone(), max_seq.max(16))?;
    let mut trainer = PtxTrainer::new(gpu_model, job.lr, job.weight_decay, max_seq.max(16))?;
    trainer.warmup_steps = 0;  // PyTorch baseline doesn't warmup; matches parity-replay.

    match job.task.as_str() {
        "parity" => run_parity(&mut trainer, job),
        other => Err(format!("unsupported task: {}", other).into()),
    }
}

/// PyTorch-matching init: dt_bias log-uniform per head, embed N(0,1) via
/// Box-Muller, in_proj/out_proj kaiming-uniform U(±√(1/fan_in)), scale=0.1.
/// Identical to the recipe in test_parity_train that converges via
/// parity-replay against PyTorch's own training trajectory.
fn apply_pytorch_init(model: &mut Mamba3Model, seed: u64) {
    let mut s: u64 = seed;
    let mut lcg = || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as u32) as f32 / u32::MAX as f32
    };
    let dt_min: f32 = 0.001;
    let dt_max: f32 = 0.1;
    let log_dt_min = dt_min.ln();
    let log_dt_max = dt_max.ln();
    for layer in model.layers.iter_mut() {
        for h in 0..layer.dt_bias.len() {
            let r = lcg();
            let dt_h = (r * (log_dt_max - log_dt_min) + log_dt_min).exp().max(1e-4);
            layer.dt_bias[h] = dt_h + (-(-dt_h).exp_m1()).ln();
        }
    }
    let mut next_normal = || -> f32 {
        let u1 = lcg().max(1e-30);
        let u2 = lcg();
        ((-2.0 * u1.ln()).sqrt()) * (2.0 * std::f32::consts::PI * u2).cos()
    };
    for w in model.embed_w.iter_mut() {
        *w = next_normal();
    }
    for layer in model.layers.iter_mut() {
        layer.scale = 0.1;
    }
    let d = model.d_model;
    for layer in model.layers.iter_mut() {
        let di = 2 * d;
        let in_proj_bound = (1.0f32 / d as f32).sqrt();
        for w in layer.in_proj_w.iter_mut() {
            *w = (lcg() * 2.0 - 1.0) * in_proj_bound;
        }
        let out_proj_bound = (1.0f32 / di as f32).sqrt();
        for w in layer.out_proj_w.iter_mut() {
            *w = (lcg() * 2.0 - 1.0) * out_proj_bound;
        }
    }
}

fn run_parity(trainer: &mut PtxTrainer, job: &Job) -> Result<JobResult, Box<dyn Error>> {
    let mut rng_state = job.seed;
    let mut next_u32 = |state: &mut u64| -> u32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state >> 33) as u32
    };
    // Resolve curriculum: explicit stages, or a single fixed-length stage.
    let stages: Vec<Stage> = match &job.stages {
        Some(s) if !s.is_empty() => s.clone(),
        _ => vec![Stage {
            min_len: job.n_bits,
            max_len: job.n_bits,
            advance_at: 2.0,  // never advances
        }],
    };
    let mut stage_idx = 0usize;

    let start = Instant::now();
    let mut best_acc = 0.0f32;
    let mut final_loss = 0.0f32;

    for step in 0..job.steps {
        let stage = &stages[stage_idx];
        // Mini-batch with gradient accumulation: zero grads, accumulate B
        // samples, then ONE AdamW step scaled by 1/B. Matches PyTorch's
        // batched-backward semantics. Without this, per-sample SGD on
        // mixed-pattern data makes the optimizer thrash (Entry 33).
        trainer.zero_gradients_only()?;
        let mut last_loss = 0.0f32;
        for _ in 0..job.batch_size {
            // Sample n_bits uniformly in [min_len, max_len]
            let span = stage.max_len - stage.min_len + 1;
            let n_bits = stage.min_len + (next_u32(&mut rng_state) as usize) % span;
            let mut bits = Vec::with_capacity(n_bits);
            let mut parity = 0u32;
            for _ in 0..n_bits {
                let b = next_u32(&mut rng_state) & 1;
                bits.push(b);
                parity ^= b;
            }
            // Token layout matches test_parity_train: [BOS, bit, SPC, bit, ..., SEP, ANSWER, EOS]
            let mut tokens: Vec<u32> = vec![256];
            for (i, &b) in bits.iter().enumerate() {
                if i > 0 { tokens.push(32); }
                tokens.push(48 + b);
            }
            tokens.push(258);
            let answer = if parity == 0 { 83 } else { 68 };
            tokens.push(answer);
            tokens.push(257);

            // Masked CE: only supervise the SEP position predicting ANSWER.
            let mut targets: Vec<u32> = vec![u32::MAX; tokens.len()];
            let answer_pos = tokens.len() - 3;
            targets[answer_pos] = answer;

            last_loss = trainer.accumulate_gradients(&tokens, &targets)?;
        }
        trainer.apply_optimizer_step_scaled(1.0 / job.batch_size as f32)?;
        let loss = last_loss / job.batch_size as f32;
        final_loss = loss;

        // Early-stop eval every 200 steps.
        if (step + 1) % 200 == 0 {
            let mut correct = 0usize;
            for _ in 0..200 {
                let span = stage.max_len - stage.min_len + 1;
                let n_bits = stage.min_len + (next_u32(&mut rng_state) as usize) % span;
                let mut test_bits = Vec::with_capacity(n_bits);
                let mut test_parity = 0u32;
                for _ in 0..n_bits {
                    let b = next_u32(&mut rng_state) & 1;
                    test_bits.push(b);
                    test_parity ^= b;
                }
                let mut test_tokens: Vec<u32> = vec![256];
                for (i, &b) in test_bits.iter().enumerate() {
                    if i > 0 { test_tokens.push(32); }
                    test_tokens.push(48 + b);
                }
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

            // Stream a per-eval row to stdout (one JSON object per line)
            // BEFORE deciding to advance/stop. Lets the orchestrator monitor
            // convergence in real time.
            let cycle_row = CycleRow {
                kind: "cycle",
                id: job.id.clone(),
                cycle: (step + 1) / 200,
                step: step + 1,
                loss: final_loss,
                fresh_acc: acc,
                best_fresh: best_acc,
                stage: stage_idx,
                elapsed_s: start.elapsed().as_secs_f64(),
            };
            let line = serde_json::to_string(&cycle_row).unwrap();
            println!("{}", line);
            std::io::stdout().flush().ok();

            // Curriculum advance: if accuracy on this stage crosses its
            // advance_at threshold, move to the next stage. Final stage's
            // advance_at acts as the global target_acc check.
            if acc >= stage.advance_at && stage_idx + 1 < stages.len() {
                stage_idx += 1;
                eprintln!("[ptxd] {} advanced to stage {}/{} at step {} (acc={:.2})",
                    job.id, stage_idx + 1, stages.len(), step + 1, acc);
            }

            if best_acc >= job.target_acc && stage_idx + 1 == stages.len() {
                let wall = start.elapsed().as_secs_f64() * 1000.0;
                return Ok(JobResult {
                    kind: "final",
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
        kind: "final",
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
