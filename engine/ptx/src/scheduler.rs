//! Slot scheduler — Phase 2.
//!
//! Owns the GPU. Accepts JSON job specs. Each job runs as a `JobRunner` on its
//! own CUDA stream so multiple jobs can co-execute in the same process. The
//! scheduler decides admission based on per-job memory / SM budget, then
//! drives all live runners forward step-by-step until each completes.
//!
//! Design rationale: see findings.md Entry 40.
//!
//! Usage (non-concurrent prototype — Step 3 of Entry 40):
//! ```ignore
//! let ctx = Arc::new(PtxContext::new()?);
//! let mut sched = Scheduler::new(ctx, /*max_concurrent=*/ 1);
//! sched.submit(job1)?;
//! sched.submit(job2)?;
//! while let Some(out) = sched.run_until_event()? {
//!     println!("{}", serde_json::to_string(&out)?);
//! }
//! ```
//!
//! The skeleton only does serial execution today (Steps 3–4 add the
//! concurrent-step semantics described in Entry 40).

use crate::runtime::PtxContext;
use crate::trainer::PtxTrainer;
use crate::model::PtxModel;
use cudarc::driver::CudaStream;
use mamba3_engine::model::Mamba3Model;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Stage {
    pub min_len: usize,
    pub max_len: usize,
    pub advance_at: f32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Job {
    pub id: String,
    pub task: String,
    #[serde(default = "d_default_d_model")] pub d_model: usize,
    #[serde(default = "d_default_d_state")] pub d_state: usize,
    #[serde(default = "d_default_headdim")] pub headdim: usize,
    #[serde(default = "d_default_n_layers")] pub n_layers: usize,
    #[serde(default = "d_default_vocab")]    pub vocab_size: usize,
    #[serde(default = "d_default_lr")]       pub lr: f32,
    #[serde(default = "d_default_wd")]       pub weight_decay: f32,
    #[serde(default = "d_default_steps")]    pub steps: usize,
    #[serde(default = "d_default_batch")]    pub batch_size: usize,
    #[serde(default = "d_default_nbits")]    pub n_bits: usize,
    #[serde(default = "d_default_target")]   pub target_acc: f32,
    #[serde(default = "d_default_seed")]     pub seed: u64,
    #[serde(default)] pub stages: Option<Vec<Stage>>,
}

fn d_default_d_model() -> usize { 32 }
fn d_default_d_state() -> usize { 16 }
fn d_default_headdim() -> usize { 16 }
fn d_default_n_layers() -> usize { 1 }
fn d_default_vocab() -> usize { 260 }
fn d_default_lr() -> f32 { 1e-3 }
fn d_default_wd() -> f32 { 0.1 }
fn d_default_steps() -> usize { 5000 }
fn d_default_batch() -> usize { 16 }
fn d_default_nbits() -> usize { 4 }
fn d_default_target() -> f32 { 0.95 }
fn d_default_seed() -> u64 { 12345 }

#[derive(Serialize, Debug, Clone)]
pub struct CycleEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub id: String,
    pub cycle: usize,
    pub step: usize,
    pub loss: f32,
    pub fresh_acc: f32,
    pub best_fresh: f32,
    pub stage: usize,
    pub elapsed_s: f64,
}

#[derive(Serialize, Debug, Clone)]
pub struct FinalEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub id: String,
    pub status: String,
    pub final_loss: f32,
    pub best_acc: f32,
    pub ms_per_step: f64,
    pub steps_executed: usize,
    pub wall_ms: f64,
}

#[derive(Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum SchedulerEvent {
    Cycle(CycleEvent),
    Final(FinalEvent),
}

/// One in-flight training job. Owns its model + trainer + dedicated CUDA
/// stream. The scheduler steps the runner forward; the runner reports
/// completion via its `state`.
pub struct JobRunner {
    pub job: Job,
    pub stream: Arc<CudaStream>,
    pub trainer: PtxTrainer,
    pub start: Instant,
    pub stage_idx: usize,
    pub stages: Vec<Stage>,
    pub step: usize,         // current training step (0-based)
    pub best_acc: f32,
    pub last_loss: f32,
    pub rng_state: u64,
    pub done: Option<FinalEvent>,
}

impl JobRunner {
    /// Build a runner for `job` on a fresh stream. The model is constructed
    /// with PyTorch-matching init (matches the recipe verified by
    /// parity-replay; see Entry 35).
    pub fn new(ctx: Arc<PtxContext>, job: Job) -> Result<Self, Box<dyn Error>> {
        let stream = ctx.new_stream()?;

        // Same init recipe as ptxd's apply_pytorch_init / test_parity_train.
        let mut cpu_model = Mamba3Model::new_random(
            job.d_model, job.d_state, job.headdim, job.n_layers, job.vocab_size,
        );
        apply_pytorch_init(&mut cpu_model, job.seed);

        let stages: Vec<Stage> = match &job.stages {
            Some(s) if !s.is_empty() => s.clone(),
            _ => vec![Stage { min_len: job.n_bits, max_len: job.n_bits, advance_at: 2.0 }],
        };
        let max_n_bits = stages.iter().map(|s| s.max_len).max().unwrap_or(job.n_bits);
        let max_seq = (1 + (2 * max_n_bits - 1).max(1) + 3).max(16);

        let gpu_model = PtxModel::from_cpu_on_stream(&cpu_model, ctx.clone(), stream.clone(), max_seq)?;
        let trainer = PtxTrainer::new(gpu_model, job.lr, job.weight_decay, max_seq)?;

        Ok(Self {
            stream,
            stages,
            stage_idx: 0,
            step: 0,
            best_acc: 0.0,
            last_loss: 0.0,
            rng_state: job.seed,
            start: Instant::now(),
            done: None,
            trainer,
            job,
        })
    }

    /// Train one mini-batch (`batch_size` accumulated samples + one AdamW
    /// step). Updates last_loss and step. Returns Some(CycleEvent) if this
    /// step landed on an eval boundary; Some(Final) if the job finished.
    pub fn advance_one_batch(&mut self) -> Result<Option<SchedulerEvent>, Box<dyn Error>> {
        if self.done.is_some() { return Ok(None); }
        let stage = self.stages[self.stage_idx].clone();

        self.trainer.zero_gradients_only()?;
        let mut last_loss = 0.0f32;
        for _ in 0..self.job.batch_size {
            let span = stage.max_len - stage.min_len + 1;
            self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let n_bits = stage.min_len + ((self.rng_state >> 33) as usize) % span;
            let mut bits = Vec::with_capacity(n_bits);
            let mut parity = 0u32;
            for _ in 0..n_bits {
                self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let b = ((self.rng_state >> 33) as u32) & 1;
                bits.push(b);
                parity ^= b;
            }
            let mut tokens: Vec<u32> = vec![256];
            for (i, &b) in bits.iter().enumerate() {
                if i > 0 { tokens.push(32); }
                tokens.push(48 + b);
            }
            tokens.push(258);
            let answer = if parity == 0 { 83 } else { 68 };
            tokens.push(answer);
            tokens.push(257);

            let mut targets: Vec<u32> = vec![u32::MAX; tokens.len()];
            let answer_pos = tokens.len() - 3;
            targets[answer_pos] = answer;

            last_loss = self.trainer.accumulate_gradients(&tokens, &targets)?;
        }
        self.trainer.apply_optimizer_step_scaled(1.0 / self.job.batch_size as f32)?;
        self.last_loss = last_loss / self.job.batch_size as f32;
        self.step += 1;

        // Eval every 200 steps. Returns Some(CycleEvent) at boundary,
        // Some(FinalEvent) on completion / convergence.
        if self.step % 200 == 0 {
            let acc = self.eval(200)?;
            if acc > self.best_acc { self.best_acc = acc; }
            let stage = &self.stages[self.stage_idx];
            let cycle = self.step / 200;
            let event = CycleEvent {
                kind: "cycle",
                id: self.job.id.clone(),
                cycle,
                step: self.step,
                loss: self.last_loss,
                fresh_acc: acc,
                best_fresh: self.best_acc,
                stage: self.stage_idx,
                elapsed_s: self.start.elapsed().as_secs_f64(),
            };
            // Curriculum advance.
            if acc >= stage.advance_at && self.stage_idx + 1 < self.stages.len() {
                self.stage_idx += 1;
            }
            // Convergence check.
            if self.best_acc >= self.job.target_acc && self.stage_idx + 1 == self.stages.len() {
                self.done = Some(self.make_final("converged"));
            }
            return Ok(Some(SchedulerEvent::Cycle(event)));
        }
        // Step budget exhausted.
        if self.step >= self.job.steps {
            self.done = Some(self.make_final(
                if self.best_acc >= 0.7 { "learning" } else { "needs_tuning" }
            ));
        }
        Ok(None)
    }

    fn make_final(&self, status: &str) -> FinalEvent {
        let wall_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        FinalEvent {
            kind: "final",
            id: self.job.id.clone(),
            status: status.to_string(),
            final_loss: self.last_loss,
            best_acc: self.best_acc,
            ms_per_step: wall_ms / self.step as f64,
            steps_executed: self.step,
            wall_ms,
        }
    }

    fn eval(&mut self, n_eval: usize) -> Result<f32, Box<dyn Error>> {
        let stage = self.stages[self.stage_idx].clone();
        let mut correct = 0usize;
        for _ in 0..n_eval {
            let span = stage.max_len - stage.min_len + 1;
            self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let n_bits = stage.min_len + ((self.rng_state >> 33) as usize) % span;
            let mut test_bits = Vec::with_capacity(n_bits);
            let mut test_parity = 0u32;
            for _ in 0..n_bits {
                self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let b = ((self.rng_state >> 33) as u32) & 1;
                test_bits.push(b);
                test_parity ^= b;
            }
            let mut test_tokens: Vec<u32> = vec![256];
            for (i, &b) in test_bits.iter().enumerate() {
                if i > 0 { test_tokens.push(32); }
                test_tokens.push(48 + b);
            }
            test_tokens.push(258);
            let logits = self.trainer.model.forward(&test_tokens)?;
            let v = self.trainer.model.vocab_size;
            let last = test_tokens.len() - 1;
            let mut best_idx = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for i in 0..v {
                if logits[last * v + i] > best_v { best_v = logits[last * v + i]; best_idx = i; }
            }
            let expected = if test_parity == 0 { 83 } else { 68 };
            if best_idx as u32 == expected { correct += 1; }
        }
        Ok(correct as f32 / n_eval as f32)
    }

    pub fn is_done(&self) -> bool { self.done.is_some() }
    pub fn take_final(&mut self) -> Option<FinalEvent> { self.done.take() }
}

/// Slot scheduler — drives a fixed-capacity pool of JobRunners. Currently
/// serial (Step 3 of Entry 40 plan); Step 4 will make `pump_one_step`
/// advance every runner concurrently on its own stream.
pub struct Scheduler {
    pub ctx: Arc<PtxContext>,
    pub max_concurrent: usize,
    pub queue: VecDeque<Job>,
    pub running: Vec<JobRunner>,
}

impl Scheduler {
    pub fn new(ctx: Arc<PtxContext>, max_concurrent: usize) -> Self {
        Self { ctx, max_concurrent, queue: VecDeque::new(), running: Vec::new() }
    }

    pub fn submit(&mut self, job: Job) {
        self.queue.push_back(job);
    }

    /// Admit jobs from the queue while we have spare slots.
    fn admit(&mut self) -> Result<(), Box<dyn Error>> {
        while self.running.len() < self.max_concurrent {
            let Some(job) = self.queue.pop_front() else { break; };
            let runner = JobRunner::new(self.ctx.clone(), job)?;
            self.running.push(runner);
        }
        Ok(())
    }

    /// Pump ONE batch on each running job. Returns all events produced
    /// (cycle and final) in submission order.
    pub fn pump_one_step(&mut self) -> Result<Vec<SchedulerEvent>, Box<dyn Error>> {
        self.admit()?;
        let mut events = Vec::new();
        let mut i = 0;
        while i < self.running.len() {
            if let Some(ev) = self.running[i].advance_one_batch()? {
                events.push(ev);
            }
            if self.running[i].is_done() {
                let mut runner = self.running.remove(i);
                if let Some(f) = runner.take_final() {
                    events.push(SchedulerEvent::Final(f));
                }
                continue;
            }
            i += 1;
        }
        // Top up new jobs as slots free.
        self.admit()?;
        Ok(events)
    }

    pub fn is_idle(&self) -> bool { self.running.is_empty() && self.queue.is_empty() }
}

/// PyTorch-matching init recipe (mirrors ptxd::apply_pytorch_init).
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
