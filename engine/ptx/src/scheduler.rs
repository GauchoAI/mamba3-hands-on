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

/// Periodic resource snapshot — emitted by the scheduler at most every
/// `tick_interval_s` (default 1.0).  Designed for compact telemetry: ~50
/// bytes serialised, 1Hz cadence ≈ 4KB/min ≈ 6MB/day. Mirrors
/// firebase_push.push_gpu_tick's shape so an external uploader can forward
/// these straight to Firebase Realtime DB without reformatting.
#[derive(Serialize, Debug, Clone)]
pub struct TickEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,           // "tick"
    pub t: f64,                       // seconds since scheduler start
    pub mem_pct: f32,                 // 0..100
    pub sm_pct: f32,                  // 0..100
    pub running: usize,
    pub queue: usize,
}

#[derive(Serialize, Debug, Clone)]
#[serde(untagged)]
pub enum SchedulerEvent {
    Cycle(CycleEvent),
    Final(FinalEvent),
    Tick(TickEvent),
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

    /// Single-shot batch (kept for backwards compatibility / test_scheduler).
    /// Internally just calls prepare + finalize back-to-back, so it has
    /// the same behaviour but is NOT the path the slot scheduler uses for
    /// concurrent execution. Use `prepare_one_batch` + `finalize_one_batch`
    /// directly when you want N runners to overlap on the GPU.
    pub fn advance_one_batch(&mut self) -> Result<Option<SchedulerEvent>, Box<dyn Error>> {
        self.prepare_one_batch()?;
        self.finalize_one_batch()
    }

    /// Phase 1: launch the batch's forward+backward kernels onto this
    /// runner's stream. NO syncs. Returns immediately after queueing.
    /// The scheduler calls this on every runner BEFORE finalize, so all
    /// streams are running in parallel by the time anyone syncs.
    pub fn prepare_one_batch(&mut self) -> Result<(), Box<dyn Error>> {
        if self.done.is_some() { return Ok(()); }
        let stage = self.stages[self.stage_idx].clone();
        self.trainer.zero_gradients_only()?;
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
            let _ = self.trainer.accumulate_gradients(&tokens, &targets)?;
        }
        Ok(())
    }

    /// Phase 2: AdamW + step bookkeeping + (at eval boundary) read loss
    /// and run eval. Apply_optimizer_step_scaled has internal syncs for the
    /// global-norm clip and per-layer d_scale read — those block ONLY on
    /// this runner's stream, so concurrent runners are unaffected (their
    /// kernels keep running on their own streams during this sync).
    pub fn finalize_one_batch(&mut self) -> Result<Option<SchedulerEvent>, Box<dyn Error>> {
        if self.done.is_some() { return Ok(None); }
        self.trainer.apply_optimizer_step_scaled(1.0 / self.job.batch_size as f32)?;
        self.step += 1;

        if self.step % 200 == 0 {
            self.last_loss = self.trainer.read_last_loss_blocking()?
                / self.job.batch_size as f32;
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
            if acc >= stage.advance_at && self.stage_idx + 1 < self.stages.len() {
                self.stage_idx += 1;
            }
            if self.best_acc >= self.job.target_acc && self.stage_idx + 1 == self.stages.len() {
                self.done = Some(self.make_final("converged"));
            }
            return Ok(Some(SchedulerEvent::Cycle(event)));
        }
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

// ---- Resource estimation --------------------------------------------------
//
// The H100 has two main resources we budget against:
//   * memory (HBM)   — 80 GB; we reserve ~64 GB to leave room for system
//   * compute (SMs)  — 132 SMs
//
// Per-job estimates are deterministic from the Job config: we know the model
// shape, so we know weights, Adam moments, gradient buffers, and activation
// cache exactly. SM consumption is harder to model precisely (depends on
// kernel mix and how the GPU scheduler co-issues blocks), so we use a coarse
// "bytes-of-work-per-step" proxy.

const H100_MEM_BUDGET: usize = 64 * 1024 * 1024 * 1024;  // 64 GB usable
const H100_SM_BUDGET:  usize = 132;                        // hardware

/// Bytes of GPU memory a job will consume once a JobRunner is built.
/// Includes weights, AdamW moment estimates, gradient buffers, and the
/// per-layer activation cache used by the cached forward + full backward.
pub fn estimate_job_memory_bytes(job: &Job) -> usize {
    let d = job.d_model;
    let ds = job.d_state;
    let nh = (2 * d) / job.headdim;             // d_inner / headdim
    let di = 2 * d;                              // d_inner
    let dip = 2 * di + 2 * ds + 3 * nh + ds / 2; // matches model.rs layout
    let v = job.vocab_size;
    let l_max = match &job.stages {
        Some(s) if !s.is_empty() => s.iter().map(|st| st.max_len).max().unwrap_or(job.n_bits),
        _ => job.n_bits,
    };
    let max_seq = (1 + (2 * l_max - 1).max(1) + 3).max(16);

    // Weights (rough; doesn't double-count the per-layer concat copies).
    let weights = v * d                                                  // embed
                + 4 * d                                                  // embed_norm + final_norm w/b
                + job.n_layers * (dip * d                                // in_proj_w
                                + d * di                                 // out_proj_w
                                + 4 * nh                                 // dt_bias + d_param + 2 unused
                                + 4 * ds                                 // b/c norm w/b
                                + 2 * d);                                // layer_norm w/b
    // AdamW: m + v moments are 2× weights.
    let adam_mv = 2 * weights;
    // Activation cache for the full backward (per-layer x_normed, projs, y_inners,
    // bps, cps, decays, dts, traps, states):
    let activations = job.n_layers * (
        max_seq * d                            // layer_inputs
      + max_seq * d                            // layer_x_normed
      + max_seq * dip                          // layer_projs
      + max_seq * di                           // layer_y_inners
      + max_seq * d                            // layer_y_post
      + max_seq * ds * 2                       // layer_bps + layer_cps
      + max_seq * nh * 3                       // layer_decays + layer_dts + layer_traps
      + (max_seq + 1) * nh * job.headdim * ds  // layer_states (the big one)
    );
    // Gradient buffers ≈ weights (we keep a d_* for everything that learns).
    let grads = weights;
    (weights + adam_mv + activations + grads) * 4   // f32 = 4 bytes
}

/// Coarse SM-blocks estimate: a Mamba-3 backward step launches matmul tiles
/// at (dip/16 × max_seq/16), an SSM scan at (n_heads, hd*ds), and various
/// LN/scatter passes. We use the matmul tile count as the dominant term —
/// the SSM scan blocks are tiny (≤ n_heads). For our specialists this is
/// 30-100 blocks per job, well under the H100's 132-SM budget.
pub fn estimate_job_sm_blocks(job: &Job) -> usize {
    let d = job.d_model;
    let nh = (2 * d) / job.headdim;
    let di = 2 * d;
    let dip = 2 * di + 2 * job.d_state + 3 * nh + job.d_state / 2;
    let l_max = match &job.stages {
        Some(s) if !s.is_empty() => s.iter().map(|st| st.max_len).max().unwrap_or(job.n_bits),
        _ => job.n_bits,
    };
    let max_seq = (1 + (2 * l_max - 1).max(1) + 3).max(16);
    let matmul_blocks = ((dip + 15) / 16) * ((max_seq + 15) / 16);
    matmul_blocks + nh
}

/// Snapshot of a runner suitable for the visualization renderer.
#[derive(Clone, Debug)]
pub struct SlotView {
    pub id: String,
    pub mem_bytes: usize,
    pub sm_blocks: usize,
    pub step: usize,
    pub max_steps: usize,
    pub best_acc: f32,
    pub last_loss: f32,
    pub elapsed_s: f64,
}

/// Slot scheduler — drives a fixed-capacity pool of JobRunners with
/// memory + SM budget tracking and a renderable resource view.
pub struct Scheduler {
    pub ctx: Arc<PtxContext>,
    pub max_concurrent: usize,
    pub mem_budget: usize,
    pub sm_budget: usize,
    pub queue: VecDeque<Job>,
    pub running: Vec<JobRunner>,
    pub used_mem: usize,
    pub used_sm: usize,
    pub start: Instant,
    /// Telemetry: emit a TickEvent at most every `tick_interval_s`. Set to
    /// f64::INFINITY to disable. Default 1.0s.
    pub tick_interval_s: f64,
    last_tick: f64,
}

impl Scheduler {
    pub fn new(ctx: Arc<PtxContext>, max_concurrent: usize) -> Self {
        Self {
            ctx, max_concurrent,
            mem_budget: H100_MEM_BUDGET,
            sm_budget: H100_SM_BUDGET,
            queue: VecDeque::new(),
            running: Vec::new(),
            used_mem: 0,
            used_sm: 0,
            start: Instant::now(),
            tick_interval_s: 1.0,
            last_tick: 0.0,
        }
    }

    /// Generate a TickEvent if the tick_interval has elapsed since the
    /// last one. Cheap; the scheduler calls this from pump_one_step.
    fn maybe_emit_tick(&mut self) -> Option<TickEvent> {
        let t = self.start.elapsed().as_secs_f64();
        if t - self.last_tick < self.tick_interval_s { return None; }
        self.last_tick = t;
        Some(TickEvent {
            kind: "tick",
            t,
            mem_pct: (self.used_mem as f64 / self.mem_budget as f64 * 100.0).min(100.0) as f32,
            sm_pct:  (self.used_sm  as f64 / self.sm_budget  as f64 * 100.0).min(100.0) as f32,
            running: self.running.len(),
            queue:   self.queue.len(),
        })
    }

    /// Override the memory/SM budgets (e.g. for a smaller GPU or a test).
    pub fn with_budget(mut self, mem_bytes: usize, sm_blocks: usize) -> Self {
        self.mem_budget = mem_bytes;
        self.sm_budget = sm_blocks;
        self
    }

    pub fn submit(&mut self, job: Job) {
        self.queue.push_back(job);
    }

    /// Admit jobs from the queue while we have spare slots AND the next
    /// job's resource estimate fits in the remaining budget. If a job is
    /// too big to ever fit (estimate > total budget) it stays at the head
    /// of the queue and we don't deadlock — caller's responsibility.
    fn admit(&mut self) -> Result<(), Box<dyn Error>> {
        while self.running.len() < self.max_concurrent {
            let Some(peek) = self.queue.front() else { break; };
            let need_mem = estimate_job_memory_bytes(peek);
            let need_sm  = estimate_job_sm_blocks(peek);
            if self.used_mem + need_mem > self.mem_budget { break; }
            if self.used_sm  + need_sm  > self.sm_budget  { break; }
            let job = self.queue.pop_front().unwrap();
            let runner = JobRunner::new(self.ctx.clone(), job)?;
            self.used_mem += need_mem;
            self.used_sm  += need_sm;
            self.running.push(runner);
        }
        Ok(())
    }

    /// Snapshot the live runners + queue for the renderer. Cheap (just
    /// borrows fields), so the caller can print a frame after every event.
    pub fn slot_views(&self) -> Vec<SlotView> {
        self.running.iter().map(|r| SlotView {
            id: r.job.id.clone(),
            mem_bytes: estimate_job_memory_bytes(&r.job),
            sm_blocks: estimate_job_sm_blocks(&r.job),
            step: r.step,
            max_steps: r.job.steps,
            best_acc: r.best_acc,
            last_loss: r.last_loss,
            elapsed_s: r.start.elapsed().as_secs_f64(),
        }).collect()
    }

    /// Render a Tetris-style frame: GPU bars, slot list, queue.  ANSI-clean
    /// (no escape codes) so the output is greppable; callers that want a
    /// live in-place display can prepend `\x1b[2J\x1b[H` themselves.
    pub fn render(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let mem_pct = (self.used_mem as f64 / self.mem_budget as f64 * 100.0).min(100.0);
        let sm_pct  = (self.used_sm  as f64 / self.sm_budget  as f64 * 100.0).min(100.0);
        writeln!(s, "GPU H100   mem [{:30}] {:5.1}%  ({} / {})",
                 bar(mem_pct, 30), mem_pct,
                 fmt_bytes(self.used_mem), fmt_bytes(self.mem_budget)).unwrap();
        writeln!(s, "           sm  [{:30}] {:5.1}%  ({} / {} blocks)",
                 bar(sm_pct, 30), sm_pct,
                 self.used_sm, self.sm_budget).unwrap();
        writeln!(s).unwrap();
        for (i, sv) in self.slot_views().iter().enumerate() {
            let progress = (sv.step as f64 / sv.max_steps as f64 * 100.0).min(100.0);
            writeln!(s, "  slot {}: {:>10} step {:>5}/{:<5} [{:20}] {:5.1}%   acc={:5.1}%  loss={:8.4}  mem={}  sm={}",
                     i + 1, sv.id, sv.step, sv.max_steps,
                     bar(progress, 20), progress,
                     sv.best_acc * 100.0, sv.last_loss,
                     fmt_bytes(sv.mem_bytes), sv.sm_blocks).unwrap();
        }
        for i in self.running.len()..self.max_concurrent {
            writeln!(s, "  slot {}: (free)", i + 1).unwrap();
        }
        writeln!(s).unwrap();
        if self.queue.is_empty() {
            writeln!(s, "  queue: (empty)").unwrap();
        } else {
            let preview: Vec<String> = self.queue.iter().take(8)
                .map(|j| j.id.clone()).collect();
            let more = if self.queue.len() > 8 {
                format!(" +{} more", self.queue.len() - 8)
            } else { String::new() };
            writeln!(s, "  queue: {} waiting [{}{}]", self.queue.len(),
                     preview.join(" "), more).unwrap();
        }
        s
    }

    /// Pump ONE batch on each running job, in TWO PHASES:
    ///   1. prepare:  every runner queues its forward+backward kernels onto
    ///      its own stream (no syncs). Fast — just kernel-launch overhead.
    ///      By the time we exit this loop, ALL streams are running in
    ///      parallel on the GPU.
    ///   2. finalize: every runner runs its AdamW step (which has internal
    ///      stream-syncs) + eval-at-boundary. Each runner's syncs only
    ///      block on its own stream, so other runners' kernel execution
    ///      continues during the wait — that's the whole point.
    /// Returns all events produced this step (cycle + final) in
    /// submission order.
    pub fn pump_one_step(&mut self) -> Result<Vec<SchedulerEvent>, Box<dyn Error>> {
        self.admit()?;
        let mut events = Vec::new();
        // Compact telemetry: tick at most every tick_interval_s.
        if let Some(tick) = self.maybe_emit_tick() {
            events.push(SchedulerEvent::Tick(tick));
        }
        // Phase 1 (PARALLEL): each runner launches its kernels on its own
        // OS thread.  Single-threaded launching is the actual bottleneck
        // (Entry 45 of findings.md): cuLaunchKernel through cudarc takes
        // ~1-5μs per call, and at ~50 launches × batch_size per runner per
        // step, N runners on one thread = N× wall time. With one thread
        // per runner each independently issues its own launches, and the
        // CPU work goes from N×T to T (the slowest runner).
        //
        // std::thread::scope gives us thread-spawning without the 'static
        // lifetime requirement — each scoped thread gets exclusive
        // &mut JobRunner ownership for the duration of prepare_one_batch.
        let prep_errors: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = self.running.iter_mut().map(|runner| {
                s.spawn(move || -> Result<(), String> {
                    runner.prepare_one_batch().map_err(|e| format!("{}", e))
                })
            }).collect();
            handles.into_iter()
                .filter_map(|h| match h.join() {
                    Ok(Ok(())) => None,
                    Ok(Err(s)) => Some(s),
                    Err(_) => Some("prepare thread panicked".into()),
                })
                .collect()
        });
        if let Some(first) = prep_errors.into_iter().next() {
            return Err(first.into());
        }
        // Phase 2: finalize.
        let mut i = 0;
        while i < self.running.len() {
            if let Some(ev) = self.running[i].finalize_one_batch()? {
                events.push(ev);
            }
            if self.running[i].is_done() {
                let job = &self.running[i].job;
                self.used_mem = self.used_mem.saturating_sub(estimate_job_memory_bytes(job));
                self.used_sm  = self.used_sm.saturating_sub(estimate_job_sm_blocks(job));
                let mut runner = self.running.remove(i);
                if let Some(f) = runner.take_final() {
                    events.push(SchedulerEvent::Final(f));
                }
                continue;
            }
            i += 1;
        }
        self.admit()?;
        Ok(events)
    }

    pub fn is_idle(&self) -> bool { self.running.is_empty() && self.queue.is_empty() }
}

// ---- ASCII helpers --------------------------------------------------------

fn bar(pct: f64, width: usize) -> String {
    let filled = ((pct.clamp(0.0, 100.0) / 100.0) * width as f64).round() as usize;
    let mut s = String::with_capacity(width * 3);
    for _ in 0..filled { s.push('█'); }
    for _ in filled..width { s.push('░'); }
    s
}

fn fmt_bytes(n: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * KB;
    const GB: usize = 1024 * MB;
    if n >= GB { format!("{:.1}GB", n as f64 / GB as f64) }
    else if n >= MB { format!("{:.0}MB", n as f64 / MB as f64) }
    else if n >= KB { format!("{:.0}KB", n as f64 / KB as f64) }
    else { format!("{}B", n) }
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
