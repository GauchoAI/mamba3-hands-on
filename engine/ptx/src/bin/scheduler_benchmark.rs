//! scheduler-benchmark — proves the slot scheduler doesn't introduce
//! interference between concurrent jobs.
//!
//! Test: run an identical job (same config, same seed) under three regimes:
//!
//!   1. ALONE   — one job, max_concurrent=1, scheduler empty otherwise
//!   2. PAIRED  — two copies of the job, max_concurrent=2
//!   3. PACKED  — N copies, max_concurrent=N
//!
//! For each regime we measure each job's *active* wall time (from when it
//! got admitted out of the queue until its FinalEvent). If the scheduler
//! has zero interference, ALONE and PACKED active times should be equal —
//! the only difference between them is how many copies are running in
//! parallel, which is exactly what the GPU is supposed to handle for free
//! when SMs aren't oversubscribed.
//!
//! Reports the ratio (median packed / alone). 1.00× means perfect
//! isolation; >1.00× means jobs are slowing each other down.

use ptx_engine::{Job, PtxContext, Scheduler, SchedulerEvent};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

fn job_template(id: &str, seed: u64) -> Job {
    Job {
        id: id.to_string(),
        task: "parity".to_string(),
        d_model: 32, d_state: 16, headdim: 16, n_layers: 2,
        vocab_size: 260, lr: 1e-3, weight_decay: 0.1,
        steps: 600,            // small fixed budget so all regimes run the same work
        batch_size: 16,
        n_bits: 3, target_acc: 0.99,   // unreachable so all jobs run the full budget
        seed,
        stages: None,
        init_from_bin: None, save_bin: None,
        batches_path: None, eval_batches_path: None,
        loss: Default::default(), optimizer: Default::default(), schedule: Default::default(),
        optimizer_state_in: None, optimizer_state_out: None,
    }
}

/// Runs `n_jobs` copies under `max_concurrent`, returns each job's wall_ms
/// (active time, from admit→final).
fn run_regime(label: &str, ctx: Arc<PtxContext>, n_jobs: usize, max_concurrent: usize)
    -> Result<Vec<f64>, Box<dyn Error>>
{
    let mut sched = Scheduler::new(ctx, max_concurrent);
    for i in 0..n_jobs {
        sched.submit(job_template(&format!("{}_{}", label, i), 12345 + i as u64));
    }

    let t_total = Instant::now();
    // Track each job's admission time so we can subtract queue wait from
    // its FinalEvent's wall_ms (which counts from JobRunner::new — that's
    // already start-of-active for that runner).
    let mut active_ms: Vec<f64> = Vec::new();

    while !sched.is_idle() {
        let events = sched.pump_one_step()?;
        for ev in events {
            if let SchedulerEvent::Final(f) = ev {
                // FinalEvent.wall_ms is computed from JobRunner.start, which
                // is set when the runner is constructed (i.e. at admission)
                // — exactly the active-time we want.
                active_ms.push(f.wall_ms);
            }
        }
    }
    let total_s = t_total.elapsed().as_secs_f64();
    eprintln!(
        "  [{}]  n={} concurrent={}  total_wall={:6.2}s  active_med={:6.2}s",
        label, n_jobs, max_concurrent, total_s,
        median(&active_ms) / 1000.0,
    );
    Ok(active_ms)
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    let mut v: Vec<f64> = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let ctx = Arc::new(PtxContext::new()?);
    eprintln!("[bench] scheduler interference benchmark");
    eprintln!("[bench] each job: parity d=32 L=2 dS=16 batch=16 lr=1e-3 600 steps");
    eprintln!();

    eprintln!("Running regimes:");
    let alone = run_regime("alone",   ctx.clone(), 1, 1)?;
    let pair  = run_regime("pair",    ctx.clone(), 2, 2)?;
    let quad  = run_regime("quad",    ctx.clone(), 4, 4)?;

    // Bigger queue than concurrency to also measure queue-wait recovery.
    let oversubscribed = run_regime("oversub", ctx.clone(), 8, 4)?;

    let alone_ms = median(&alone);
    eprintln!("\n=== Per-job active time (median, ms) ===");
    eprintln!("{:<14} {:>10}  ratio_vs_alone", "regime", "median_ms");
    eprintln!("{}", "-".repeat(45));
    let print = |label: &str, xs: &[f64]| {
        let m = median(xs);
        let ratio = if alone_ms > 0.0 { m / alone_ms } else { 0.0 };
        eprintln!("{:<14} {:>10.1}  {:>5.2}x", label, m, ratio);
    };
    print("alone (n=1)", &alone);
    print("pair  (n=2)", &pair);
    print("quad  (n=4)", &quad);
    print("oversub (8/4)", &oversubscribed);

    eprintln!("\nInterpretation:");
    eprintln!("  ratio == 1.00x  → no interference; concurrent jobs are free");
    eprintln!("  ratio > 1.00x   → jobs slowing each other down (SM contention)");
    eprintln!("  ratio < 1.00x   → measurement noise (cycle alignment)");

    // Compact JSON summary line — easy for an external monitor to scrape.
    println!("{}", serde_json::json!({
        "type": "scheduler_benchmark",
        "alone_ms": alone_ms,
        "pair_ms":  median(&pair),
        "quad_ms":  median(&quad),
        "oversub_ms": median(&oversubscribed),
        "pair_ratio":  median(&pair)  / alone_ms,
        "quad_ratio":  median(&quad)  / alone_ms,
        "oversub_ratio": median(&oversubscribed) / alone_ms,
        "n_alone": alone.len(), "n_pair": pair.len(),
        "n_quad": quad.len(),  "n_oversub": oversubscribed.len(),
    }));
    Ok(())
}
