//! ptxd — JSON job daemon for the PTX Mamba-3 training engine.
//!
//! Reads JSON jobs from stdin (one object per line) and runs them through
//! the slot Scheduler, streaming results to stdout. Supports concurrent
//! jobs via `--concurrent N` (default 1, backward compatible with the
//! sequential ptxd v1 contract).
//!
//! ## JSON contract
//!
//! Input (one per line):
//! ```json
//! {"id":"j1","task":"parity","n_bits":3,"d_model":32,"d_state":16,
//!  "headdim":16,"n_layers":1,"vocab_size":260,
//!  "lr":1e-3,"weight_decay":0.1,"batch_size":16,"steps":5000,
//!  "target_acc":0.95,"seed":12345,
//!  "stages":[{"min_len":2,"max_len":4,"advance_at":0.9}, ...]}
//! ```
//!
//! Output (multiple per job):
//! ```jsonl
//! {"type":"cycle", "id":"j1", "cycle":1, "step":200, "loss":..., ...}
//! ...
//! {"type":"final", "id":"j1", "status":"converged", "best_acc":..., ...}
//! ```
//!
//! ## Usage
//!
//! ```sh
//! # Sequential (one job at a time):
//! cat jobs.jsonl | ./ptxd
//!
//! # Up to 4 concurrent jobs co-executing on different streams:
//! cat jobs.jsonl | ./ptxd --concurrent 4
//! ```

use ptx_engine::{Job, PtxContext, Scheduler, SchedulerEvent};
use std::error::Error;
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::time::Instant;

fn parse_args() -> usize {
    let mut concurrent = 1usize;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--concurrent" | "-j" => {
                concurrent = argv.get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1);
                i += 2;
            }
            "-h" | "--help" => {
                println!("ptxd — JSON job daemon for PTX Mamba-3 training\n");
                println!("Usage: ptxd [--concurrent N] < jobs.jsonl");
                println!("  --concurrent N   max number of jobs running in parallel (default 1)");
                std::process::exit(0);
            }
            other => {
                eprintln!("[ptxd] unknown arg: {}", other);
                std::process::exit(2);
            }
        }
    }
    concurrent
}

fn main() -> Result<(), Box<dyn Error>> {
    let max_concurrent = parse_args();
    eprintln!("[ptxd] compiling PTX kernels...");
    let t0 = Instant::now();
    let ctx = Arc::new(PtxContext::new()?);
    eprintln!(
        "[ptxd] ready in {:.2}s, max_concurrent={}, awaiting jobs on stdin (one JSON per line)",
        t0.elapsed().as_secs_f64(), max_concurrent,
    );

    let mut sched = Scheduler::new(ctx, max_concurrent);
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    // Drain stdin first (collect all jobs), then run the scheduler.  This is
    // simpler than interleaving stdin reads with pump_one_step and matches
    // how three_populations.py uses the daemon (one batch of jobs, wait for
    // all to finish, exit). Streaming-stdin support is a future upgrade.
    let mut n_submitted = 0usize;
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
                    "type": "parse_error", "error": e.to_string()
                });
                writeln!(out, "{}", err)?;
                out.flush()?;
                continue;
            }
        };
        eprintln!(
            "[ptxd] submitted {} (task={}, d={}, L={}, {} steps)",
            job.id, job.task, job.d_model, job.n_layers, job.steps,
        );
        sched.submit(job);
        n_submitted += 1;
    }
    eprintln!("[ptxd] {} jobs submitted, running...", n_submitted);

    while !sched.is_idle() {
        let events = sched.pump_one_step()?;
        for ev in events {
            // Each event is either a cycle row (every 200 steps) or a final
            // row (once per job).  Match the existing JSON contract that
            // ptxd_specialist.py / smoke tests expect.
            let line = serde_json::to_string(&ev)?;
            writeln!(out, "{}", line)?;
            out.flush()?;
        }
    }
    eprintln!("[ptxd] all jobs complete in {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
