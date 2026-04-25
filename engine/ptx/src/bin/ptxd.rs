//! ptxd — JSON job daemon for the PTX Mamba-3 training engine.
//!
//! Reads JSON jobs from stdin (one object per line) and runs them through
//! the slot Scheduler, streaming results to stdout. Supports concurrent
//! jobs via `--concurrent N` (default 1, backward compatible with the
//! sequential ptxd v1 contract).
//!
//! ## Hot-plug daemon mode (default)
//!
//! ptxd runs forever as a daemon: a stdin-reader thread pushes new jobs
//! into a channel, the main loop pumps the scheduler AND drains the channel
//! every iteration. New jobs can be submitted at any time, even while
//! existing jobs are running. Exit is on stdin EOF *and* scheduler idle —
//! which matches the legacy "pipe one job, close stdin, wait for final"
//! pattern AND supports a long-lived three_populations.py keeping ptxd
//! alive across many specialist runs.
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
//!
//! # Long-lived daemon: keep stdin open, ptxd accepts jobs as they arrive.
//! # Exits when stdin closes AND all jobs have finished.
//! mkfifo /tmp/ptxd.in
//! ./ptxd < /tmp/ptxd.in &
//! echo '{"id":"j1",...}' > /tmp/ptxd.in
//! # ... later, after all jobs done:
//! exec 9>/tmp/ptxd.in; exec 9>&-   # close fifo → ptxd exits when idle
//! ```

use ptx_engine::{Job, PtxContext, Scheduler};
use std::error::Error;
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

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

/// Messages from the stdin-reader thread to the main loop.
enum StdinMsg {
    Job(Job),
    /// JSON couldn't parse. Forward as parse_error to stdout, keep going.
    ParseError(String),
}

fn main() -> Result<(), Box<dyn Error>> {
    let max_concurrent = parse_args();
    eprintln!("[ptxd] compiling PTX kernels...");
    let t0 = Instant::now();
    let ctx = Arc::new(PtxContext::new()?);
    eprintln!(
        "[ptxd] ready in {:.2}s, max_concurrent={}, hot-plug daemon awaiting jobs on stdin",
        t0.elapsed().as_secs_f64(), max_concurrent,
    );

    let mut sched = Scheduler::new(ctx, max_concurrent);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    // Spawn a thread to drain stdin into a channel. The thread exits when
    // stdin closes (EOF), which drops the Sender and signals the main loop
    // that no more jobs are coming. The main loop only exits when both
    //   (a) the Sender has been dropped (stdin closed), AND
    //   (b) the scheduler is idle (all running jobs have finalized)
    // simultaneously, so we never drop a job mid-run.
    let (tx, rx) = mpsc::channel::<StdinMsg>();
    let stdin_thread = thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => { eprintln!("[ptxd] stdin error: {}", e); break; }
            };
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }
            match serde_json::from_str::<Job>(trimmed) {
                Ok(job) => {
                    if tx.send(StdinMsg::Job(job)).is_err() { break; }
                }
                Err(e) => {
                    if tx.send(StdinMsg::ParseError(e.to_string())).is_err() { break; }
                }
            }
        }
        // Sender dropped on thread exit → main loop's try_recv returns Disconnected.
    });

    let mut n_submitted = 0usize;
    let mut stdin_closed = false;
    loop {
        // 1. Drain the stdin channel — any number of new jobs may have arrived
        //    since the last iteration. Non-blocking; if there's nothing, we
        //    just fall through to the scheduler pump.
        loop {
            match rx.try_recv() {
                Ok(StdinMsg::Job(job)) => {
                    eprintln!(
                        "[ptxd] submitted {} (task={}, d={}, L={}, {} steps)",
                        job.id, job.task, job.d_model, job.n_layers, job.steps,
                    );
                    sched.submit(job);
                    n_submitted += 1;
                }
                Ok(StdinMsg::ParseError(e)) => {
                    let err = serde_json::json!({
                        "type": "parse_error", "error": e
                    });
                    writeln!(out, "{}", err)?;
                    out.flush()?;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if !stdin_closed {
                        stdin_closed = true;
                        eprintln!("[ptxd] stdin closed; will exit when {} running job(s) finish",
                                  if sched.is_idle() { 0 } else { 1 });
                    }
                    break;
                }
            }
        }

        // 2. Pump the scheduler one step. Returns any cycle/final/tick events
        //    that fired; we forward them to stdout.
        let events = sched.pump_one_step()?;
        for ev in events {
            let line = serde_json::to_string(&ev)?;
            writeln!(out, "{}", line)?;
            out.flush()?;
        }

        // 3. Termination: stdin EOF AND no jobs remain anywhere.
        if stdin_closed && sched.is_idle() {
            break;
        }

        // 4. Idle wait. When there's nothing running and stdin is still open,
        //    don't busy-poll — sleep briefly so we don't burn a core. The
        //    pump above is non-blocking so this only runs in the truly-idle
        //    case (long-lived daemon waiting for new jobs).
        if sched.is_idle() && !stdin_closed {
            thread::sleep(Duration::from_millis(20));
        }
    }

    let _ = stdin_thread.join();
    eprintln!(
        "[ptxd] {} jobs run; daemon exiting after {:.1}s",
        n_submitted, t0.elapsed().as_secs_f64(),
    );
    Ok(())
}
