//! tetris-demo — renders the slot scheduler as it packs jobs onto the GPU.
//!
//! Submits a queue of parity jobs of varying sizes, runs the scheduler with
//! a configurable concurrency cap, and prints an ASCII "Tetris view" of GPU
//! resource usage after every event.  Use `clear` ANSI codes for a live
//! refresh, or pipe to `cat` for a scrollable history.
//!
//! ```sh
//! ./target/release/tetris-demo                # default: 8 jobs, max-concurrent=4
//! ./target/release/tetris-demo --concurrent 8 # try to pack 8 jobs at once
//! ./target/release/tetris-demo --no-clear     # no ANSI clear, scroll history
//! ```

use ptx_engine::{Job, PtxContext, Scheduler, SchedulerEvent};
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

fn parse_args() -> (usize, bool) {
    let mut concurrent = 4usize;
    let mut clear = true;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--concurrent" | "-j" => {
                concurrent = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(4);
                i += 2;
            }
            "--no-clear" => { clear = false; i += 1; }
            "-h" | "--help" => {
                println!("tetris-demo — visualize the slot scheduler packing jobs");
                println!("  --concurrent N   max running jobs (default 4)");
                println!("  --no-clear       don't ANSI-clear between frames");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {}", other); std::process::exit(2); }
        }
    }
    (concurrent, clear)
}

/// Build a heterogeneous queue: some small fast jobs, some bigger slower
/// ones, so the visualization shows different shapes of slots.
fn build_queue() -> Vec<Job> {
    let mut jobs = Vec::new();
    // 8 jobs total — mix of model sizes and step budgets.
    let configs = [
        ("alpha",   32, 16,  1, 7,    1500),
        ("bravo",   32, 16,  2, 12345, 2000),
        ("charlie", 32, 16,  2, 42,   1500),
        ("delta",   64,  8,  4, 7,    1500),
        ("echo",    32, 16,  1, 999,  1500),
        ("foxtrot", 64,  8,  2, 100,  1500),
        ("golf",    32, 16,  2, 31337, 2000),
        ("hotel",   32, 16,  1, 2024, 1500),
    ];
    for &(name, d_model, d_state, n_layers, seed, steps) in &configs {
        jobs.push(Job {
            id: name.to_string(),
            task: "parity".to_string(),
            d_model, d_state, headdim: 16, n_layers,
            vocab_size: 260,
            lr: 1e-3, weight_decay: 0.1,
            steps, batch_size: 16,
            n_bits: 3, target_acc: 0.95,
            seed,
            stages: None,
        });
    }
    jobs
}

fn render_frame(sched: &Scheduler, t0: Instant, last_event: &str, clear: bool) {
    if clear {
        // Clear screen + home cursor (ANSI). Use --no-clear to disable.
        print!("\x1b[2J\x1b[H");
    }
    println!("=== ptxd tetris view  ({:>5.1}s elapsed, last: {}) ===",
        t0.elapsed().as_secs_f64(), last_event);
    print!("{}", sched.render());
    println!();
}

fn main() -> Result<(), Box<dyn Error>> {
    let (concurrent, clear) = parse_args();
    let ctx = Arc::new(PtxContext::new()?);
    eprintln!("[tetris-demo] PtxContext ready, max_concurrent={}", concurrent);

    let mut sched = Scheduler::new(ctx, concurrent);
    for job in build_queue() {
        sched.submit(job);
    }
    let t0 = Instant::now();
    let mut last_event = String::from("(starting)");
    render_frame(&sched, t0, &last_event, clear);

    while !sched.is_idle() {
        let events = sched.pump_one_step()?;
        for ev in &events {
            last_event = match ev {
                SchedulerEvent::Cycle(c) => format!(
                    "{} c{} step={} acc={:.0}%",
                    c.id, c.cycle, c.step, c.fresh_acc * 100.0,
                ),
                SchedulerEvent::Final(f) => format!(
                    "{} done ({}, best={:.0}%, {:.1}s)",
                    f.id, f.status, f.best_acc * 100.0, f.wall_ms / 1000.0,
                ),
                SchedulerEvent::Tick(t) => format!(
                    "tick t={:.1}s mem={:.0}% sm={:.0}% running={} queue={}",
                    t.t, t.mem_pct, t.sm_pct, t.running, t.queue,
                ),
            };
        }
        // Render after each pump (whether or not events were emitted).
        if !events.is_empty() {
            render_frame(&sched, t0, &last_event, clear);
        }
    }
    if clear { print!("\x1b[2J\x1b[H"); }
    println!("=== ptxd tetris demo done in {:.1}s ===", t0.elapsed().as_secs_f64());
    print!("{}", sched.render());
    Ok(())
}
