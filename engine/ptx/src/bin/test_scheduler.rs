//! Smoke test for the Scheduler abstraction.
//!
//! Submits 2 small parity jobs with max_concurrent=2, pumps the scheduler
//! until both complete, prints all events. Validates that two JobRunners
//! co-exist in one process on different streams.

use ptx_engine::{Job, PtxContext, Scheduler, SchedulerEvent};
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let ctx = Arc::new(PtxContext::new()?);
    eprintln!("[test-scheduler] PtxContext ready");

    let mut sched = Scheduler::new(ctx, /* max_concurrent = */ 2);

    // Two simple parity jobs (different seeds so they have different RNG paths).
    sched.submit(Job {
        id: "alpha".into(),
        task: "parity".into(),
        d_model: 32, d_state: 16, headdim: 16, n_layers: 2, vocab_size: 260,
        lr: 1e-3, weight_decay: 0.1,
        steps: 1500, batch_size: 16,
        n_bits: 3, target_acc: 0.95,
        seed: 7,
        stages: None,
        init_from_bin: None, save_bin: None,
        batches_path: None, eval_batches_path: None,
        loss: Default::default(), optimizer: Default::default(), schedule: Default::default(),
        optimizer_state_in: None, optimizer_state_out: None,
    });
    sched.submit(Job {
        id: "beta".into(),
        task: "parity".into(),
        d_model: 32, d_state: 16, headdim: 16, n_layers: 2, vocab_size: 260,
        lr: 1e-3, weight_decay: 0.1,
        steps: 1500, batch_size: 16,
        n_bits: 3, target_acc: 0.95,
        seed: 12345,
        stages: None,
        init_from_bin: None, save_bin: None,
        batches_path: None, eval_batches_path: None,
        loss: Default::default(), optimizer: Default::default(), schedule: Default::default(),
        optimizer_state_in: None, optimizer_state_out: None,
    });

    let t0 = Instant::now();
    let mut total_events = 0;
    while !sched.is_idle() {
        let events = sched.pump_one_step()?;
        for ev in events {
            total_events += 1;
            match ev {
                SchedulerEvent::Cycle(c) => println!(
                    "[cycle] id={} cycle={} step={} loss={:.4} acc={:.0}% best={:.0}% stage={} elapsed={:.1}s",
                    c.id, c.cycle, c.step, c.loss, c.fresh_acc * 100.0,
                    c.best_fresh * 100.0, c.stage, c.elapsed_s,
                ),
                SchedulerEvent::Final(f) => println!(
                    "[final] id={} status={} best_acc={:.0}% loss={:.4} ms/step={:.2} wall={:.1}s steps={}",
                    f.id, f.status, f.best_acc * 100.0, f.final_loss,
                    f.ms_per_step, f.wall_ms / 1000.0, f.steps_executed,
                ),
                SchedulerEvent::Tick(_) => {} // ignore
            }
        }
    }
    println!(
        "\n[test-scheduler] all jobs done. Wall {:.1}s, {} events.",
        t0.elapsed().as_secs_f64(), total_events,
    );
    Ok(())
}
