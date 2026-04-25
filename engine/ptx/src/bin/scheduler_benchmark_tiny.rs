//! Like scheduler-benchmark, but with a TINY model (d=8 L=1 dS=8) that
//! doesn't saturate the GPU. If the slot scheduler genuinely co-executes
//! across streams (it should, since we have prepare/finalize and the GPU
//! has headroom), the ratio_vs_alone for pair/quad will drop below 1.5×.
//! If we still see N×, the bottleneck is host-side launch serialization
//! and we need multi-threaded launches.

use ptx_engine::{Job, PtxContext, Scheduler, SchedulerEvent};
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

fn job_template(id: &str, seed: u64) -> Job {
    Job {
        id: id.to_string(),
        task: "parity".to_string(),
        d_model: 8, d_state: 8, headdim: 8, n_layers: 1,
        vocab_size: 260, lr: 1e-3, weight_decay: 0.1,
        steps: 600,
        batch_size: 16,
        n_bits: 3, target_acc: 0.99,   // unreachable so all run full budget
        seed,
        stages: None,
    }
}

fn run_regime(label: &str, ctx: Arc<PtxContext>, n_jobs: usize, max_concurrent: usize)
    -> Result<Vec<f64>, Box<dyn Error>>
{
    let mut sched = Scheduler::new(ctx, max_concurrent);
    for i in 0..n_jobs {
        sched.submit(job_template(&format!("{}_{}", label, i), 12345 + i as u64));
    }
    let t_total = Instant::now();
    let mut active_ms: Vec<f64> = Vec::new();
    while !sched.is_idle() {
        let events = sched.pump_one_step()?;
        for ev in events {
            if let SchedulerEvent::Final(f) = ev {
                active_ms.push(f.wall_ms);
            }
        }
    }
    let total_s = t_total.elapsed().as_secs_f64();
    eprintln!(
        "  [{}]  n={} concurrent={}  total_wall={:6.2}s  active_med={:6.2}s",
        label, n_jobs, max_concurrent, total_s, median(&active_ms) / 1000.0,
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
    eprintln!("[bench-tiny] tiny model d=8 L=1 dS=8 — well below GPU saturation");
    eprintln!();
    let alone = run_regime("alone", ctx.clone(), 1, 1)?;
    let pair  = run_regime("pair",  ctx.clone(), 2, 2)?;
    let quad  = run_regime("quad",  ctx.clone(), 4, 4)?;
    let octa  = run_regime("octa",  ctx.clone(), 8, 8)?;

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
    print("octa  (n=8)", &octa);

    println!("{}", serde_json::json!({
        "type": "scheduler_benchmark_tiny",
        "model": "d=8 L=1 dS=8",
        "alone_ms": alone_ms,
        "pair_ms":  median(&pair),  "pair_ratio":  median(&pair)  / alone_ms,
        "quad_ms":  median(&quad),  "quad_ratio":  median(&quad)  / alone_ms,
        "octa_ms":  median(&octa),  "octa_ratio":  median(&octa)  / alone_ms,
    }));
    Ok(())
}
