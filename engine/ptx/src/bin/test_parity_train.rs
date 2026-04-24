//! Parity training, parameterized to match the findings.md winning config.
//! Defaults are the line 1867 config (d=64 L=4 dS=8) with PyTorch-style
//! curriculum learning (start easy, expand as accuracy crosses thresholds).
//!
//! Usage:
//!   cargo run --release --bin test-parity-train
//!   cargo run --release --bin test-parity-train -- --batch 256 --max-cycles 30
//!
//! All args are optional.

use mamba3_engine::model::Mamba3Model;
use ptx_engine::{PtxContext, PtxModel, PtxTrainer};
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

struct Args {
    d_model: usize,
    d_state: usize,
    headdim: usize,
    n_layers: usize,
    vocab_size: usize,
    lr: f32,
    weight_decay: f32,
    batch_size: usize,
    steps_per_cycle: usize,
    max_cycles: usize,
    target_acc: f32,
    seed: u64,
    no_curriculum: bool,
    fixed_nbits: usize,
}

impl Default for Args {
    fn default() -> Self {
        // findings.md line 1867: parity  d=64  L=4  dS=8  jit  cuda  100%
        // PyTorch specialist_trainer defaults: batch=256, lr=1e-3, wd=0.1,
        // steps_per_cycle=200, max_cycles=10 (converges in 1 cycle for this config).
        Self {
            d_model: 64,
            d_state: 8,
            headdim: 16,
            n_layers: 4,
            vocab_size: 260,
            lr: 1e-3,
            weight_decay: 0.1,
            batch_size: 256,
            steps_per_cycle: 200,
            max_cycles: 30,
            target_acc: 0.95,
            seed: 12345,
            no_curriculum: false,
            fixed_nbits: 4, // only used if --no-curriculum
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let a = &argv[i];
        let val = || argv[i + 1].clone();
        match a.as_str() {
            "--d-model" => { args.d_model = val().parse().unwrap(); i += 2; }
            "--d-state" => { args.d_state = val().parse().unwrap(); i += 2; }
            "--headdim" => { args.headdim = val().parse().unwrap(); i += 2; }
            "--layers" => { args.n_layers = val().parse().unwrap(); i += 2; }
            "--vocab" => { args.vocab_size = val().parse().unwrap(); i += 2; }
            "--lr" => { args.lr = val().parse().unwrap(); i += 2; }
            "--weight-decay" | "--wd" => { args.weight_decay = val().parse().unwrap(); i += 2; }
            "--batch" | "--batch-size" => { args.batch_size = val().parse().unwrap(); i += 2; }
            "--steps-per-cycle" => { args.steps_per_cycle = val().parse().unwrap(); i += 2; }
            "--max-cycles" => { args.max_cycles = val().parse().unwrap(); i += 2; }
            "--target-acc" => { args.target_acc = val().parse().unwrap(); i += 2; }
            "--seed" => { args.seed = val().parse().unwrap(); i += 2; }
            "--no-curriculum" => { args.no_curriculum = true; i += 1; }
            "--fixed-nbits" => { args.fixed_nbits = val().parse().unwrap(); i += 2; }
            "-h" | "--help" => { println!("see source for flags"); std::process::exit(0); }
            _ => { eprintln!("unknown arg: {}", a); std::process::exit(2); }
        }
    }
    args
}

#[derive(Clone, Copy)]
struct Stage { min_len: usize, max_len: usize, advance_at: f32 }

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args();
    println!("=== PTX Parity Training ===");
    println!("Config: d={} L={} dS={} hd={} batch={} lr={} wd={} cycles={}x{} steps",
        args.d_model, args.n_layers, args.d_state, args.headdim,
        args.batch_size, args.lr, args.weight_decay,
        args.max_cycles, args.steps_per_cycle);
    let t0 = Instant::now();
    let ptx = Arc::new(PtxContext::new()?);
    println!("kernels compiled in {:.2}s", t0.elapsed().as_secs_f64());

    let cpu_model = Mamba3Model::new_random(
        args.d_model, args.d_state, args.headdim, args.n_layers, args.vocab_size,
    );
    println!("Model: {} params", cpu_model.param_count());

    // Max sequence length for scratch buffer sizing: max_bits (16) + BOS + SEP + answer + EOS
    let max_seq = 21;
    let gpu_model = PtxModel::from_cpu(&cpu_model, ptx.clone(), max_seq)?;
    let mut trainer = PtxTrainer::new(gpu_model, args.lr, args.weight_decay, max_seq)?;

    // Curriculum matches problems/parity/problem.yaml:
    // stage 1: min_len 2, max_len 4, advance at 90%
    // stage 2: min_len 3, max_len 8, advance at 90%
    // stage 3: min_len 4, max_len 16, advance at 95%
    let stages: Vec<Stage> = if args.no_curriculum {
        vec![Stage { min_len: args.fixed_nbits, max_len: args.fixed_nbits, advance_at: 2.0 }]
    } else {
        vec![
            Stage { min_len: 2, max_len: 4,  advance_at: 0.90 },
            Stage { min_len: 3, max_len: 8,  advance_at: 0.90 },
            Stage { min_len: 4, max_len: 16, advance_at: 0.95 },
        ]
    };
    let mut stage_idx = 0;

    let mut rng_state: u64 = args.seed;
    let mut next_bit = |s: &mut u64| -> u32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*s >> 33) & 1) as u32
    };
    let mut rand_range = |s: &mut u64, lo: usize, hi: usize| -> usize {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = ((*s >> 33) as u32) as usize;
        lo + r % (hi - lo + 1)
    };

    let start = Instant::now();
    let mut best_acc = 0.0f32;
    let mut total_train_steps = 0usize;

    for cycle in 0..args.max_cycles {
        let stage = stages[stage_idx];
        let mut total_loss = 0.0f32;
        let mut loss_count = 0usize;

        for _step in 0..args.steps_per_cycle {
            let mut batch_loss = 0.0f32;
            for _ in 0..args.batch_size {
                let n_bits = rand_range(&mut rng_state, stage.min_len, stage.max_len);
                let mut bits = Vec::with_capacity(n_bits);
                let mut parity = 0u32;
                for _ in 0..n_bits {
                    let b = next_bit(&mut rng_state);
                    bits.push(b);
                    parity ^= b;
                }
                let mut tokens: Vec<u32> = vec![256];
                for (i, &b) in bits.iter().enumerate() {
                    if i > 0 { tokens.push(32); } // space (matches generator)
                    tokens.push(48 + b);
                }
                tokens.push(258);
                let answer = if parity == 0 { 83 } else { 68 };
                tokens.push(answer);
                tokens.push(257);

                let mut targets = tokens[1..].to_vec();
                targets.push(257);

                batch_loss += trainer.train_step(&tokens, &targets)?;
                total_train_steps += 1;
            }
            total_loss += batch_loss / args.batch_size as f32;
            loss_count += 1;
        }
        let avg_loss = total_loss / loss_count as f32;

        // Eval: sample 200 random n_bits ∈ [stage.min_len, stage.max_len]
        let mut correct = 0usize;
        let n_eval = 200;
        for _ in 0..n_eval {
            let n_bits = rand_range(&mut rng_state, stage.min_len, stage.max_len);
            let mut test_bits = Vec::with_capacity(n_bits);
            let mut test_parity = 0u32;
            for _ in 0..n_bits {
                let b = next_bit(&mut rng_state);
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
        let acc = correct as f32 / n_eval as f32;
        best_acc = best_acc.max(acc);
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "  [parity] cycle {:>2}  stage={}(len {}-{})  loss={:.4}  acc={:.0}%  best={:.0}%  ({:.1}s)",
            cycle + 1, stage_idx + 1, stage.min_len, stage.max_len,
            avg_loss, acc * 100.0, best_acc * 100.0, elapsed,
        );

        // Advance curriculum on threshold
        if acc >= stage.advance_at && stage_idx + 1 < stages.len() {
            stage_idx += 1;
            let ns = stages[stage_idx];
            println!("  ★ advanced to stage {} (len {}-{}, advance at {:.0}%)",
                stage_idx + 1, ns.min_len, ns.max_len, ns.advance_at * 100.0);
        }

        // Stop if target reached on final stage
        if stage_idx + 1 == stages.len() && acc >= args.target_acc {
            println!("  ★ target {:.0}% reached — stopping", args.target_acc * 100.0);
            break;
        }
    }

    let total_time = start.elapsed().as_secs_f64() * 1000.0;
    let ms_per_step = total_time / total_train_steps as f64;
    println!(
        "\nFinal: best_acc={:.0}%  {:.2}ms/step  {:.0} steps/sec  ({} train steps total)",
        best_acc * 100.0, ms_per_step, 1000.0 / ms_per_step, total_train_steps,
    );
    let status = if best_acc >= args.target_acc { "PASS" }
        else if best_acc > 0.7 { "LEARNING" }
        else { "NEEDS TUNING" };
    println!("Status: {}", status);
    Ok(())
}
