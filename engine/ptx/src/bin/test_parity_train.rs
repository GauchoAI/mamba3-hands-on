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
        // Defaults: the config that currently trains STABLY on the PTX
        // backward we have. Not the PyTorch-winning config — that one needs
        // the remaining gradient closures (bp/cp LN-bwd, RoPE-bwd,
        // d_dt_bias, d_scale, correct bx-timestep coupling) to converge.
        //
        // To target the PyTorch winning config once those land, override:
        //   --d-model 64 --layers 4 --d-state 8 --headdim 16
        //   --batch 256 --steps-per-cycle 200 --max-cycles 10
        //   --target-acc 0.95
        //   (curriculum is already on by default)
        //
        // Reference: findings.md Entry 24 line 1867 — this config hits 100%
        // parity in one cycle via `specialist_trainer.py --scan-backend jit
        // --device cuda`. Our PTX reaches ~58% stably today; the gap is
        // gradient-coverage, not precision.
        Self {
            d_model: 32,
            d_state: 16,
            headdim: 16,
            n_layers: 1,
            vocab_size: 260,
            lr: 1e-3,
            weight_decay: 0.1,
            batch_size: 16,
            steps_per_cycle: 200,
            max_cycles: 25,  // enough for curriculum to try all stages
            target_acc: 0.95,
            seed: 12345,
            no_curriculum: false,
            fixed_nbits: 4,
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

    let mut cpu_model = Mamba3Model::new_random(
        args.d_model, args.d_state, args.headdim, args.n_layers, args.vocab_size,
    );

    // --- Match PyTorch (mamba3_minimal.py) init ----------------------------
    // The CPU `new_random` uses Xavier-uniform everywhere and dt_bias=-3.0
    // uniformly across heads. PyTorch uses two important recipe-specific inits:
    //   1. dt_bias log-uniform per head: each head gets a different timescale
    //      sampled in [dt_min=0.001, dt_max=0.1]. dt_bias = inv_softplus(dt) =
    //      dt + log(-expm1(-dt)).  This is the classic Mamba init and is the
    //      most likely fix for the variable-length parity plateau.
    //   2. embed_w ~ N(0, 1) (PyTorch nn.Embedding default), much larger than
    //      our Xavier-uniform default (~7× difference for V=260, d=64).
    {
        // LCG matching the seed style elsewhere in this binary.
        let mut s: u64 = args.seed;
        let mut lcg = || -> f32 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as u32) as f32 / u32::MAX as f32
        };
        let dt_min: f32 = 0.001;
        let dt_max: f32 = 0.1;
        let log_dt_min = dt_min.ln();
        let log_dt_max = dt_max.ln();
        for layer in cpu_model.layers.iter_mut() {
            for h in 0..layer.dt_bias.len() {
                let r = lcg();                               // U[0,1)
                let dt_h = (r * (log_dt_max - log_dt_min) + log_dt_min).exp().max(1e-4);
                // inv_softplus: bias such that softplus(bias) = dt_h.
                layer.dt_bias[h] = dt_h + (-(-dt_h).exp_m1()).ln();
            }
        }
        // Box–Muller standard normal for embedding to match PyTorch nn.Embedding.
        let mut next_normal = || -> f32 {
            let u1 = lcg().max(1e-30);
            let u2 = lcg();
            ((-2.0 * u1.ln()).sqrt()) * (2.0 * std::f32::consts::PI * u2).cos()
        };
        for w in cpu_model.embed_w.iter_mut() {
            *w = next_normal();
        }

        // Layer scale: PyTorch inits at 0.01 but with autograd it can move
        // freely. With our discrete steps and small scalar grads, scale=0.01
        // is too small for the optimizer to grab onto and gets driven to 0.
        // Bump initial scale to 0.1 so the SSM signal is detectable from
        // step 1 and the gradient has direction.
        for layer in cpu_model.layers.iter_mut() {
            layer.scale = 0.1;
        }
    }
    // After mutating cpu_model, push the new weights to GPU. (`from_cpu` below
    // uploads the post-mutation values.)
    println!("Model: {} params  (dt_bias log-uniform; embed N(0,1))",
        cpu_model.param_count());

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
            // Mini-batch accumulation: zero grads once, accumulate across
            // batch_size samples, then a single AdamW step scaled by 1/B.
            // This matches PyTorch's batched-backward semantics and avoids
            // the per-sample-AdamW thrashing that destabilises mixed-length
            // training. (For fixed-length parity the per-sample path also
            // works; for variable length, accumulation is the difference
            // between bouncing-loss and convergence.)
            trainer.zero_gradients_only()?;
            // accumulate_gradients returns the running CUMULATIVE loss in the
            // GPU accumulator (it's reset by zero_gradients_only via do_zero
            // in the first call's path). The last returned value IS the sum.
            let mut last_loss = 0.0f32;
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

                last_loss = trainer.accumulate_gradients(&tokens, &targets)?;
                total_train_steps += 1;
            }
            // Single AdamW step for the whole mini-batch. extra_g_mul = 1/B
            // averages the accumulated sum, matching PyTorch batched semantics.
            trainer.apply_optimizer_step_scaled(1.0 / args.batch_size as f32)?;
            // last_loss is the cumulative sum of n_active_per_sample-normalised
            // losses across the batch. Divide by batch size for mean.
            total_loss += last_loss / args.batch_size as f32;
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
        // Probe learnable scale per layer — confirms d_scale is flowing.
        let scales: Vec<String> = trainer.model.layers.iter()
            .map(|l| format!("{:.3}", l.scale)).collect();
        println!(
            "  [parity] cycle {:>2}  stage={}(len {}-{})  loss={:.4}  acc={:.0}%  best={:.0}%  scales=[{}]  ({:.1}s)",
            cycle + 1, stage_idx + 1, stage.min_len, stage.max_len,
            avg_loss, acc * 100.0, best_acc * 100.0, scales.join(","), elapsed,
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
