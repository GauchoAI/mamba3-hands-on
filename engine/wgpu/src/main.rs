//! Mamba-3 Engine — benchmark + model inference test.
//!
//! Usage:
//!   cargo run --release                          # SSM scan benchmark
//!   cargo run --release -- --model /tmp/model.bin # Full model inference

use mamba3_engine::scan::GpuContext;
use mamba3_engine::model::Mamba3Model;
use std::time::Instant;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if let Some(pos) = args.iter().position(|a| a == "--model") {
        let model_path = &args[pos + 1];
        if args.iter().any(|a| a == "--train") {
            run_training_bench(model_path);
        } else {
            run_model_inference(model_path);
        }
    } else {
        run_scan_benchmark();
    }
}

fn run_model_inference(model_path: &str) {
    println!("Loading model from {}...", model_path);
    let model = Mamba3Model::from_bin(Path::new(model_path))
        .expect("Failed to load model");

    println!("Model: d={}, L={}, dS={}, hd={}, vocab={}",
        model.d_model, model.n_layers, model.d_state, model.headdim, model.vocab_size);
    println!("  d_inner={}, n_heads={}", model.d_inner, model.n_heads);

    // Test tokens: BOS(256) + some bytes + SEP(258)
    let tokens: Vec<u32> = vec![256, 48, 32, 49, 32, 48, 258]; // "0 1 0" for parity
    println!("\nInput tokens: {:?}", tokens);

    let start = Instant::now();
    let logits = model.forward(&tokens);
    let elapsed = start.elapsed();

    let l = tokens.len();
    let v = model.vocab_size;

    // Show predictions at each position
    println!("\nPredictions (top-3 per position):");
    for t in 0..l {
        let mut indexed: Vec<(usize, f32)> = (0..v)
            .map(|i| (i, logits[t * v + i]))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top3: Vec<String> = indexed.iter().take(3)
            .map(|(idx, val)| {
                let ch = if *idx < 256 {
                    format!("'{}'", *idx as u8 as char)
                } else {
                    format!("special({})", idx)
                };
                format!("{}: {:.3}", ch, val)
            })
            .collect();

        let input_ch = if tokens[t] < 256 {
            format!("'{}'", tokens[t] as u8 as char)
        } else {
            format!("special({})", tokens[t])
        };
        println!("  pos {} ({}) → {}", t, input_ch, top3.join(", "));
    }

    println!("\nInference time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

    // Benchmark: run 100 times
    let n = 100;
    let start = Instant::now();
    for _ in 0..n {
        let _ = model.forward(&tokens);
    }
    let total = start.elapsed();
    let ms = total.as_secs_f64() * 1000.0 / n as f64;
    let tps = tokens.len() as f64 / (ms / 1000.0);
    println!("Benchmark: {:.3}ms/inference, {:.0} tokens/sec", ms, tps);
}

fn run_training_bench(model_path: &str) {
    use mamba3_engine::train::TrainState;

    println!("Loading model for training from {}...", model_path);
    let model = Mamba3Model::from_bin(Path::new(model_path))
        .expect("Failed to load model");

    println!("Model: {} params", model.param_count());
    let mut state = TrainState::new(model, 1e-3, 0.1);

    // Training data: simple next-token prediction
    // BOS(256) + "0 1 0" + SEP(258) + "D" + EOS(257)
    let tokens: Vec<u32> = vec![256, 48, 32, 49, 32, 48, 258, 68, 257];
    let targets: Vec<u32> = vec![48, 32, 49, 32, 48, 258, 68, 257, 257]; // shifted

    println!("\nTraining for 10 steps...");
    let start = Instant::now();
    for step in 0..10 {
        let loss = state.train_step(&tokens, &targets);
        if (step + 1) % 5 == 0 {
            println!("  step {}: loss = {:.4}", step + 1, loss);
        }
    }
    let elapsed = start.elapsed();
    let ms_per_step = elapsed.as_secs_f64() * 1000.0 / 10.0;
    println!("\nTraining: {:.1}ms/step ({:.1} steps/sec)",
        ms_per_step, 1000.0 / ms_per_step);

    // Verify model changed
    let logits_after = state.model.forward(&tokens);
    let pred_after = state.model.predict(&tokens);
    println!("Predictions after training: {:?}", pred_after);
}

fn run_scan_benchmark() {
    let gpu = pollster::block_on(GpuContext::new()).expect("Failed to init GPU");

    let b: u32 = 4;
    let l: u32 = 16;
    let h: u32 = 8;
    let hd: u32 = 16;
    let ds: u32 = 16;

    let inp: Vec<f32> = (0..(b*l*h*hd*ds) as usize).map(|i| (i as f32 * 0.001).sin()).collect();
    let decay: Vec<f32> = (0..(b*l*h) as usize).map(|i| 0.5 + 0.4 * (i as f32 * 0.01).sin()).collect();
    let c: Vec<f32> = (0..(b*l*h*ds) as usize).map(|i| (i as f32 * 0.002).cos()).collect();
    let x: Vec<f32> = (0..(b*l*h*hd) as usize).map(|i| (i as f32 * 0.003).sin()).collect();
    let z_silu: Vec<f32> = (0..(b*l*h*hd) as usize).map(|i| {
        let z = (i as f32 * 0.004).sin();
        z * (1.0 / (1.0 + (-z).exp()))
    }).collect();
    let d: Vec<f32> = (0..h as usize).map(|i| 0.1 * i as f32).collect();

    // CPU reference
    let cpu = ssm_scan_cpu(&inp, &decay, &c, &x, &z_silu, &d,
        b as usize, l as usize, h as usize, hd as usize, ds as usize);
    let gpu_result = pollster::block_on(gpu.run_scan(&inp, &decay, &c, &x, &z_silu, &d, b, l, h, hd, ds))
        .expect("GPU failed");
    let max_diff: f32 = cpu.iter().zip(gpu_result.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    println!("=== SSM Scan: wgpu vs CPU ===");
    println!("  Max diff: {:.2e} {}", max_diff, if max_diff < 1e-4 { "PASS" } else { "FAIL" });

    // Benchmark
    let b2: u32 = 32;
    let l2: u32 = 64;
    let inp2: Vec<f32> = vec![0.01; (b2*l2*h*hd*ds) as usize];
    let decay2: Vec<f32> = vec![0.9; (b2*l2*h) as usize];
    let c2: Vec<f32> = vec![0.1; (b2*l2*h*ds) as usize];
    let x2: Vec<f32> = vec![0.05; (b2*l2*h*hd) as usize];
    let z2: Vec<f32> = vec![0.5; (b2*l2*h*hd) as usize];

    for _ in 0..5 {
        let _ = pollster::block_on(gpu.run_scan(&inp2, &decay2, &c2, &x2, &z2, &d, b2, l2, h, hd, ds));
    }
    let n = 100;
    let start = Instant::now();
    for _ in 0..n {
        let _ = pollster::block_on(gpu.run_scan(&inp2, &decay2, &c2, &x2, &z2, &d, b2, l2, h, hd, ds));
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!("  wgpu: {:.2}ms/scan, {:.0} tokens/sec", ms, (b2*l2) as f64 / (ms/1000.0));
}

fn ssm_scan_cpu(
    inp: &[f32], decay: &[f32], c: &[f32], x: &[f32], z_silu: &[f32], d: &[f32],
    b: usize, l: usize, h: usize, hd: usize, ds: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; b * l * h * hd];
    for bi in 0..b {
        for hi in 0..h {
            let mut state = vec![0.0f32; hd * ds];
            for t in 0..l {
                let dec = decay[bi * l * h + t * h + hi];
                for p in 0..hd {
                    for n in 0..ds {
                        let si = p * ds + n;
                        state[si] = dec * state[si] + inp[((bi*l+t)*h+hi)*hd*ds + p*ds + n];
                    }
                }
                for p in 0..hd {
                    let mut sum = 0.0f32;
                    for n in 0..ds {
                        sum += state[p*ds+n] * c[((bi*l+t)*h+hi)*ds + n];
                    }
                    sum += d[hi] * x[((bi*l+t)*h+hi)*hd + p];
                    y[((bi*l+t)*h+hi)*hd + p] = sum * z_silu[((bi*l+t)*h+hi)*hd + p];
                }
            }
        }
    }
    y
}
