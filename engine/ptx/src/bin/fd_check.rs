//! Finite-difference gradient correctness test.
//!
//! For each trainable parameter tensor, picks a few sample indices, perturbs
//! each by ±ε, measures (L(w+ε) − L(w−ε)) / (2ε) using forward-only cross-
//! entropy, and compares to the analytical gradient produced by our backward
//! kernels.  A gradient that doesn't pass FD is wrong by definition — it
//! doesn't matter whether parity trains or not.
//!
//! Usage:
//!   cargo run --release --bin fd-check
//!   cargo run --release --bin fd-check -- --eps 1e-3 --tol 1e-2

use cudarc::driver::CudaSlice;
use mamba3_engine::model::Mamba3Model;
use ptx_engine::{PtxContext, PtxModel, PtxTrainer};
use std::error::Error;
use std::sync::Arc;

struct Args {
    d_model: usize,
    d_state: usize,
    headdim: usize,
    n_layers: usize,
    vocab_size: usize,
    eps: f32,
    tol: f32,
    samples_per_tensor: usize,
    warmup_steps: usize,
}

impl Default for Args {
    fn default() -> Self {
        // Model big enough to keep typical gradients above the FD noise floor
        // (~2.5e-5 at eps=0.01).  d=64 L=2 exercises two SSM layers which is
        // also enough to catch cross-layer propagation bugs.
        Self {
            d_model: 64,
            d_state: 8,
            headdim: 16,
            n_layers: 2,
            vocab_size: 260,
            eps: 1e-2,       // larger eps → lower FD noise floor (~2.5e-5)
            tol: 1e-1,       // 10% rel-err — fp32 FD is only ~6 sig figs usable
            samples_per_tensor: 3,
            warmup_steps: 20,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = || argv[i + 1].clone();
        match key {
            "--d-model" => { a.d_model = val().parse().unwrap(); i += 2; }
            "--d-state" => { a.d_state = val().parse().unwrap(); i += 2; }
            "--headdim" => { a.headdim = val().parse().unwrap(); i += 2; }
            "--layers"  => { a.n_layers = val().parse().unwrap(); i += 2; }
            "--vocab"   => { a.vocab_size = val().parse().unwrap(); i += 2; }
            "--eps"     => { a.eps = val().parse().unwrap(); i += 2; }
            "--tol"     => { a.tol = val().parse().unwrap(); i += 2; }
            "--samples" => { a.samples_per_tensor = val().parse().unwrap(); i += 2; }
            "--warmup"  => { a.warmup_steps = val().parse().unwrap(); i += 2; }
            _ => { eprintln!("unknown arg: {}", key); std::process::exit(2); }
        }
    }
    a
}

/// CPU cross-entropy matching `cross_entropy_fwd_bwd` kernel.
fn cpu_loss(logits: &[f32], targets: &[u32], vocab: usize, l: usize) -> f32 {
    let mut loss = 0.0f32;
    for t in 0..l {
        let off = t * vocab;
        let target = targets[t] as usize;
        let mx = logits[off..off + vocab]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for i in 0..vocab {
            exp_sum += (logits[off + i] - mx).exp();
        }
        loss -= (logits[off + target] - mx) - exp_sum.ln();
    }
    loss / l as f32
}

fn forward_loss(model: &PtxModel, tokens: &[u32], targets: &[u32]) -> Result<f32, Box<dyn Error>> {
    let logits = model.forward(tokens)?;
    Ok(cpu_loss(&logits, targets, model.vocab_size, tokens.len()))
}

fn read_one(buf: &CudaSlice<f32>, stream: &Arc<cudarc::driver::CudaStream>, idx: usize) -> Result<f32, Box<dyn Error>> {
    let view = buf.slice(idx..idx + 1);
    let v = stream.memcpy_dtov(&view)?;
    Ok(v[0])
}

fn write_one(buf: &mut CudaSlice<f32>, stream: &Arc<cudarc::driver::CudaStream>, idx: usize, val: f32) -> Result<(), Box<dyn Error>> {
    let mut view = buf.slice_mut(idx..idx + 1);
    stream.memcpy_htod(&[val], &mut view)?;
    Ok(())
}

fn rel_err(a: f32, b: f32) -> f32 {
    let denom = a.abs().max(b.abs()).max(1e-8);
    (a - b).abs() / denom
}

/// Run FD check at (layer_idx, tensor_name, elem_idx).
/// Returns (analytical, fd, passed).
fn fd_one(
    trainer: &mut PtxTrainer,
    tokens: &[u32],
    targets: &[u32],
    which: Tensor,
    eps: f32,
    tol: f32,
    noise_floor: f32,
) -> Result<(f32, f32, bool, &'static str), Box<dyn Error>> {
    // Compute analytical gradient (no optimizer step)
    trainer.compute_gradients_only(tokens, targets)?;
    let stream = trainer.model.ptx.stream.clone();

    let analytical = {
        let (grad_buf, idx) = which.grad_ref(trainer);
        read_one(grad_buf, &stream, idx)?
    };

    // Save original param value
    let (_, idx) = which.grad_ref(trainer);
    let p0 = read_one(which.param_ref(trainer), &stream, idx)?;

    // +ε forward
    write_one(which.param_ref_mut(trainer), &stream, idx, p0 + eps)?;
    let loss_plus = forward_loss(&trainer.model, tokens, targets)?;

    // -ε forward
    write_one(which.param_ref_mut(trainer), &stream, idx, p0 - eps)?;
    let loss_minus = forward_loss(&trainer.model, tokens, targets)?;

    // Restore
    write_one(which.param_ref_mut(trainer), &stream, idx, p0)?;

    let fd = (loss_plus - loss_minus) / (2.0 * eps);
    let err = rel_err(analytical, fd);
    // Classification:
    //   "noise"  — both analytical and FD are within noise floor ⇒ inconclusive
    //               (pass, but don't credit it as a confirmed-correct gradient)
    //   PASS     — signal above noise AND matches FD within tol
    //   FAIL     — disagrees outside tolerance
    let (passed, tag) = if analytical.abs() < noise_floor && fd.abs() < noise_floor {
        (true, "noise")
    } else if err < tol {
        (true, "PASS")
    } else {
        (false, "FAIL")
    };
    Ok((analytical, fd, passed, tag))
}

#[derive(Clone, Copy)]
enum Tensor {
    Embed { idx: usize },
    InProj { layer: usize, idx: usize },
    OutProj { layer: usize, idx: usize },
    DParam { layer: usize, idx: usize },
    DtBias { layer: usize, idx: usize },
    LayerNormW { layer: usize, idx: usize },
    FNormW { idx: usize },
}

impl Tensor {
    fn grad_ref<'a>(&self, t: &'a mut PtxTrainer) -> (&'a CudaSlice<f32>, usize) {
        match *self {
            Tensor::Embed { idx } => (&t.train_scratch.d_embed, idx),
            Tensor::InProj { layer, idx } => (&t.train_scratch.d_in_proj_w[layer], idx),
            Tensor::OutProj { layer, idx } => (&t.train_scratch.d_out_proj_w[layer], idx),
            Tensor::DParam { layer, idx } => (&t.train_scratch.d_d_param[layer], idx),
            Tensor::DtBias { layer, idx } => (&t.train_scratch.d_dt_bias[layer], idx),
            Tensor::LayerNormW { layer, idx } => (&t.train_scratch.d_layer_norm_w[layer], idx),
            Tensor::FNormW { idx } => (&t.train_scratch.d_fnorm_w, idx),
        }
    }

    fn param_ref<'a>(&self, t: &'a PtxTrainer) -> &'a CudaSlice<f32> {
        match *self {
            Tensor::Embed { .. } => &t.model.embed_w,
            Tensor::InProj { layer, .. } => &t.model.layers[layer].in_proj_w,
            Tensor::OutProj { layer, .. } => &t.model.layers[layer].out_proj_w,
            Tensor::DParam { layer, .. } => &t.model.layers[layer].d_param,
            Tensor::DtBias { layer, .. } => &t.model.layers[layer].dt_bias,
            Tensor::LayerNormW { layer, .. } => &t.model.layers[layer].layer_norm_w,
            Tensor::FNormW { .. } => &t.model.final_norm_w,
        }
    }

    fn param_ref_mut<'a>(&self, t: &'a mut PtxTrainer) -> &'a mut CudaSlice<f32> {
        match *self {
            Tensor::Embed { .. } => &mut t.model.embed_w,
            Tensor::InProj { layer, .. } => &mut t.model.layers[layer].in_proj_w,
            Tensor::OutProj { layer, .. } => &mut t.model.layers[layer].out_proj_w,
            Tensor::DParam { layer, .. } => &mut t.model.layers[layer].d_param,
            Tensor::DtBias { layer, .. } => &mut t.model.layers[layer].dt_bias,
            Tensor::LayerNormW { layer, .. } => &mut t.model.layers[layer].layer_norm_w,
            Tensor::FNormW { .. } => &mut t.model.final_norm_w,
        }
    }

    fn name(&self) -> String {
        match *self {
            Tensor::Embed { idx } => format!("embed_w[{}]", idx),
            Tensor::InProj { layer, idx } => format!("layer{}.in_proj_w[{}]", layer, idx),
            Tensor::OutProj { layer, idx } => format!("layer{}.out_proj_w[{}]", layer, idx),
            Tensor::DParam { layer, idx } => format!("layer{}.d_param[{}]", layer, idx),
            Tensor::DtBias { layer, idx } => format!("layer{}.dt_bias[{}]", layer, idx),
            Tensor::LayerNormW { layer, idx } => format!("layer{}.layer_norm_w[{}]", layer, idx),
            Tensor::FNormW { idx } => format!("final_norm_w[{}]", idx),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args();
    println!("=== Finite-difference gradient check ===");
    println!(
        "Model: d={}, L={}, dS={}, hd={}, V={}  eps={}  tol={}  warmup={}",
        args.d_model, args.n_layers, args.d_state, args.headdim,
        args.vocab_size, args.eps, args.tol, args.warmup_steps
    );
    let ptx = Arc::new(PtxContext::new()?);

    let cpu_model = Mamba3Model::new_random(
        args.d_model, args.d_state, args.headdim, args.n_layers, args.vocab_size,
    );
    let gpu_model = PtxModel::from_cpu(&cpu_model, ptx.clone(), 16)?;
    let mut trainer = PtxTrainer::new(gpu_model, 1e-3, 0.1, 16)?;

    // Fixed input: diverse 8-token sequence across the vocab so every path
    // participates.  Targets are deliberately un-correlated with tokens so
    // loss is non-trivial (well above softmax-uniform).
    let tokens: Vec<u32> = vec![47, 132, 5, 201, 88, 19, 249, 73];
    let targets: Vec<u32> = vec![250, 11, 174, 66, 3, 189, 42, 100];

    // Warm up the model: at fresh init most gradients sit under the FD noise
    // floor, so FD tags them "noise" (inconclusive). A handful of training
    // steps pushes weights off the symmetric init and grows the signal.
    if args.warmup_steps > 0 {
        print!("warming up: {} train steps ... ", args.warmup_steps);
        let mut last_loss = 0.0f32;
        for _ in 0..args.warmup_steps {
            last_loss = trainer.train_step(&tokens, &targets)?;
        }
        println!("loss={:.4}", last_loss);
    }

    // Populate gradients once, then pick the indices with the LARGEST
    // analytical gradient magnitude per tensor — those are above FD noise
    // floor and thus FD-testable.  Random sampling puts us in noise almost
    // everywhere at fresh init.
    trainer.compute_gradients_only(&tokens, &targets)?;
    let stream = trainer.model.ptx.stream.clone();

    let top_indices = |buf: &CudaSlice<f32>, k: usize| -> Result<Vec<usize>, Box<dyn Error>> {
        let vals = stream.memcpy_dtov(buf)?;
        let mut paired: Vec<(usize, f32)> = vals.iter().enumerate()
            .map(|(i, v)| (i, v.abs()))
            .collect();
        paired.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(paired.into_iter().take(k).map(|(i, _)| i).collect())
    };

    let s = args.samples_per_tensor;
    let mut checks: Vec<Tensor> = Vec::new();

    for idx in top_indices(&trainer.train_scratch.d_embed, s)? {
        checks.push(Tensor::Embed { idx });
    }
    for li in 0..args.n_layers {
        for idx in top_indices(&trainer.train_scratch.d_in_proj_w[li], s)? {
            checks.push(Tensor::InProj { layer: li, idx });
        }
        for idx in top_indices(&trainer.train_scratch.d_out_proj_w[li], s)? {
            checks.push(Tensor::OutProj { layer: li, idx });
        }
        for idx in top_indices(&trainer.train_scratch.d_d_param[li],
                                cpu_model.n_heads.min(s))? {
            checks.push(Tensor::DParam { layer: li, idx });
        }
        for idx in top_indices(&trainer.train_scratch.d_dt_bias[li],
                                cpu_model.n_heads.min(s))? {
            checks.push(Tensor::DtBias { layer: li, idx });
        }
        for idx in top_indices(&trainer.train_scratch.d_layer_norm_w[li], s)? {
            checks.push(Tensor::LayerNormW { layer: li, idx });
        }
    }
    for idx in top_indices(&trainer.train_scratch.d_fnorm_w, s)? {
        checks.push(Tensor::FNormW { idx });
    }

    // FD noise floor = 1 ULP of loss divided by 2·eps.  For fp32 loss ~ O(1)
    // (cross-entropy on 64-way vocab is log(64) ≈ 4.2), one ULP is ~5e-7, so
    // noise floor at eps=0.01 is ~2.5e-5.  Scale conservatively.
    let noise_floor = 1e-6f32.max(5e-7 / args.eps);

    let mut n_pass = 0usize;
    let mut n_fail = 0usize;
    let mut n_noise = 0usize;
    let mut n_total = 0usize;
    println!("\n{:<28} {:>14} {:>14} {:>10}  {}",
        "tensor[idx]", "analytical", "finite-diff", "rel-err", "verdict");
    println!("FD noise floor ≈ {:.2e}  (grads below this are inconclusive, not FAIL)",
        noise_floor);
    println!("{}", "-".repeat(85));
    for tensor in &checks {
        match fd_one(&mut trainer, &tokens, &targets, *tensor, args.eps, args.tol, noise_floor) {
            Ok((analytical, fd, _passed, tag)) => {
                let err = rel_err(analytical, fd);
                println!("{:<28} {:>14.6} {:>14.6} {:>10.3}  {}",
                    tensor.name(), analytical, fd, err, tag);
                match tag {
                    "PASS" => n_pass += 1,
                    "FAIL" => n_fail += 1,
                    "noise" => n_noise += 1,
                    _ => {}
                }
                n_total += 1;
            }
            Err(e) => {
                println!("{:<28} ERROR: {}", tensor.name(), e);
                n_total += 1;
            }
        }
    }

    println!("\nResult: {} confirmed-correct, {} inconclusive-noise, {} confirmed-wrong  (of {})",
        n_pass, n_noise, n_fail, n_total);
    Ok(())
}
