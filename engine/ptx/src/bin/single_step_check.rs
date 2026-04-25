//! Take ONE training step from the same starting weights, on the same input,
//! as PyTorch did in /tmp/single_step_compare.py. Compare resulting weights
//! tensor by tensor.
//!
//! Reads:
//!   /tmp/single_step_initial.bin       (starting weights)
//!   /tmp/single_step_pytorch_after.bin (PyTorch weights after 1 step)
//!   /tmp/single_step_tokens.bin        (input tokens)
//!   /tmp/single_step_meta.bin          (answer_pos: u32, pytorch_loss: f32)

use mamba3_engine::model::Mamba3Model;
use ptx_engine::{PtxContext, PtxModel, PtxTrainer};
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

fn read_bytes(path: &str) -> Vec<u8> {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {} failed: {}", path, e));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    buf
}

fn read_floats(path: &str) -> Vec<f32> {
    let buf = read_bytes(path);
    bytemuck::cast_slice::<u8, f32>(&buf).to_vec()
}

fn rel_err(a: f32, b: f32) -> f32 {
    let m = a.abs().max(b.abs()).max(1e-12);
    (a - b).abs() / m
}

fn diff_named(name: &str, ours: &[f32], pytorch: &[f32]) -> (f32, f32, f32) {
    assert_eq!(ours.len(), pytorch.len(), "{}: length mismatch", name);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut sum_abs = 0.0f64;
    for i in 0..ours.len() {
        let d = (ours[i] - pytorch[i]).abs();
        if d > max_abs { max_abs = d; }
        let r = rel_err(ours[i], pytorch[i]);
        if r > max_rel { max_rel = r; }
        sum_abs += d as f64;
    }
    let mean_abs = (sum_abs / ours.len() as f64) as f32;
    println!("  {:>20}  N={:>7}  max_abs={:.3e}  mean_abs={:.3e}  max_rel={:.3e}",
        name, ours.len(), max_abs, mean_abs, max_rel);
    (max_abs, mean_abs, max_rel)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Read starting state.
    let cpu_model = Mamba3Model::from_bin(Path::new("/tmp/single_step_initial.bin"))?;
    println!("loaded weights: d={} dS={} hd={} L={} V={}",
        cpu_model.d_model, cpu_model.d_state, cpu_model.headdim,
        cpu_model.n_layers, cpu_model.vocab_size);

    let ptx = Arc::new(PtxContext::new()?);
    let max_seq = 64;
    let gpu_model = PtxModel::from_cpu(&cpu_model, ptx.clone(), max_seq)?;
    let mut trainer = PtxTrainer::new(gpu_model, 1e-3, 0.1, max_seq)?;
    trainer.warmup_steps = 0;   // no warmup so step 1 has full lr — matches PyTorch

    // Tokens + target.
    let tokens: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&read_bytes("/tmp/single_step_tokens.bin")).to_vec();
    let meta_bytes = read_bytes("/tmp/single_step_meta.bin");
    let answer_pos = u32::from_le_bytes([meta_bytes[0], meta_bytes[1], meta_bytes[2], meta_bytes[3]]) as usize;
    let pytorch_loss = f32::from_le_bytes([meta_bytes[4], meta_bytes[5], meta_bytes[6], meta_bytes[7]]);
    let mut targets: Vec<u32> = vec![u32::MAX; tokens.len()];
    targets[answer_pos] = tokens[answer_pos + 1];   // ANSWER token
    println!("tokens: {:?}  answer_pos={}  target={}  pytorch_loss={:.6}",
        tokens, answer_pos, targets[answer_pos], pytorch_loss);

    // Run one PTX training step.
    let our_loss = trainer.train_step(&tokens, &targets)?;
    println!("PTX loss before step: {:.6}    (PyTorch: {:.6}, diff={:+.6e})",
        our_loss, pytorch_loss, our_loss - pytorch_loss);

    // Read our updated weights from device — easiest: write a helper that
    // serialises the live PtxModel back to the from_bin layout, then read
    // PyTorch's after.bin and diff. For now, read each tensor individually.
    let stream = trainer.model.ptx.stream.clone();
    let read = |buf: &cudarc::driver::CudaSlice<f32>| -> Vec<f32> {
        stream.memcpy_dtov(buf).unwrap()
    };

    // Load PyTorch after.bin and re-parse it tensor-by-tensor.
    let after_bytes = read_bytes("/tmp/single_step_pytorch_after.bin");
    let after_floats: &[f32] = bytemuck::cast_slice(&after_bytes[28..]);  // skip 7-u32 header
    let mut off = 0usize;
    let mut take = |n: usize| -> &[f32] { let s = &after_floats[off..off+n]; off += n; s };

    let d = cpu_model.d_model;
    let v = cpu_model.vocab_size;
    let dip = trainer.model.d_in_proj;
    let di = trainer.model.d_inner;
    let nh = trainer.model.n_heads;
    let ds = cpu_model.d_state;

    println!("\nPer-tensor diff (PTX after step  vs  PyTorch after step):");
    diff_named("embed_w", &read(&trainer.model.embed_w), take(v * d));
    diff_named("embed_norm_w", &read(&trainer.model.embed_norm_w), take(d));
    diff_named("embed_norm_b", &read(&trainer.model.embed_norm_b), take(d));
    for li in 0..cpu_model.n_layers {
        let layer = &trainer.model.layers[li];
        diff_named(&format!("L{}.in_proj_w", li),    &read(&layer.in_proj_w), take(dip * d));
        diff_named(&format!("L{}.out_proj_w", li),   &read(&layer.out_proj_w), take(d * di));
        diff_named(&format!("L{}.dt_bias", li),      &read(&layer.dt_bias), take(nh));
        diff_named(&format!("L{}.d_param", li),      &read(&layer.d_param), take(nh));
        diff_named(&format!("L{}.b_norm_w", li),     &read(&layer.b_norm_w), take(ds));
        diff_named(&format!("L{}.b_norm_b", li),     &read(&layer.b_norm_b), take(ds));
        diff_named(&format!("L{}.c_norm_w", li),     &read(&layer.c_norm_w), take(ds));
        diff_named(&format!("L{}.c_norm_b", li),     &read(&layer.c_norm_b), take(ds));
        diff_named(&format!("L{}.layer_norm_w", li), &read(&layer.layer_norm_w), take(d));
        diff_named(&format!("L{}.layer_norm_b", li), &read(&layer.layer_norm_b), take(d));
        let pytorch_scale = take(1)[0];
        let our_scale = trainer.model.layers[li].scale;
        println!("  {:>20}  N={:>7}  ours={:+.6}  pytorch={:+.6}  diff={:+.3e}",
            format!("L{}.scale", li), 1, our_scale, pytorch_scale, our_scale - pytorch_scale);
    }
    diff_named("final_norm_w", &read(&trainer.model.final_norm_w), take(d));
    diff_named("final_norm_b", &read(&trainer.model.final_norm_b), take(d));
    Ok(())
}
