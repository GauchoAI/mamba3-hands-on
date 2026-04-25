//! Forward-parity check: load PyTorch-exported weights via Mamba3Model::from_bin,
//! upload to PtxModel, run forward on a fixed token sequence, compare logits
//! element-wise against the PyTorch logits saved alongside the weights.
//!
//! This is the question fd-check can't answer: does our forward produce the
//! same output as mamba3_minimal.Mamba3Block at identical weights?  If the
//! max-abs error is above ~1e-4, we have a forward kernel bug.
//!
//! Usage:
//!   cargo run --release --bin forward-parity
//! Reads:
//!   /tmp/forward_parity.bin            (weights, our from_bin format)
//!   /tmp/forward_parity_logits.bin     (PyTorch logits, header L V then floats)
//!   /tmp/forward_parity_tokens.bin     (input tokens, raw u32s)

use mamba3_engine::model::Mamba3Model;
use ptx_engine::{PtxContext, PtxModel};
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

fn read_floats(path: &str) -> Vec<f32> {
    let mut f = File::open(path).expect(&format!("open {}", path));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    bytemuck::cast_slice::<u8, f32>(&buf).to_vec()
}

fn main() -> Result<(), Box<dyn Error>> {
    let cpu_model = Mamba3Model::from_bin(Path::new("/tmp/forward_parity.bin"))?;
    println!(
        "loaded weights: d={} dS={} hd={} L={} V={}",
        cpu_model.d_model, cpu_model.d_state, cpu_model.headdim,
        cpu_model.n_layers, cpu_model.vocab_size,
    );

    let ptx = Arc::new(PtxContext::new()?);
    let max_seq = 64;
    let gpu_model = PtxModel::from_cpu(&cpu_model, ptx.clone(), max_seq)?;

    // Read tokens.
    let mut tokens_bytes = Vec::new();
    File::open("/tmp/forward_parity_tokens.bin")?.read_to_end(&mut tokens_bytes)?;
    let tokens: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&tokens_bytes).to_vec();
    println!("input tokens ({}): {:?}", tokens.len(), tokens);

    // Read PyTorch logits header (L, V) then floats.
    let mut logits_buf = Vec::new();
    File::open("/tmp/forward_parity_logits.bin")?.read_to_end(&mut logits_buf)?;
    let header: &[u32] = bytemuck::cast_slice(&logits_buf[..8]);
    let l = header[0] as usize;
    let v = header[1] as usize;
    let pytorch_logits: &[f32] = bytemuck::cast_slice(&logits_buf[8..]);
    assert_eq!(pytorch_logits.len(), l * v);
    println!("PyTorch logits: L={} V={}  sum={:.6}", l, v, pytorch_logits.iter().sum::<f32>());

    // Run our forward.
    let our_logits = gpu_model.forward(&tokens)?;
    assert_eq!(our_logits.len(), l * v);
    println!("PTX     logits: L={} V={}  sum={:.6}", l, v, our_logits.iter().sum::<f32>());

    // Element-wise comparison.
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    let mut sum_abs = 0.0f64;
    for i in 0..l*v {
        let d = (our_logits[i] - pytorch_logits[i]).abs();
        if d > max_abs { max_abs = d; max_idx = i; }
        sum_abs += d as f64;
    }
    let mean_abs = sum_abs / (l * v) as f64;
    let row = max_idx / v;
    let col = max_idx % v;
    println!(
        "\nForward-parity: max_abs_err={:.6e} at [t={}, v={}]  mean_abs_err={:.6e}",
        max_abs, row, col, mean_abs,
    );
    println!(
        "  PyTorch[{}, {}] = {:.6}    PTX[{}, {}] = {:.6}    diff = {:+.6}",
        row, col, pytorch_logits[max_idx],
        row, col, our_logits[max_idx],
        our_logits[max_idx] - pytorch_logits[max_idx],
    );

    let verdict = if max_abs < 1e-4 { "BIT-PARITY ✓" }
        else if max_abs < 1e-2 { "small drift (numerics)" }
        else { "FORWARD MISMATCH — ours diverges from PyTorch" };
    println!("Verdict: {}", verdict);
    Ok(())
}
