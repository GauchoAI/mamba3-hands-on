//! Evaluate the trained parity.bin model on many random parity inputs.
//! Used to verify whether the Rust CPU + PTX match AND whether the model
//! actually learned parity (independent of our Rust TrainState code path).

use mamba3_engine::model::Mamba3Model;
use ptx_engine::{PtxContext, PtxModel};
use std::error::Error;
use std::path::Path;
use std::sync::Arc;

fn encode_parity(bits: &[u32]) -> (Vec<u32>, u32) {
    // "b1 b2 b3 ..." space-separated. Tokens:
    // BOS(256), '0'/'1' tokens, spaces, SEP(258).  Final target is at SEP position.
    let mut toks: Vec<u32> = vec![256];
    for (i, &b) in bits.iter().enumerate() {
        if i > 0 { toks.push(32); } // space
        toks.push(48 + b);           // '0' or '1'
    }
    toks.push(258); // SEP
    let parity = bits.iter().sum::<u32>() & 1;
    let expected = if parity == 0 { 83 } else { 68 }; // S or D
    (toks, expected)
}

fn main() -> Result<(), Box<dyn Error>> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/parity.bin".to_string());
    println!("=== Parity model accuracy check ===");
    println!("Model: {}", model_path);

    let cpu_model = Mamba3Model::from_bin(Path::new(&model_path))?;
    println!(
        "  d={}, L={}, d_state={}, headdim={}, vocab={}, d_inner={}, H={}",
        cpu_model.d_model, cpu_model.n_layers, cpu_model.d_state,
        cpu_model.headdim, cpu_model.vocab_size, cpu_model.d_inner, cpu_model.n_heads
    );

    let ptx = Arc::new(PtxContext::new()?);
    let gpu_model = PtxModel::from_cpu(&cpu_model, ptx.clone(), 64)?;

    let mut rng: u64 = 42;
    let mut next = || -> u32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) & 1) as u32
    };

    // Accuracy at several bit lengths
    for &n_bits in &[3usize, 4, 5, 6, 8] {
        let mut correct_cpu = 0usize;
        let mut correct_ptx = 0usize;
        let n_trials = 200;
        let mut max_fwd_diff = 0.0f32;
        let mut mismatch = 0usize;

        for _ in 0..n_trials {
            let bits: Vec<u32> = (0..n_bits).map(|_| next()).collect();
            let (toks, expected) = encode_parity(&bits);

            let cpu_logits = cpu_model.forward(&toks);
            let ptx_logits = gpu_model.forward(&toks)?;

            let diff = cpu_logits.iter().zip(ptx_logits.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if diff > max_fwd_diff { max_fwd_diff = diff; }

            // Prediction at last position (SEP). Both S and D relevant.
            let v = cpu_model.vocab_size;
            let last = toks.len() - 1;
            let argmax = |logits: &[f32]| -> u32 {
                let mut best = (0usize, f32::NEG_INFINITY);
                for i in 0..v {
                    if logits[last * v + i] > best.1 { best = (i, logits[last * v + i]); }
                }
                best.0 as u32
            };
            let cpu_pred = argmax(&cpu_logits);
            let ptx_pred = argmax(&ptx_logits);
            if cpu_pred != ptx_pred { mismatch += 1; }
            if cpu_pred == expected { correct_cpu += 1; }
            if ptx_pred == expected { correct_ptx += 1; }
        }
        println!(
            "n_bits={}: CPU acc={:.1}%  PTX acc={:.1}%  cpu↔ptx argmax mismatch={}  max |CPU-PTX fwd diff|={:.3e}",
            n_bits,
            100.0 * correct_cpu as f32 / n_trials as f32,
            100.0 * correct_ptx as f32 / n_trials as f32,
            mismatch,
            max_fwd_diff
        );
    }
    Ok(())
}
