//! Binary batch file format — the wire protocol between Python (which owns
//! task generators, tokenization, and curriculum) and ptxd (which only knows
//! how to do forward+backward+optimizer on a stream of `(tokens, targets)`
//! pairs).
//!
//! Layout (little-endian throughout):
//!
//! ```text
//!   [magic: u32 = 0x42544348]   ('BTCH')
//!   [version: u32 = 1]
//!   [n_examples: u32]
//!   [flags: u32 = 0]            (reserved for future: teacher_logits flag, etc.)
//!   for each example:
//!     [n_tokens: u32]
//!     [tokens:   u32 * n_tokens]
//!     [targets:  u32 * n_tokens]   (u32::MAX = ignore at this position)
//! ```
//!
//! The Python writer is in `batch_writer.py` at the repo root. Tests:
//! the round-trip is exercised end-to-end by `ptxd_specialist.py` for
//! every task — if a batch is malformed, training fails immediately
//! with a parse error.

use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

pub const BATCH_MAGIC: u32 = 0x42544348;
pub const BATCH_VERSION: u32 = 1;

/// One supervised example: a token sequence plus a target sequence of the
/// same length. `target == u32::MAX` at a position means "ignore — don't
/// supervise here." Matches the IGNORE convention `accumulate_gradients`
/// already uses.
#[derive(Debug, Clone)]
pub struct Example {
    pub tokens: Vec<u32>,
    pub targets: Vec<u32>,
}

/// In-memory batch reader. Loads the entire file once at construction
/// (typical cycle-sized files are a few MB). Cycles back to the start
/// when exhausted, so callers don't need to count.
pub struct BatchReader {
    examples: Vec<Example>,
    cursor: usize,
}

impl BatchReader {
    pub fn open(path: &Path) -> Result<Self, Box<dyn Error>> {
        let f = File::open(path)
            .map_err(|e| format!("BatchReader::open({}): {}", path.display(), e))?;
        let mut r = BufReader::new(f);

        let magic = read_u32(&mut r)?;
        if magic != BATCH_MAGIC {
            return Err(format!("bad magic: 0x{:x} (want 0x{:x})", magic, BATCH_MAGIC).into());
        }
        let version = read_u32(&mut r)?;
        if version != BATCH_VERSION {
            return Err(format!("unsupported batch format version {}", version).into());
        }
        let n_examples = read_u32(&mut r)? as usize;
        let _flags = read_u32(&mut r)?;

        let mut examples = Vec::with_capacity(n_examples);
        for i in 0..n_examples {
            let n_tokens = read_u32(&mut r)? as usize;
            if n_tokens == 0 || n_tokens > 65536 {
                return Err(format!("example {}: implausible n_tokens={}", i, n_tokens).into());
            }
            let mut tokens = vec![0u32; n_tokens];
            let mut targets = vec![0u32; n_tokens];
            read_u32_vec(&mut r, &mut tokens)?;
            read_u32_vec(&mut r, &mut targets)?;
            examples.push(Example { tokens, targets });
        }
        Ok(Self { examples, cursor: 0 })
    }

    pub fn n_examples(&self) -> usize { self.examples.len() }

    /// Index-based read without advancing the cursor. Used by the runner
    /// at construction time to compute `max_seq` from the longest example.
    pub fn peek_example(&self, idx: usize) -> &Example { &self.examples[idx] }

    /// Return the next example, wrapping around to the start of the file
    /// when we run out. The wrap-around makes eval files trivially reusable
    /// across cycles.
    pub fn next_example(&mut self) -> &Example {
        let ex = &self.examples[self.cursor];
        self.cursor = (self.cursor + 1) % self.examples.len();
        ex
    }

    /// Reset the cursor to the start. Used between eval passes so the
    /// first example is reproducible.
    pub fn rewind(&mut self) { self.cursor = 0; }
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, Box<dyn Error>> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u32_vec<R: Read>(r: &mut R, out: &mut [u32]) -> Result<(), Box<dyn Error>> {
    let bytes = bytemuck::cast_slice_mut::<u32, u8>(out);
    r.read_exact(bytes)?;
    if cfg!(target_endian = "big") {
        for x in out.iter_mut() { *x = u32::from_le(*x); }
    }
    Ok(())
}
