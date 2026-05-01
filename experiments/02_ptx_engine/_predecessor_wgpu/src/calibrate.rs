//! Hardware calibration — runs once, persists to disk, used for all subsequent runs.
//!
//! Benchmarks matmul strategies at various sizes and finds optimal thresholds.
//! Results cached in ~/.mamba3/calibration.json
//!
//! Usage:
//!   mamba3-bench --calibrate           # run calibration
//!   # Results used automatically by model inference/training

use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct HardwareProfile {
    pub hostname: String,
    pub arch: String,
    pub cpu_cores: usize,
    pub has_gpu: bool,
    pub gpu_name: String,
    pub parallel_threshold: usize,  // ops below this → single-thread
    pub use_gpu_matmul: bool,       // use wgpu for matmul
    pub calibrated_at: u64,
}

impl HardwareProfile {
    pub fn load_or_calibrate() -> Self {
        let path = Self::config_path();
        if let Ok(data) = std::fs::read_to_string(&path) {
            if let Some(profile) = Self::parse(&data) {
                eprintln!("  Loaded calibration from {}", path.display());
                return profile;
            }
        }
        eprintln!("  No calibration found — running hardware benchmark...");
        let profile = Self::calibrate();
        if let Err(e) = profile.save() {
            eprintln!("  Warning: couldn't save calibration: {}", e);
        }
        profile
    }

    fn config_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        let dir = PathBuf::from(home).join(".mamba3");
        let _ = std::fs::create_dir_all(&dir);
        dir.join("calibration.json")
    }

    pub fn calibrate() -> Self {
        let hostname = hostname();
        let arch = std::env::consts::ARCH.to_string();
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get()).unwrap_or(1);

        eprintln!("  Host: {} ({}, {} cores)", hostname, arch, cpu_cores);

        // Test GPU availability
        let (has_gpu, gpu_name) = test_gpu();
        if has_gpu {
            eprintln!("  GPU: {}", gpu_name);
        } else {
            eprintln!("  GPU: none");
        }

        // Benchmark matmul at different sizes
        let sizes = vec![
            (7, 64, 320),     // small: single sequence, in_proj
            (7, 128, 64),     // small: single sequence, out_proj
            (32, 64, 320),    // medium: batched in_proj
            (64, 128, 64),    // medium: batched out_proj
            (128, 256, 640),  // large: bigger model
        ];

        let mut parallel_threshold = usize::MAX;

        for (m, k, n) in &sizes {
            let ops = m * k * n;
            let a = vec![0.1f32; m * k];
            let b = vec![0.1f32; n * k];
            let mut out = vec![0.0f32; m * n];

            // Single-thread SIMD
            let t0 = Instant::now();
            for _ in 0..20 {
                single_matmul(&mut out, &a, &b, *m, *k, *n);
            }
            let single_us = t0.elapsed().as_micros() as f64 / 20.0;

            // Parallel (rayon)
            let t0 = Instant::now();
            for _ in 0..20 {
                parallel_matmul(&mut out, &a, &b, *m, *k, *n);
            }
            let par_us = t0.elapsed().as_micros() as f64 / 20.0;

            let winner = if par_us < single_us { "parallel" } else { "single" };
            let speedup = single_us / par_us;
            eprintln!("  {}x{}x{} ({} ops): single={:.0}us parallel={:.0}us → {} ({:.1}x)",
                m, k, n, ops, single_us, par_us, winner, speedup);

            if par_us < single_us && parallel_threshold == usize::MAX {
                parallel_threshold = ops;
            }
        }

        if parallel_threshold == usize::MAX {
            eprintln!("  Decision: always single-thread SIMD");
        } else {
            eprintln!("  Decision: parallel above {} ops", parallel_threshold);
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();

        HardwareProfile {
            hostname, arch, cpu_cores, has_gpu, gpu_name,
            parallel_threshold,
            use_gpu_matmul: has_gpu, // enable if GPU available
            calibrated_at: now,
        }
    }

    fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let json = format!(
            "{{\n  \"hostname\": \"{}\",\n  \"arch\": \"{}\",\n  \"cpu_cores\": {},\n  \
             \"has_gpu\": {},\n  \"gpu_name\": \"{}\",\n  \"parallel_threshold\": {},\n  \
             \"use_gpu_matmul\": {},\n  \"calibrated_at\": {}\n}}",
            self.hostname, self.arch, self.cpu_cores,
            self.has_gpu, self.gpu_name, self.parallel_threshold,
            self.use_gpu_matmul, self.calibrated_at
        );
        std::fs::write(Self::config_path(), json)?;
        eprintln!("  Saved calibration to {}", Self::config_path().display());
        Ok(())
    }

    fn parse(json: &str) -> Option<Self> {
        // Simple JSON parsing without serde
        let get_str = |key: &str| -> Option<String> {
            let pat = format!("\"{}\":", key);
            let pos = json.find(&pat)? + pat.len();
            let rest = json[pos..].trim();
            if rest.starts_with('"') {
                let end = rest[1..].find('"')? + 1;
                Some(rest[1..end].to_string())
            } else {
                None
            }
        };
        let get_num = |key: &str| -> Option<usize> {
            let pat = format!("\"{}\":", key);
            let pos = json.find(&pat)? + pat.len();
            let rest = json[pos..].trim();
            let end = rest.find(|c: char| !c.is_digit(10)).unwrap_or(rest.len());
            rest[..end].parse().ok()
        };
        let get_bool = |key: &str| -> Option<bool> {
            let pat = format!("\"{}\":", key);
            let pos = json.find(&pat)? + pat.len();
            let rest = json[pos..].trim();
            Some(rest.starts_with("true"))
        };

        Some(HardwareProfile {
            hostname: get_str("hostname")?,
            arch: get_str("arch")?,
            cpu_cores: get_num("cpu_cores")?,
            has_gpu: get_bool("has_gpu")?,
            gpu_name: get_str("gpu_name").unwrap_or_default(),
            parallel_threshold: get_num("parallel_threshold")?,
            use_gpu_matmul: get_bool("use_gpu_matmul")?,
            calibrated_at: get_num("calibrated_at")? as u64,
        })
    }
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn test_gpu() -> (bool, String) {
    match pollster::block_on(crate::scan::GpuContext::new()) {
        Ok(gpu) => {
            // Run a tiny scan to verify it works
            let inp = vec![0.1f32; 16 * 16];
            let decay = vec![0.9f32; 1];
            let c = vec![0.1f32; 16];
            let x = vec![0.1f32; 16];
            let z = vec![0.1f32; 16];
            let d = vec![0.1f32; 1];
            match pollster::block_on(gpu.run_scan(&inp, &decay, &c, &x, &z, &d, 1, 1, 1, 16, 16)) {
                Ok(_) => (true, "wgpu".into()),
                Err(_) => (false, String::new()),
            }
        }
        Err(_) => (false, String::new()),
    }
}

fn single_matmul(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            out[i * n + j] = crate::model::dot_simd(a_row, &b[j * k..(j + 1) * k], k);
        }
    }
}

fn parallel_matmul(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    use rayon::prelude::*;
    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            row[j] = crate::model::dot_simd(a_row, &b[j * k..(j + 1) * k], k);
        }
    });
}

/// Global profile — loaded once, used everywhere
static PROFILE: std::sync::OnceLock<HardwareProfile> = std::sync::OnceLock::new();

pub fn get_profile() -> &'static HardwareProfile {
    PROFILE.get_or_init(HardwareProfile::load_or_calibrate)
}
