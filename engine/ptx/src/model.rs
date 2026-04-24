//! PtxModel — forward inference using hand-written PTX kernels on NVIDIA GPU.
//!
//! v1: per-op kernels, bit-exact vs CPU reference. Correctness first,
//! creative iteration in v2+.

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use mamba3_engine::model::Mamba3Model;
use std::error::Error;
use std::sync::Arc;

use crate::runtime::PtxContext;

pub struct PtxLayer {
    pub in_proj_w: CudaSlice<f32>,
    pub out_proj_w: CudaSlice<f32>,
    pub dt_bias: CudaSlice<f32>,
    pub d_param: CudaSlice<f32>,
    pub b_norm_w: CudaSlice<f32>,
    pub b_norm_b: CudaSlice<f32>,
    pub c_norm_w: CudaSlice<f32>,
    pub c_norm_b: CudaSlice<f32>,
    pub layer_norm_w: CudaSlice<f32>,
    pub layer_norm_b: CudaSlice<f32>,
    pub scale: f32,
    pub d_in_proj: usize,
    pub num_rope_angles: usize,
}

pub struct PtxModel {
    pub ptx: Arc<PtxContext>,

    pub d_model: usize,
    pub d_state: usize,
    pub d_inner: usize,
    pub headdim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub vocab_size: usize,

    pub embed_w: CudaSlice<f32>,
    pub embed_norm_w: CudaSlice<f32>,
    pub embed_norm_b: CudaSlice<f32>,
    pub layers: Vec<PtxLayer>,
    pub final_norm_w: CudaSlice<f32>,
    pub final_norm_b: CudaSlice<f32>,
}

impl PtxModel {
    pub fn from_cpu(model: &Mamba3Model, ptx: Arc<PtxContext>) -> Result<Self, Box<dyn Error>> {
        let stream = ptx.ctx.default_stream();
        let embed_w = stream.memcpy_stod(&model.embed_w)?;
        let embed_norm_w = stream.memcpy_stod(&model.embed_norm_w)?;
        let embed_norm_b = stream.memcpy_stod(&model.embed_norm_b)?;
        let final_norm_w = stream.memcpy_stod(&model.final_norm_w)?;
        let final_norm_b = stream.memcpy_stod(&model.final_norm_b)?;

        let mut layers = Vec::new();
        for lw in &model.layers {
            layers.push(PtxLayer {
                in_proj_w: stream.memcpy_stod(&lw.in_proj_w)?,
                out_proj_w: stream.memcpy_stod(&lw.out_proj_w)?,
                dt_bias: stream.memcpy_stod(&lw.dt_bias)?,
                d_param: stream.memcpy_stod(&lw.d_param)?,
                b_norm_w: stream.memcpy_stod(&lw.b_norm_w)?,
                b_norm_b: stream.memcpy_stod(&lw.b_norm_b)?,
                c_norm_w: stream.memcpy_stod(&lw.c_norm_w)?,
                c_norm_b: stream.memcpy_stod(&lw.c_norm_b)?,
                layer_norm_w: stream.memcpy_stod(&lw.layer_norm_w)?,
                layer_norm_b: stream.memcpy_stod(&lw.layer_norm_b)?,
                scale: lw.scale,
                d_in_proj: lw.d_in_proj,
                num_rope_angles: lw.num_rope_angles,
            });
        }

        Ok(Self {
            ptx,
            d_model: model.d_model,
            d_state: model.d_state,
            d_inner: model.d_inner,
            headdim: model.headdim,
            n_heads: model.n_heads,
            n_layers: model.n_layers,
            vocab_size: model.vocab_size,
            embed_w,
            embed_norm_w,
            embed_norm_b,
            layers,
            final_norm_w,
            final_norm_b,
        })
    }

    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
        let stream = self.ptx.ctx.default_stream();
        let l = tokens.len();
        let d = self.d_model;
        let di = self.d_inner;
        let ds = self.d_state;
        let hd = self.headdim;
        let nh = self.n_heads;

        let tokens_dev = stream.memcpy_stod(tokens)?;
        let mut x = stream.alloc_zeros::<f32>(l * d)?;

        // 1. Embedding gather
        self.launch_embed_gather(&stream, &tokens_dev, &mut x, l)?;

        // 2. Embed norm
        self.launch_layer_norm(&stream, &mut x, &self.embed_norm_w, &self.embed_norm_b, l, d)?;

        // 3. Layers
        for layer in &self.layers {
            // Pre-norm on a copy
            let mut x_normed = stream.alloc_zeros::<f32>(l * d)?;
            {
                let n_i = (l * d) as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.copy_f32);
                lb.arg(&x);
                lb.arg(&mut x_normed);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: (((l * d) as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
            self.launch_layer_norm(
                &stream,
                &mut x_normed,
                &layer.layer_norm_w,
                &layer.layer_norm_b,
                l,
                d,
            )?;

            // in_proj: proj = x_normed @ in_proj_w.T  → (L, dip)
            let dip = layer.d_in_proj;
            let mut proj = stream.alloc_zeros::<f32>(l * dip)?;
            self.launch_matmul_t(&stream, &x_normed, &layer.in_proj_w, &mut proj, l, dip, d)?;

            // SSM params: dt, decay, trap
            let mut dt = stream.alloc_zeros::<f32>(l * nh)?;
            let mut decay = stream.alloc_zeros::<f32>(l * nh)?;
            let mut trap = stream.alloc_zeros::<f32>(l * nh)?;
            {
                let l_i = l as i32;
                let h_i = nh as i32;
                let dip_i = dip as i32;
                let di_i = di as i32;
                let ds_i = ds as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.compute_ssm_params);
                lb.arg(&proj);
                lb.arg(&layer.dt_bias);
                lb.arg(&mut dt);
                lb.arg(&mut decay);
                lb.arg(&mut trap);
                lb.arg(&l_i);
                lb.arg(&h_i);
                lb.arg(&dip_i);
                lb.arg(&di_i);
                lb.arg(&ds_i);
                let cfg = LaunchConfig {
                    grid_dim: (l as u32, 1, 1),
                    block_dim: (nh as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // dt_mean
            let mut dt_mean = stream.alloc_zeros::<f32>(l)?;
            {
                let l_i = l as i32;
                let h_i = nh as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.compute_dt_mean);
                lb.arg(&dt);
                lb.arg(&mut dt_mean);
                lb.arg(&l_i);
                lb.arg(&h_i);
                let cfg = LaunchConfig {
                    grid_dim: (l as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // phase (sequential over t)
            let n_angles = layer.num_rope_angles;
            let mut phase = stream.alloc_zeros::<f32>(l * n_angles)?;
            {
                let l_i = l as i32;
                let na_i = n_angles as i32;
                let dip_i = dip as i32;
                let di_i = di as i32;
                let ds_i = ds as i32;
                let h_i = nh as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.compute_phase);
                lb.arg(&proj);
                lb.arg(&dt_mean);
                lb.arg(&mut phase);
                lb.arg(&l_i);
                lb.arg(&na_i);
                lb.arg(&dip_i);
                lb.arg(&di_i);
                lb.arg(&ds_i);
                lb.arg(&h_i);
                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (n_angles.max(1) as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // extract bp, cp
            let mut bp = stream.alloc_zeros::<f32>(l * ds)?;
            let mut cp = stream.alloc_zeros::<f32>(l * ds)?;
            {
                let l_i = l as i32;
                let ds_i = ds as i32;
                let di_i = di as i32;
                let dip_i = dip as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.extract_bp_cp);
                lb.arg(&proj);
                lb.arg(&mut bp);
                lb.arg(&mut cp);
                lb.arg(&l_i);
                lb.arg(&ds_i);
                lb.arg(&di_i);
                lb.arg(&dip_i);
                let cfg = LaunchConfig {
                    grid_dim: (l as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // layer-norm bp, cp
            self.launch_layer_norm(&stream, &mut bp, &layer.b_norm_w, &layer.b_norm_b, l, ds)?;
            self.launch_layer_norm(&stream, &mut cp, &layer.c_norm_w, &layer.c_norm_b, l, ds)?;

            // apply RoPE to bp, cp
            self.launch_apply_rope(&stream, &mut bp, &phase, l, ds, n_angles)?;
            self.launch_apply_rope(&stream, &mut cp, &phase, l, ds, n_angles)?;

            // z_silu
            let mut z_silu = stream.alloc_zeros::<f32>(l * di)?;
            {
                let l_i = l as i32;
                let di_i = di as i32;
                let dip_i = dip as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.compute_z_silu);
                lb.arg(&proj);
                lb.arg(&mut z_silu);
                lb.arg(&l_i);
                lb.arg(&di_i);
                lb.arg(&dip_i);
                let cfg = LaunchConfig {
                    grid_dim: ((di as u32 + 31) / 32, l as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }

            // SSM scan
            let mut y_inner = stream.alloc_zeros::<f32>(l * nh * hd)?;
            {
                let l_i = l as i32;
                let h_i = nh as i32;
                let hd_i = hd as i32;
                let ds_i = ds as i32;
                let di_i = di as i32;
                let dip_i = dip as i32;
                let mut lb = stream.launch_builder(&self.ptx.k.ssm_scan_sequential);
                lb.arg(&proj);
                lb.arg(&bp);
                lb.arg(&cp);
                lb.arg(&dt);
                lb.arg(&decay);
                lb.arg(&trap);
                lb.arg(&layer.d_param);
                lb.arg(&z_silu);
                lb.arg(&mut y_inner);
                lb.arg(&l_i);
                lb.arg(&h_i);
                lb.arg(&hd_i);
                lb.arg(&ds_i);
                lb.arg(&di_i);
                lb.arg(&dip_i);
                let smem_bytes = ((hd * ds) + hd) as u32 * 4;
                let cfg = LaunchConfig {
                    grid_dim: (nh as u32, 1, 1),
                    block_dim: ((hd * ds) as u32, 1, 1),
                    shared_mem_bytes: smem_bytes,
                };
                unsafe { lb.launch(cfg)? };
            }

            // out_proj: y_out = y_inner @ out_proj_w.T → (L, d)
            // y_inner shape (L, H, hd) is contiguous = (L, di).
            let mut y_out = stream.alloc_zeros::<f32>(l * d)?;
            self.launch_matmul_t(&stream, &y_inner, &layer.out_proj_w, &mut y_out, l, d, di)?;

            // residual: x += scale * y_out
            {
                let n_i = (l * d) as i32;
                let scale_f = layer.scale;
                let mut lb = stream.launch_builder(&self.ptx.k.residual_add);
                lb.arg(&mut x);
                lb.arg(&y_out);
                lb.arg(&scale_f);
                lb.arg(&n_i);
                let cfg = LaunchConfig {
                    grid_dim: (((l * d) as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { lb.launch(cfg)? };
            }
        }

        // 4. Final norm
        self.launch_layer_norm(&stream, &mut x, &self.final_norm_w, &self.final_norm_b, l, d)?;

        // 5. LM head: logits = x @ embed_w.T  → (L, vocab)
        let mut logits = stream.alloc_zeros::<f32>(l * self.vocab_size)?;
        self.launch_matmul_t(&stream, &x, &self.embed_w, &mut logits, l, self.vocab_size, d)?;

        stream.synchronize()?;
        let host = stream.memcpy_dtov(&logits)?;
        Ok(host)
    }

    // ------ launch helpers -------------------------------------------------

    fn launch_embed_gather(
        &self,
        stream: &Arc<CudaStream>,
        tokens: &CudaSlice<u32>,
        x: &mut CudaSlice<f32>,
        l: usize,
    ) -> Result<(), Box<dyn Error>> {
        let l_i = l as i32;
        let d_i = self.d_model as i32;
        let v_i = self.vocab_size as i32;
        let mut lb = stream.launch_builder(&self.ptx.k.embed_gather);
        lb.arg(tokens);
        lb.arg(&self.embed_w);
        lb.arg(x);
        lb.arg(&l_i);
        lb.arg(&d_i);
        lb.arg(&v_i);
        let cfg = LaunchConfig {
            grid_dim: (l as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    fn launch_layer_norm(
        &self,
        stream: &Arc<CudaStream>,
        x: &mut CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        l: usize,
        d: usize,
    ) -> Result<(), Box<dyn Error>> {
        let l_i = l as i32;
        let d_i = d as i32;
        let mut lb = stream.launch_builder(&self.ptx.k.layer_norm);
        lb.arg(x);
        lb.arg(w);
        lb.arg(b);
        lb.arg(&l_i);
        lb.arg(&d_i);
        let cfg = LaunchConfig {
            grid_dim: (l as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    fn launch_matmul_t(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), Box<dyn Error>> {
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let mut lb = stream.launch_builder(&self.ptx.k.matmul_t);
        lb.arg(a);
        lb.arg(b);
        lb.arg(c);
        lb.arg(&m_i);
        lb.arg(&n_i);
        lb.arg(&k_i);
        let cfg = LaunchConfig {
            grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg)? };
        Ok(())
    }

    fn launch_apply_rope(
        &self,
        stream: &Arc<CudaStream>,
        v: &mut CudaSlice<f32>,
        phase: &CudaSlice<f32>,
        l: usize,
        ds: usize,
        n_angles: usize,
    ) -> Result<(), Box<dyn Error>> {
        let l_i = l as i32;
        let ds_i = ds as i32;
        let na_i = n_angles as i32;
        let mut lb = stream.launch_builder(&self.ptx.k.apply_rope);
        lb.arg(v);
        lb.arg(phase);
        lb.arg(&l_i);
        lb.arg(&ds_i);
        lb.arg(&na_i);
        let cfg = LaunchConfig {
            grid_dim: (l as u32, 1, 1),
            block_dim: (n_angles.max(1) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { lb.launch(cfg)? };
        Ok(())
    }
}

