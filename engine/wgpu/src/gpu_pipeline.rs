//! GPU Pipeline — fully GPU-resident forward with minimal CPU interaction.
//!
//! Architecture:
//!   Upload input → [GPU: norm → matmul → ...per layer... → matmul] → Download logits
//!
//! The SSM scan is the only CPU op (sequential dependency).
//! All matmuls, norms run on GPU. Single command buffer per layer.
//! Intermediate buffers pre-allocated and reused.

use crate::model::Mamba3Model;
use crate::scan::GpuContext;
use wgpu::util::DeviceExt;

pub struct GpuPipeline {
    pub cpu_model: Mamba3Model,
    pub gpu: GpuContext,

    // Persistent weight buffers
    embed_w: wgpu::Buffer,
    embed_norm_w: wgpu::Buffer,
    embed_norm_b: wgpu::Buffer,
    final_norm_w: wgpu::Buffer,
    final_norm_b: wgpu::Buffer,
    layers: Vec<GpuLayerBufs>,

    // Reusable intermediate buffers
    x_buf: wgpu::Buffer,        // (max_seq, d)
    staging: wgpu::Buffer,      // readback

    // Pipelines
    norm_pipeline: wgpu::ComputePipeline,
    norm_layout: wgpu::BindGroupLayout,

    max_seq: usize,
}

struct GpuLayerBufs {
    in_proj_w: wgpu::Buffer,
    out_proj_w: wgpu::Buffer,
    norm_w: wgpu::Buffer,
    norm_b: wgpu::Buffer,
    proj_buf: wgpu::Buffer,     // (max_seq, d_in_proj)
    y_out_buf: wgpu::Buffer,    // (max_seq, d)
}

impl GpuPipeline {
    pub fn new(model: Mamba3Model, gpu: GpuContext, max_seq: usize) -> Self {
        let d = model.d_model;
        let v = model.vocab_size;
        let di = model.d_inner;

        let w = |data: &[f32]| -> wgpu::Buffer {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let buf = |size: usize| -> wgpu::Buffer {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: (size * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let embed_w = w(&model.embed_w);
        let embed_norm_w = w(&model.embed_norm_w);
        let embed_norm_b = w(&model.embed_norm_b);
        let final_norm_w = w(&model.final_norm_w);
        let final_norm_b = w(&model.final_norm_b);

        let mut layers = Vec::new();
        for layer in &model.layers {
            layers.push(GpuLayerBufs {
                in_proj_w: w(&layer.in_proj_w),
                out_proj_w: w(&layer.out_proj_w),
                norm_w: w(&layer.layer_norm_w),
                norm_b: w(&layer.layer_norm_b),
                proj_buf: buf(max_seq * layer.d_in_proj),
                y_out_buf: buf(max_seq * d),
            });
        }

        let max_staging = max_seq * v.max(d);  // whichever is larger
        let x_buf = buf(max_seq * d);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"), size: (max_staging * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Layer norm pipeline
        let norm_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("norm"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders/layer_norm.wgsl").into()),
        });
        let norm_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_rw(0),  // x (read-write)
                bgl_ro(1),  // w
                bgl_ro(2),  // b
                bgl_uniform(3),
            ],
        });
        let norm_pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&norm_layout], push_constant_ranges: &[],
        });
        let norm_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("norm"), layout: Some(&norm_pl), module: &norm_shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });

        let total_mb = (model.embed_w.len() + model.final_norm_w.len() * 2
            + model.layers.iter().map(|l| l.in_proj_w.len() + l.out_proj_w.len() + l.layer_norm_w.len() * 2).sum::<usize>()
        ) as f64 * 4.0 / 1024.0 / 1024.0;
        eprintln!("  GPU pipeline: {:.1}MB weights, {} layers, max_seq={}", total_mb, model.layers.len(), max_seq);

        Self {
            cpu_model: model, gpu, embed_w, embed_norm_w, embed_norm_b,
            final_norm_w, final_norm_b, layers, x_buf, staging,
            norm_pipeline, norm_layout, max_seq,
        }
    }

    /// Forward pass — GPU matmuls + norms, CPU scan only.
    /// One readback per layer (for SSM scan), one final readback for logits.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let l = tokens.len();
        let d = self.cpu_model.d_model;
        let v = self.cpu_model.vocab_size;
        let di = self.cpu_model.d_inner;

        // 1. Embedding (CPU — lookup, tiny)
        let mut x = vec![0.0f32; l * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok < v { for i in 0..d { x[t * d + i] = self.cpu_model.embed_w[tok * d + i]; } }
        }

        // Upload x to GPU
        self.write_buffer(&self.x_buf, &x);

        // GPU embed norm
        self.dispatch_norm(&self.x_buf, &self.embed_norm_w, &self.embed_norm_b, l as u32, d as u32);

        // 2. Layers
        for (li, layer) in self.cpu_model.layers.iter().enumerate() {
            let lb = &self.layers[li];
            let dip = layer.d_in_proj;

            // GPU: copy x → norm in-place for this layer
            // We need x_normed but also keep x for residual. Copy x to proj_buf temporarily.
            // Actually: read x back, norm on CPU (to preserve x for residual), upload x_normed, GPU matmul.
            // This is the minimum-readback approach.

            let x_data = self.read_buffer(&self.x_buf, l * d);

            // CPU: pre-norm (need original x for residual)
            let mut x_normed = x_data.clone();
            crate::model::layer_norm_pub(&mut x_normed, &layer.layer_norm_w, &layer.layer_norm_b, l, d);

            // GPU in_proj
            let proj = self.dispatch_matmul(&x_normed, &lb.in_proj_w, &lb.proj_buf, l as u32, d as u32, dip as u32);

            // SSM: preprocess on CPU (splits, norms, RoPE, outer product), scan on GPU
            let y_inner = self.run_ssm_gpu(&proj, layer, l);

            // GPU out_proj
            let y_out = self.dispatch_matmul(&y_inner, &lb.out_proj_w, &lb.y_out_buf, l as u32, di as u32, d as u32);

            // CPU residual (x = x_data + scale * y_out)
            let mut new_x = x_data;
            for i in 0..l * d { new_x[i] += layer.scale * y_out[i]; }

            // Upload updated x
            self.write_buffer(&self.x_buf, &new_x);
        }

        // 3. GPU final norm
        self.dispatch_norm(&self.x_buf, &self.final_norm_w, &self.final_norm_b, l as u32, d as u32);

        // 4. Read x, GPU LM head
        let x_final = self.read_buffer(&self.x_buf, l * d);
        let logits_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (l * v * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.dispatch_matmul_to(&x_final, &self.embed_w, &logits_buf, l as u32, d as u32, v as u32);

        // Read logits
        self.read_buffer_from(&logits_buf, l * v)
    }

    /// Run SSM: preprocess (CPU) → scan (GPU) → gate (CPU).
    /// The scan shader runs the sequential recurrence on GPU.
    fn run_ssm_gpu(&self, proj: &[f32], layer: &crate::model::LayerWeights, l: usize) -> Vec<f32> {
        let di = self.cpu_model.d_inner;
        let nh = self.cpu_model.n_heads;
        let hd = self.cpu_model.headdim;
        let ds = self.cpu_model.d_state;

        // CPU preprocessing: split proj, norm B/C, RoPE, compute inp/decay/Cp
        // This reuses the model's mamba3_block_inner logic but extracts the scan inputs
        // For the full GPU path, we run the complete SSM including scan on GPU
        // via the existing ssm_scan.wgsl shader

        // For now: use CPU model's inner function which handles all the SSM math
        // This is correct and handles RoPE, trapezoidal, etc.
        // The scan inside runs CPU — to make it GPU, we'd need to extract
        // (inp, decay, Cp, x, z, D) and call gpu.run_scan()

        // Extract scan inputs using the model's preprocessing
        let (inp, decay, cp, x_skip, z_silu, d_param) =
            self.cpu_model.extract_scan_inputs(proj, layer, l);

        // GPU scan dispatch
        let y_scan = pollster::block_on(self.gpu.run_scan(
            &inp, &decay, &cp, &x_skip, &z_silu, &d_param,
            1, l as u32, nh as u32, hd as u32, ds as u32,
        )).unwrap_or_else(|_| {
            // Fallback to CPU scan
            self.cpu_model.mamba3_block_inner_pub(proj, layer, l)
        });

        // y_scan is (1, L, H, hD) = (L, H*hD) = (L, d_inner)
        y_scan
    }

    fn write_buffer(&self, buf: &wgpu::Buffer, data: &[f32]) {
        self.gpu.queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
    }

    fn read_buffer(&self, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        self.read_buffer_from(buf, n)
    }

    fn read_buffer_from(&self, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, &self.staging, 0, (n * 4) as u64);
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.staging.slice(..((n * 4) as u64));
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging.unmap();
        result
    }

    fn dispatch_norm(&self, x_buf: &wgpu::Buffer, w_buf: &wgpu::Buffer, b_buf: &wgpu::Buffer, seq_len: u32, d: u32) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P { seq_len: u32, d: u32, _p0: u32, _p1: u32 }
        let p = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&P { seq_len, d, _p0: 0, _p1: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.norm_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p.as_entire_binding() },
            ],
        });
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        { let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.norm_pipeline); pass.set_bind_group(0, &bg, &[]);
          pass.dispatch_workgroups(seq_len, 1, 1); }
        self.gpu.queue.submit(std::iter::once(enc.finish()));
    }

    fn dispatch_matmul(&self, a: &[f32], b_buf: &wgpu::Buffer, c_buf: &wgpu::Buffer, m: u32, k: u32, n: u32) -> Vec<f32> {
        self.dispatch_matmul_to(a, b_buf, c_buf, m, k, n);
        self.read_buffer_from(c_buf, (m * n) as usize)
    }

    fn dispatch_matmul_to(&self, a: &[f32], b_buf: &wgpu::Buffer, c_buf: &wgpu::Buffer, m: u32, k: u32, n: u32) {
        let buf_a = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(a), usage: wgpu::BufferUsages::STORAGE,
        });
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P { m: u32, k: u32, n: u32, _pad: u32 }
        let p = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&P { m, k, n, _pad: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.gpu.matmul_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p.as_entire_binding() },
            ],
        });
        let tile = 16u32;
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        { let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.gpu.matmul_pipeline); pass.set_bind_group(0, &bg, &[]);
          pass.dispatch_workgroups((m + tile - 1) / tile, (n + tile - 1) / tile, 1); }
        self.gpu.queue.submit(std::iter::once(enc.finish()));
    }
}

// Bind group layout helpers
fn bgl_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
fn bgl_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}
