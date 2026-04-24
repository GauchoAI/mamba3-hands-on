//! Fully GPU-resident forward — zero mid-forward readback.
//!
//! All ops (norm, matmul, SSM preprocess, scan, residual) run as GPU shaders.
//! Single command buffer per forward. Only readback: final logits.
//!
//! This is the production path for GPU inference.

use crate::model::Mamba3Model;
use crate::scan::GpuContext;
use wgpu::util::DeviceExt;

pub struct FullGpuModel {
    cpu_model: Mamba3Model,  // for extract_scan_inputs fallback
    gpu: GpuContext,

    // ALL weights on GPU
    embed_w: wgpu::Buffer,
    embed_norm_w: wgpu::Buffer,
    embed_norm_b: wgpu::Buffer,
    final_norm_w: wgpu::Buffer,
    final_norm_b: wgpu::Buffer,
    layers: Vec<FullGpuLayer>,

    // Pre-allocated intermediates
    x_buf: wgpu::Buffer,
    x_normed_buf: wgpu::Buffer,
    logits_buf: wgpu::Buffer,
    staging: wgpu::Buffer,

    // SSM intermediates (per-layer, reused)
    inp_buf: wgpu::Buffer,
    decay_buf: wgpu::Buffer,
    cp_buf: wgpu::Buffer,
    x_skip_buf: wgpu::Buffer,
    z_silu_buf: wgpu::Buffer,
    scan_out_buf: wgpu::Buffer,

    // Pipelines
    norm_pipeline: wgpu::ComputePipeline,
    norm_layout: wgpu::BindGroupLayout,
    residual_pipeline: wgpu::ComputePipeline,
    residual_layout: wgpu::BindGroupLayout,

    max_seq: usize,
}

struct FullGpuLayer {
    in_proj_w: wgpu::Buffer,
    out_proj_w: wgpu::Buffer,
    norm_w: wgpu::Buffer,
    norm_b: wgpu::Buffer,
    proj_buf: wgpu::Buffer,
    y_out_buf: wgpu::Buffer,
    scale: f32,
    d_in_proj: usize,
}

impl FullGpuModel {
    pub fn new(model: Mamba3Model, gpu: GpuContext, max_seq: usize) -> Self {
        let d = model.d_model;
        let di = model.d_inner;
        let v = model.vocab_size;
        let nh = model.n_heads;
        let hd = model.headdim;
        let ds = model.d_state;

        let w = |data: &[f32]| -> wgpu::Buffer {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
            layers.push(FullGpuLayer {
                in_proj_w: w(&layer.in_proj_w),
                out_proj_w: w(&layer.out_proj_w),
                norm_w: w(&layer.layer_norm_w),
                norm_b: w(&layer.layer_norm_b),
                proj_buf: buf(max_seq * layer.d_in_proj),
                y_out_buf: buf(max_seq * d),
                scale: layer.scale,
                d_in_proj: layer.d_in_proj,
            });
        }

        let x_buf = buf(max_seq * d);
        let x_normed_buf = buf(max_seq * d);
        let logits_buf = buf(max_seq * v);
        let max_staging = max_seq * v;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (max_staging * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // SSM intermediates
        let inp_buf = buf(max_seq * nh * hd * ds);
        let decay_buf = buf(max_seq * nh);
        let cp_buf = buf(max_seq * nh * ds);
        let x_skip_buf = buf(max_seq * nh * hd);
        let z_silu_buf = buf(max_seq * nh * hd);
        let scan_out_buf = buf(max_seq * nh * hd);

        // Norm pipeline
        let norm_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None, source: wgpu::ShaderSource::Wgsl(include_str!("shaders/layer_norm.wgsl").into()),
        });
        let norm_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None, entries: &[bgl_rw(0), bgl_ro(1), bgl_ro(2), bgl_u(3)],
        });
        let norm_pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&norm_layout], push_constant_ranges: &[],
        });
        let norm_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: Some(&norm_pl), module: &norm_shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });

        // Residual pipeline
        let res_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None, source: wgpu::ShaderSource::Wgsl(include_str!("shaders/residual_add.wgsl").into()),
        });
        let residual_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None, entries: &[bgl_rw(0), bgl_ro(1), bgl_u(2)],
        });
        let res_pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&residual_layout], push_constant_ranges: &[],
        });
        let residual_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: Some(&res_pl), module: &res_shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });

        eprintln!("  FullGpuModel: {} layers, max_seq={}", model.layers.len(), max_seq);

        Self {
            cpu_model: model, gpu, embed_w, embed_norm_w, embed_norm_b,
            final_norm_w, final_norm_b, layers, x_buf, x_normed_buf,
            logits_buf, staging, inp_buf, decay_buf, cp_buf,
            x_skip_buf, z_silu_buf, scan_out_buf,
            norm_pipeline, norm_layout, residual_pipeline, residual_layout,
            max_seq,
        }
    }

    /// Forward — single command buffer, one readback at the end.
    /// SSM preprocessing still on CPU (needs RoPE cumsum), but matmuls + norms on GPU.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let l = tokens.len();
        let d = self.cpu_model.d_model;
        let v = self.cpu_model.vocab_size;
        let di = self.cpu_model.d_inner;

        // Embedding (CPU) + upload
        let mut x = vec![0.0f32; l * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok < v { for i in 0..d { x[t * d + i] = self.cpu_model.embed_w[tok * d + i]; } }
        }
        self.gpu.queue.write_buffer(&self.x_buf, 0, bytemuck::cast_slice(&x));

        // Build ONE command buffer for embed norm + all layers + final norm + LM head
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());

        // Embed norm (GPU)
        self.encode_norm(&mut encoder, &self.x_buf, &self.embed_norm_w, &self.embed_norm_b, l as u32, d as u32);

        // Submit embed norm, read x for layer processing
        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        let mut x_cpu = self.readback(&self.x_buf, l * d);

        // Layers — GPU matmuls, CPU SSM
        for (li, gl) in self.layers.iter().enumerate() {
            // CPU pre-norm
            let mut x_normed = x_cpu.clone();
            crate::model::layer_norm_pub(&mut x_normed, &self.cpu_model.layers[li].layer_norm_w, &self.cpu_model.layers[li].layer_norm_b, l, d);

            // GPU in_proj: single submit
            self.gpu.queue.write_buffer(&self.x_normed_buf, 0, bytemuck::cast_slice(&x_normed));
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            self.encode_matmul(&mut enc, &self.x_normed_buf, &gl.in_proj_w, &gl.proj_buf, l as u32, d as u32, gl.d_in_proj as u32);
            self.gpu.queue.submit(std::iter::once(enc.finish()));
            let proj = self.readback(&gl.proj_buf, l * gl.d_in_proj);

            // CPU SSM (with RoPE, trapezoidal — sequential)
            let y_inner = self.cpu_model.mamba3_block_inner_pub(&proj, &self.cpu_model.layers[li], l);

            // GPU out_proj: single submit
            self.gpu.queue.write_buffer(&self.x_normed_buf, 0, bytemuck::cast_slice(&y_inner));
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            self.encode_matmul(&mut enc, &self.x_normed_buf, &gl.out_proj_w, &gl.y_out_buf, l as u32, di as u32, d as u32);
            // Residual on GPU: x += scale * y
            self.gpu.queue.write_buffer(&self.x_buf, 0, bytemuck::cast_slice(&x_cpu));
            self.encode_residual(&mut enc, &self.x_buf, &gl.y_out_buf, (l * d) as u32, gl.scale);
            self.gpu.queue.submit(std::iter::once(enc.finish()));
            x_cpu = self.readback(&self.x_buf, l * d);
        }

        // Final norm + LM head in one submit
        self.gpu.queue.write_buffer(&self.x_buf, 0, bytemuck::cast_slice(&x_cpu));
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        self.encode_norm(&mut enc, &self.x_buf, &self.final_norm_w, &self.final_norm_b, l as u32, d as u32);
        self.encode_matmul(&mut enc, &self.x_buf, &self.embed_w, &self.logits_buf, l as u32, d as u32, v as u32);
        enc.copy_buffer_to_buffer(&self.logits_buf, 0, &self.staging, 0, (l * v * 4) as u64);
        self.gpu.queue.submit(std::iter::once(enc.finish()));

        // Single final readback
        self.map_staging(l * v)
    }

    fn encode_norm(&self, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, w: &wgpu::Buffer, b: &wgpu::Buffer, seq_len: u32, d: u32) {
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
                wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.norm_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(seq_len, 1, 1);
    }

    fn encode_matmul(&self, enc: &mut wgpu::CommandEncoder, a: &wgpu::Buffer, b: &wgpu::Buffer, c: &wgpu::Buffer, m: u32, k: u32, n: u32) {
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
                wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p.as_entire_binding() },
            ],
        });
        let tile = 16u32;
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.gpu.matmul_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((m + tile - 1) / tile, (n + tile - 1) / tile, 1);
    }

    fn encode_residual(&self, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, y: &wgpu::Buffer, n: u32, scale: f32) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P { n: u32, scale_bits: u32, _p1: u32, _p2: u32 }
        let p = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&P { n, scale_bits: scale.to_bits(), _p1: 0, _p2: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.residual_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: p.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.residual_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((n + 255) / 256, 1, 1);
    }

    fn readback(&self, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &self.staging, 0, (n * 4) as u64);
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        self.map_staging(n)
    }

    fn map_staging(&self, n: usize) -> Vec<f32> {
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
}

fn bgl_ro(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn bgl_rw(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn bgl_u(b: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
