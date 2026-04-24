//! GPU-resident model — pre-allocated buffers, minimal dispatch overhead.
//!
//! All weight buffers AND intermediate buffers are allocated at model load.
//! Each forward: write input → submit dispatches → read output.
//! No buffer allocation during inference.

use crate::model::{Mamba3Model, LayerWeights};
use crate::scan::GpuContext;
use wgpu::util::DeviceExt;

/// Pre-allocated GPU buffers for one layer
struct LayerBuffers {
    in_proj_w: wgpu::Buffer,   // weight (persistent)
    out_proj_w: wgpu::Buffer,  // weight (persistent)
    proj_buf: wgpu::Buffer,    // intermediate: in_proj output (l * d_in_proj)
    y_inner_buf: wgpu::Buffer, // intermediate: SSM output (l * d_inner)
    y_out_buf: wgpu::Buffer,   // intermediate: out_proj output (l * d_model)
}

pub struct GpuModel {
    pub cpu_model: Mamba3Model,
    pub gpu: GpuContext,
    // Pre-allocated buffers
    embed_w_buf: wgpu::Buffer,
    layer_bufs: Vec<LayerBuffers>,
    // Reusable input/output buffers
    x_buf: wgpu::Buffer,          // current hidden state (l * d_model)
    x_normed_buf: wgpu::Buffer,   // normed input (l * d_model)
    logits_buf: wgpu::Buffer,     // output logits (l * vocab)
    staging_buf: wgpu::Buffer,    // readback staging
    // Fixed params
    max_seq_len: usize,
}

impl GpuModel {
    pub fn new(model: Mamba3Model, gpu: GpuContext, max_seq_len: usize) -> Self {
        let d = model.d_model;
        let di = model.d_inner;
        let v = model.vocab_size;

        let buf = |label: &str, size: usize| -> wgpu::Buffer {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (size * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let weight_buf = |label: &str, data: &[f32]| -> wgpu::Buffer {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        // Weight buffers (persistent)
        let embed_w_buf = weight_buf("embed_w", &model.embed_w);

        let mut layer_bufs = Vec::new();
        for (i, layer) in model.layers.iter().enumerate() {
            let dip = layer.d_in_proj;
            layer_bufs.push(LayerBuffers {
                in_proj_w: weight_buf(&format!("in_proj_{}", i), &layer.in_proj_w),
                out_proj_w: weight_buf(&format!("out_proj_{}", i), &layer.out_proj_w),
                proj_buf: buf(&format!("proj_{}", i), max_seq_len * dip),
                y_inner_buf: buf(&format!("y_inner_{}", i), max_seq_len * di),
                y_out_buf: buf(&format!("y_out_{}", i), max_seq_len * d),
            });
        }

        // Reusable buffers
        let x_buf = buf("x", max_seq_len * d);
        let x_normed_buf = buf("x_normed", max_seq_len * d);
        let logits_buf = buf("logits", max_seq_len * v);

        let staging_size = max_seq_len * v;  // largest readback = logits
        let staging_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (staging_size * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let total_gpu_bytes = model.embed_w.len() * 4
            + model.layers.iter().map(|l| (l.in_proj_w.len() + l.out_proj_w.len()) * 4).sum::<usize>()
            + layer_bufs.len() * max_seq_len * (d + di + d) * 4  // intermediates
            + max_seq_len * (d + d + v) * 4;  // x, x_normed, logits
        eprintln!("  GPU memory: {:.2} MB ({} weight + {} intermediate)",
            total_gpu_bytes as f64 / 1024.0 / 1024.0,
            model.layers.len(), layer_bufs.len());

        Self { cpu_model: model, gpu, embed_w_buf, layer_bufs, x_buf, x_normed_buf, logits_buf, staging_buf, max_seq_len }
    }

    /// Forward: matmuls on GPU, SSM scan on CPU, zero buffer allocation.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let l = tokens.len();
        let d = self.cpu_model.d_model;
        let v = self.cpu_model.vocab_size;
        let di = self.cpu_model.d_inner;

        // 1. Embedding + norm (CPU — tiny, not worth GPU)
        let mut x = vec![0.0f32; l * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok < v {
                for i in 0..d { x[t * d + i] = self.cpu_model.embed_w[tok * d + i]; }
            }
        }
        crate::model::layer_norm_pub(&mut x, &self.cpu_model.embed_norm_w, &self.cpu_model.embed_norm_b, l, d);

        // 2. Layers
        for (li, layer) in self.cpu_model.layers.iter().enumerate() {
            let lb = &self.layer_bufs[li];

            // Pre-norm (CPU)
            let mut x_normed = x.clone();
            crate::model::layer_norm_pub(&mut x_normed, &layer.layer_norm_w, &layer.layer_norm_b, l, d);

            // GPU in_proj: x_normed × in_proj_w^T → proj
            let dip = layer.d_in_proj;
            let proj = self.gpu_matmul_prealloc(
                &x_normed, &lb.in_proj_w, &lb.proj_buf,
                l as u32, d as u32, dip as u32
            );

            // CPU SSM scan
            let y_inner = self.cpu_model.mamba3_block_inner_pub(&proj, layer, l);

            // GPU out_proj: y_inner × out_proj_w^T → y_out
            let y_out = self.gpu_matmul_prealloc(
                &y_inner, &lb.out_proj_w, &lb.y_out_buf,
                l as u32, di as u32, d as u32
            );

            for i in 0..l * d { x[i] += layer.scale * y_out[i]; }
        }

        // 3. Final norm (CPU)
        crate::model::layer_norm_pub(&mut x, &self.cpu_model.final_norm_w, &self.cpu_model.final_norm_b, l, d);

        // 4. GPU LM head
        let logits = self.gpu_matmul_prealloc(
            &x, &self.embed_w_buf, &self.logits_buf,
            l as u32, d as u32, v as u32
        );

        logits
    }

    /// Matmul with pre-allocated output buffer — only uploads A, no buffer creation.
    fn gpu_matmul_prealloc(
        &self, a: &[f32], b_buf: &wgpu::Buffer, c_buf: &wgpu::Buffer,
        m: u32, k: u32, n: u32
    ) -> Vec<f32> {
        let output_size = (m * n) as usize;

        // Write A to a temporary upload buffer
        let buf_a = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A_upload"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P { m: u32, k: u32, n: u32, _pad: u32 }
        let buf_params = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&P { m, k, n, _pad: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mm"),
            layout: &self.gpu.matmul_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
            ],
        });

        let tile = 16u32;
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.gpu.matmul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((m + tile - 1) / tile, (n + tile - 1) / tile, 1);
        }

        // Read back from pre-allocated C buffer
        encoder.copy_buffer_to_buffer(c_buf, 0, &self.staging_buf, 0, (output_size * 4) as u64);
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.staging_buf.slice(..((output_size * 4) as u64));
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buf.unmap();

        result
    }
}
