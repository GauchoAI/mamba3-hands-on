//! GPU-resident model — all weights on GPU, single command buffer per forward.
//! Eliminates per-matmul upload/download overhead.
//!
//! Flow:
//!   1. Load model → upload all weights to GPU buffers (once)
//!   2. Forward: upload input (tiny) → dispatch all layers → download logits
//!   3. Total: 1 upload + 1 download per forward

use crate::model::Mamba3Model;
use crate::scan::GpuContext;

/// Model with weights resident on GPU
pub struct GpuModel {
    pub cpu_model: Mamba3Model,  // keep CPU copy for SSM scan
    pub gpu: GpuContext,
    // Per-layer GPU buffers (uploaded once)
    layer_in_proj_bufs: Vec<wgpu::Buffer>,
    layer_out_proj_bufs: Vec<wgpu::Buffer>,
    embed_buf: wgpu::Buffer,
}

impl GpuModel {
    pub fn new(model: Mamba3Model, gpu: GpuContext) -> Self {
        // Upload all weight matrices to GPU
        let embed_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("embed_w"),
            contents: bytemuck::cast_slice(&model.embed_w),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let mut layer_in_proj_bufs = Vec::new();
        let mut layer_out_proj_bufs = Vec::new();

        for (i, layer) in model.layers.iter().enumerate() {
            layer_in_proj_bufs.push(gpu.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("in_proj_{}", i)),
                    contents: bytemuck::cast_slice(&layer.in_proj_w),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            ));
            layer_out_proj_bufs.push(gpu.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("out_proj_{}", i)),
                    contents: bytemuck::cast_slice(&layer.out_proj_w),
                    usage: wgpu::BufferUsages::STORAGE,
                }
            ));
        }

        let n_weights: usize = model.embed_w.len()
            + model.layers.iter().map(|l| l.in_proj_w.len() + l.out_proj_w.len()).sum::<usize>();
        eprintln!("  Uploaded {} weight floats ({:.1} MB) to GPU",
            n_weights, n_weights as f64 * 4.0 / 1024.0 / 1024.0);

        Self { cpu_model: model, gpu, layer_in_proj_bufs, layer_out_proj_bufs, embed_buf }
    }

    /// Forward pass — GPU matmuls with persistent buffers, CPU for SSM scan.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let l = tokens.len();
        let d = self.cpu_model.d_model;
        let v = self.cpu_model.vocab_size;

        // 1. Embedding + norm (CPU — small)
        let mut x = vec![0.0f32; l * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok < v {
                for i in 0..d { x[t * d + i] = self.cpu_model.embed_w[tok * d + i]; }
            }
        }
        crate::model::layer_norm_pub(&mut x, &self.cpu_model.embed_norm_w, &self.cpu_model.embed_norm_b, l, d);

        // 2. Layers — GPU matmuls + CPU scan
        for (li, layer) in self.cpu_model.layers.iter().enumerate() {
            let mut x_normed = x.clone();
            crate::model::layer_norm_pub(&mut x_normed, &layer.layer_norm_w, &layer.layer_norm_b, l, d);

            // GPU in_proj using persistent buffer
            let dip = layer.d_in_proj;
            let proj = self.gpu_matmul_persistent(
                &x_normed, &self.layer_in_proj_bufs[li],
                l as u32, d as u32, dip as u32
            );

            // CPU SSM scan (sequential)
            let y_inner = self.cpu_model.mamba3_block_inner_pub(&proj, layer, l);

            // GPU out_proj using persistent buffer
            let di = self.cpu_model.d_inner;
            let y_out = self.gpu_matmul_persistent(
                &y_inner, &self.layer_out_proj_bufs[li],
                l as u32, di as u32, d as u32
            );

            for i in 0..l * d { x[i] += layer.scale * y_out[i]; }
        }

        // 3. Final norm (CPU)
        crate::model::layer_norm_pub(&mut x, &self.cpu_model.final_norm_w, &self.cpu_model.final_norm_b, l, d);

        // 4. GPU LM head using persistent embed buffer
        let logits = self.gpu_matmul_persistent(
            &x, &self.embed_buf, l as u32, d as u32, v as u32
        );

        logits
    }

    /// Matmul using a pre-uploaded B buffer — only uploads A and downloads C.
    fn gpu_matmul_persistent(&self, a: &[f32], b_buf: &wgpu::Buffer, m: u32, k: u32, n: u32) -> Vec<f32> {
        let output_size = (m * n) as usize;

        // Upload A (input — changes each call)
        let buf_a = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A"),
            contents: bytemuck::cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // C output buffer
        let buf_c = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C"),
            size: (output_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Params
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P { m: u32, k: u32, n: u32, _pad: u32 }
        let buf_params = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&P { m, k, n, _pad: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Bind group — B uses the persistent buffer
        let bind_group = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul"),
            layout: &self.gpu.matmul_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
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

        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (output_size * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buf_c, 0, &staging, 0, (output_size * 4) as u64);
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        result
    }
}

use wgpu::util::DeviceExt;
