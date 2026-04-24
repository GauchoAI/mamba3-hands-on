//! SSM Scan GPU implementation via wgpu compute shaders.
//!
//! Runs the sequential SSM scan on any GPU backend (Vulkan, Metal, DX12).
//! Uses explicit fp32 arithmetic — no FMA, no TF32, no hidden precision loss.

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    b: u32,
    l: u32,
    h: u32,
    hd: u32,
    ds: u32,
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    // Matmul pipeline
    matmul_pipeline: wgpu::ComputePipeline,
    matmul_layout: wgpu::BindGroupLayout,
}

impl GpuContext {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or("No GPU adapter found")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mamba3"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            }, None)
            .await?;

        eprintln!(
            "wgpu: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        // Load shader
        let shader_src = include_str!("shaders/ssm_scan.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssm_scan"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Bind group layout: 7 storage buffers + 1 uniform
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssm_scan_layout"),
            entries: &[
                // inp, decay, C, x, z_silu (read-only storage)
                bgl_entry(0, false),
                bgl_entry(1, false),
                bgl_entry(2, false),
                bgl_entry(3, false),
                bgl_entry(4, false),
                bgl_entry(5, false),
                // y (read-write storage)
                bgl_entry(6, true),
                // params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssm_scan_pipeline"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ssm_scan"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Matmul pipeline
        let matmul_shader_src = include_str!("shaders/matmul.wgsl");
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(matmul_shader_src.into()),
        });

        let matmul_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_layout"),
            entries: &[
                bgl_entry(0, false),  // A
                bgl_entry(1, false),  // B
                bgl_entry(2, true),   // C
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let matmul_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul_pipeline"),
            bind_group_layouts: &[&matmul_layout],
            push_constant_ranges: &[],
        });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul"),
            layout: Some(&matmul_pipeline_layout),
            module: &matmul_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self { device, queue, pipeline, bind_group_layout, matmul_pipeline, matmul_layout })
    }

    pub async fn run_scan(
        &self,
        inp: &[f32], decay: &[f32], c: &[f32],
        x: &[f32], z_silu: &[f32], d: &[f32],
        b: u32, l: u32, h: u32, hd: u32, ds: u32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let output_size = (b * l * h * hd) as usize;

        // Create GPU buffers
        let buf_inp = self.create_storage_buffer("inp", bytemuck::cast_slice(inp));
        let buf_decay = self.create_storage_buffer("decay", bytemuck::cast_slice(decay));
        let buf_c = self.create_storage_buffer("C", bytemuck::cast_slice(c));
        let buf_x = self.create_storage_buffer("x", bytemuck::cast_slice(x));
        let buf_z = self.create_storage_buffer("z_silu", bytemuck::cast_slice(z_silu));
        let buf_d = self.create_storage_buffer("D", bytemuck::cast_slice(d));

        let buf_y = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"),
            size: (output_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Params { b, l, h, hd, ds };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssm_scan_bind"),
            layout: &self.bind_group_layout,
            entries: &[
                bg_entry(0, &buf_inp),
                bg_entry(1, &buf_decay),
                bg_entry(2, &buf_c),
                bg_entry(3, &buf_x),
                bg_entry(4, &buf_z),
                bg_entry(5, &buf_d),
                bg_entry(6, &buf_y),
                bg_entry(7, &buf_params),
            ],
        });

        // Dispatch
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(b * h, 1, 1);
        }

        // Read back
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (output_size * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buf_y, 0, &staging, 0, (output_size * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// GPU matrix multiply: C = A × B^T. A is (m,k), B is (n,k), C is (m,n).
    pub async fn run_matmul(
        &self, a: &[f32], b: &[f32], m: u32, k: u32, n: u32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let output_size = (m * n) as usize;

        let buf_a = self.create_storage_buffer("A", bytemuck::cast_slice(a));
        let buf_b = self.create_storage_buffer("B", bytemuck::cast_slice(b));
        let buf_c = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C"),
            size: (output_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulParams { m: u32, k: u32, n: u32, _pad: u32 }
        let params = MatmulParams { m, k, n, _pad: 0 };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matmul_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind"),
            layout: &self.matmul_layout,
            entries: &[
                bg_entry(0, &buf_a),
                bg_entry(1, &buf_b),
                bg_entry(2, &buf_c),
                bg_entry(3, &buf_params),
            ],
        });

        let tile = 16u32;
        let wg_x = (m + tile - 1) / tile;
        let wg_y = (n + tile - 1) / tile;

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (output_size * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buf_c, 0, &staging, 0, (output_size * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    fn create_storage_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE,
        })
    }
}

// Helpers
use wgpu::util::DeviceExt;

fn bgl_entry(binding: u32, read_write: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: !read_write },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
