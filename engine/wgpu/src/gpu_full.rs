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

    // Per-layer weight buffers for SSM prep
    layer_dt_bias: Vec<wgpu::Buffer>,
    layer_b_norm_w: Vec<wgpu::Buffer>,
    layer_b_norm_b: Vec<wgpu::Buffer>,
    layer_c_norm_w: Vec<wgpu::Buffer>,
    layer_c_norm_b: Vec<wgpu::Buffer>,
    layer_d_param: Vec<wgpu::Buffer>,

    // Pipelines
    norm_pipeline: wgpu::ComputePipeline,
    norm_layout: wgpu::BindGroupLayout,
    residual_pipeline: wgpu::ComputePipeline,
    residual_layout: wgpu::BindGroupLayout,
    ssm_prep_pipeline: wgpu::ComputePipeline,
    ssm_prep_layout: wgpu::BindGroupLayout,
    fused_pipeline: wgpu::ComputePipeline,
    fused_layout: wgpu::BindGroupLayout,
    fused_scale_layout: wgpu::BindGroupLayout,

    max_seq: usize,

    // Pre-built bind groups (created once at load, reused every forward)
    embed_norm_bg: wgpu::BindGroup,
    final_norm_bg: wgpu::BindGroup,
    layer_norm_bgs: Vec<wgpu::BindGroup>,
    layer_in_proj_bgs: Vec<wgpu::BindGroup>,
    layer_out_proj_bgs: Vec<wgpu::BindGroup>,
    layer_residual_bgs: Vec<wgpu::BindGroup>,
    layer_prep_bgs: Vec<wgpu::BindGroup>,
    layer_scan_bgs: Vec<wgpu::BindGroup>,
    head_matmul_bg: wgpu::BindGroup,
    // Params buffers that need L updated each forward
    all_param_bufs: Vec<wgpu::Buffer>,
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

        // SSM prep pipeline (12 bindings: proj, dt_bias, b_norm_w/b, c_norm_w/b, inp, decay, Cp, x_skip, z_silu, params)
        let prep_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None, source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ssm_prep.wgsl").into()),
        });
        let ssm_prep_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None, entries: &[
                bgl_ro(0), bgl_ro(1), bgl_ro(2), bgl_ro(3), bgl_ro(4), bgl_ro(5),
                bgl_rw(6), bgl_rw(7), bgl_rw(8), bgl_rw(9), bgl_rw(10), bgl_u(11),
            ],
        });
        let prep_pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&ssm_prep_layout], push_constant_ranges: &[],
        });
        let ssm_prep_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: Some(&prep_pl), module: &prep_shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });

        // Per-layer SSM weight buffers
        let mut layer_dt_bias = Vec::new();
        let mut layer_b_norm_w = Vec::new();
        let mut layer_b_norm_b = Vec::new();
        let mut layer_c_norm_w = Vec::new();
        let mut layer_c_norm_b = Vec::new();
        let mut layer_d_param = Vec::new();
        for layer in &model.layers {
            layer_dt_bias.push(w(&layer.dt_bias));
            layer_b_norm_w.push(w(&layer.b_norm_w));
            layer_b_norm_b.push(w(&layer.b_norm_b));
            layer_c_norm_w.push(w(&layer.c_norm_w));
            layer_c_norm_b.push(w(&layer.c_norm_b));
            layer_d_param.push(w(&layer.d_param));
        }

        // Pre-build ALL bind groups with WRITABLE params buffers.
        // Params are updated with actual L via queue.write_buffer each forward.
        // Bind groups stay valid because they reference the same buffer objects.
        let l = max_seq as u32; // Params init value — overwritten each forward
        let d32 = d as u32;
        let di32 = di as u32;
        let v32 = v as u32;
        let nh32 = nh as u32;
        let hd32 = hd as u32;
        let ds32 = ds as u32;

        let mk_norm_bg = |x: &wgpu::Buffer, ww: &wgpu::Buffer, bb: &wgpu::Buffer, seq: u32, dim: u32| -> wgpu::BindGroup {
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct NP { s: u32, d: u32, _0: u32, _1: u32 }
            let pb = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::bytes_of(&NP { s: seq, d: dim, _0: 0, _1: 0 }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &norm_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ww.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: bb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: pb.as_entire_binding() },
                ],
            })
        };

        let mk_matmul_bg = |a: &wgpu::Buffer, b: &wgpu::Buffer, c: &wgpu::Buffer, m: u32, k: u32, n: u32| -> wgpu::BindGroup {
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct MP { m: u32, k: u32, n: u32, _pad: u32 }
            let pb = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::bytes_of(&MP { m, k, n, _pad: 0 }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &gpu.matmul_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: pb.as_entire_binding() },
                ],
            })
        };

        let mk_res_bg = |x: &wgpu::Buffer, y: &wgpu::Buffer, n: u32, scale: f32| -> wgpu::BindGroup {
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct RP { n: u32, sb: u32, _0: u32, _1: u32 }
            let pb = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::bytes_of(&RP { n, sb: scale.to_bits(), _0: 0, _1: 0 }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &residual_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: pb.as_entire_binding() },
                ],
            })
        };

        let embed_norm_bg = mk_norm_bg(&x_buf, &embed_norm_w, &embed_norm_b, l, d32);
        let final_norm_bg = mk_norm_bg(&x_buf, &final_norm_w, &final_norm_b, l, d32);
        let head_matmul_bg = mk_matmul_bg(&x_buf, &embed_w, &logits_buf, l, d32, v32);

        let mut layer_norm_bgs = Vec::new();
        let mut layer_in_proj_bgs = Vec::new();
        let mut layer_out_proj_bgs = Vec::new();
        let mut layer_residual_bgs = Vec::new();
        let mut layer_prep_bgs = Vec::new();
        let mut layer_scan_bgs = Vec::new();

        for (li, gl) in layers.iter().enumerate() {
            layer_norm_bgs.push(mk_norm_bg(&x_normed_buf, &gl.norm_w, &gl.norm_b, l, d32));
            layer_in_proj_bgs.push(mk_matmul_bg(&x_normed_buf, &gl.in_proj_w, &gl.proj_buf, l, d32, gl.d_in_proj as u32));
            layer_out_proj_bgs.push(mk_matmul_bg(&scan_out_buf, &gl.out_proj_w, &gl.y_out_buf, l, di32, d32));
            layer_residual_bgs.push(mk_res_bg(&x_buf, &gl.y_out_buf, l * d32, gl.scale));

            // SSM prep bind group
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct PP { l: u32, di: u32, ds: u32, nh: u32, hd: u32, dip: u32, na: u32, _p: u32 }
            let pp = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::bytes_of(&PP {
                    l, di: di32, ds: ds32, nh: nh32, hd: hd32, dip: gl.d_in_proj as u32, na: ds32 / 2, _p: 0
                }), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            layer_prep_bgs.push(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &ssm_prep_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gl.proj_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: layer_dt_bias[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: layer_b_norm_w[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: layer_b_norm_b[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: layer_c_norm_w[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: layer_c_norm_b[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: inp_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: decay_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: cp_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: x_skip_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: z_silu_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: pp.as_entire_binding() },
                ],
            }));

            // SSM scan bind group
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct SP { b: u32, l: u32, h: u32, hd: u32, ds: u32 }
            let sp = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::bytes_of(&SP { b: 1, l, h: nh32, hd: hd32, ds: ds32 }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            layer_scan_bgs.push(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &gpu.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: inp_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: decay_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cp_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: x_skip_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: z_silu_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: layer_d_param[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: scan_out_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: sp.as_entire_binding() },
                ],
            }));
        }

        // Fused layer pipeline (12 bindings in group 0, 1 in group 1)
        let fused_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None, source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fused_layer.wgsl").into()),
        });
        let fused_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None, entries: &[
                bgl_rw(0), bgl_ro(1), bgl_ro(2), bgl_ro(3), bgl_ro(4),
                bgl_ro(5), bgl_ro(6), bgl_ro(7), bgl_ro(8), bgl_ro(9), bgl_ro(10), bgl_u(11),
            ],
        });
        let fused_scale_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None, entries: &[bgl_u(0)],
        });
        let fused_pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&fused_layout, &fused_scale_layout], push_constant_ranges: &[],
        });
        let fused_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: Some(&fused_pl), module: &fused_shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });

        eprintln!("  FullGpuModel: {} layers, max_seq={}, pre-built {} bind groups",
            model.layers.len(), max_seq, 2 + layers.len() * 6 + 1);

        Self {
            cpu_model: model, gpu, embed_w, embed_norm_w, embed_norm_b,
            final_norm_w, final_norm_b, layers, x_buf, x_normed_buf,
            logits_buf, staging, inp_buf, decay_buf, cp_buf,
            x_skip_buf, z_silu_buf, scan_out_buf,
            layer_dt_bias, layer_b_norm_w, layer_b_norm_b,
            layer_c_norm_w, layer_c_norm_b, layer_d_param,
            norm_pipeline, norm_layout, residual_pipeline, residual_layout,
            ssm_prep_pipeline, ssm_prep_layout,
            fused_pipeline, fused_layout, fused_scale_layout,
            all_param_bufs: Vec::new(),
            max_seq,
            embed_norm_bg, final_norm_bg, head_matmul_bg,
            layer_norm_bgs, layer_in_proj_bgs, layer_out_proj_bgs,
            layer_residual_bgs, layer_prep_bgs, layer_scan_bgs,
        }
    }

    /// Forward — ZERO mid-forward readback. One upload, one download.
    /// All ops (norm, matmul, SSM prep, scan, residual) run as GPU shaders.
    /// Single command buffer. Single submit. Single readback. Zero mid-forward transfers.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let l = tokens.len();
        let d = self.cpu_model.d_model;
        let v = self.cpu_model.vocab_size;
        let di = self.cpu_model.d_inner;
        let nh = self.cpu_model.n_heads as u32;
        let hd = self.cpu_model.headdim as u32;
        let ds = self.cpu_model.d_state as u32;

        // Embedding (CPU lookup, ~0.5KB)
        let mut x = vec![0.0f32; l * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok < v { for i in 0..d { x[t * d + i] = self.cpu_model.embed_w[tok * d + i]; } }
        }

        // === THE ONLY UPLOAD ===
        self.gpu.queue.write_buffer(&self.x_buf, 0, bytemuck::cast_slice(&x));

        // === SINGLE COMMAND BUFFER - ENTIRE FORWARD ===
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());

        // Note: pre-built bind groups use max_seq for params.
        // Dispatch with actual l — excess workgroups exit early via bounds check in shader.
        let l32 = l as u32;

        // Embed norm
        { let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.norm_pipeline);
          pass.set_bind_group(0, &self.embed_norm_bg, &[]);
          pass.dispatch_workgroups(l32, 1, 1); }

        // All layers — pre-built bind groups, zero creation
        let tile = 16u32;
        for li in 0..self.layers.len() {
            let gl = &self.layers[li];
            let dip = gl.d_in_proj as u32;

            // Copy x → x_normed
            enc.copy_buffer_to_buffer(&self.x_buf, 0, &self.x_normed_buf, 0, (l * d * 4) as u64);

            // Pre-norm
            { let mut pass = enc.begin_compute_pass(&Default::default());
              pass.set_pipeline(&self.norm_pipeline);
              pass.set_bind_group(0, &self.layer_norm_bgs[li], &[]);
              pass.dispatch_workgroups(l as u32, 1, 1); }

            // In-proj matmul
            { let mut pass = enc.begin_compute_pass(&Default::default());
              pass.set_pipeline(&self.gpu.matmul_pipeline);
              pass.set_bind_group(0, &self.layer_in_proj_bgs[li], &[]);
              pass.dispatch_workgroups((l as u32 + tile - 1) / tile, (dip + tile - 1) / tile, 1); }

            // SSM prep
            { let mut pass = enc.begin_compute_pass(&Default::default());
              pass.set_pipeline(&self.ssm_prep_pipeline);
              pass.set_bind_group(0, &self.layer_prep_bgs[li], &[]);
              pass.dispatch_workgroups(nh, 1, 1); }

            // SSM scan
            { let mut pass = enc.begin_compute_pass(&Default::default());
              pass.set_pipeline(&self.gpu.pipeline);
              pass.set_bind_group(0, &self.layer_scan_bgs[li], &[]);
              pass.dispatch_workgroups(nh, 1, 1); }

            // Out-proj matmul
            { let mut pass = enc.begin_compute_pass(&Default::default());
              pass.set_pipeline(&self.gpu.matmul_pipeline);
              pass.set_bind_group(0, &self.layer_out_proj_bgs[li], &[]);
              pass.dispatch_workgroups((l as u32 + tile - 1) / tile, (d as u32 + tile - 1) / tile, 1); }

            // Residual
            { let mut pass = enc.begin_compute_pass(&Default::default());
              pass.set_pipeline(&self.residual_pipeline);
              pass.set_bind_group(0, &self.layer_residual_bgs[li], &[]);
              pass.dispatch_workgroups(((l * d) as u32 + 255) / 256, 1, 1); }
        }

        // Final norm + LM head — pre-built
        { let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.norm_pipeline);
          pass.set_bind_group(0, &self.final_norm_bg, &[]);
          pass.dispatch_workgroups(l as u32, 1, 1); }
        { let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.gpu.matmul_pipeline);
          pass.set_bind_group(0, &self.head_matmul_bg, &[]);
          pass.dispatch_workgroups((l as u32 + tile - 1) / tile, (v as u32 + tile - 1) / tile, 1); }

        // === THE ONLY READBACK ===
        enc.copy_buffer_to_buffer(&self.logits_buf, 0, &self.staging, 0, (l * v * 4) as u64);

        // === SINGLE SUBMIT ===
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        self.map_staging(l * v)
    }

    /// Fused forward: ONE dispatch per layer (norm+matmul+SSM+matmul+residual fused).
    /// Total dispatches: 1 (embed_norm) + N (fused layers) + 1 (final_norm) + 1 (head) = N+3
    /// vs old: 6N+3 dispatches. For 5 layers: 8 vs 33 dispatches.
    pub fn forward_fused(&self, tokens: &[u32]) -> Vec<f32> {
        let l = tokens.len();
        let d = self.cpu_model.d_model;
        let v = self.cpu_model.vocab_size;
        let di = self.cpu_model.d_inner;

        // Embedding (CPU)
        let mut x = vec![0.0f32; l * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok < v { for i in 0..d { x[t * d + i] = self.cpu_model.embed_w[tok * d + i]; } }
        }
        self.gpu.queue.write_buffer(&self.x_buf, 0, bytemuck::cast_slice(&x));

        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let l32 = l as u32;
        let tile = 16u32;

        // Embed norm
        { let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.norm_pipeline);
          pass.set_bind_group(0, &self.embed_norm_bg, &[]);
          pass.dispatch_workgroups(l32, 1, 1); }

        // Fused layers — ONE dispatch each
        for (li, gl) in self.layers.iter().enumerate() {
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct FP { l: u32, d: u32, di: u32, ds: u32, nh: u32, hd: u32, dip: u32, na: u32 }
            let fp = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&FP {
                    l: l32, d: d as u32, di: di as u32, ds: self.cpu_model.d_state as u32,
                    nh: self.cpu_model.n_heads as u32, hd: self.cpu_model.headdim as u32,
                    dip: gl.d_in_proj as u32, na: (self.cpu_model.d_state / 2) as u32,
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct SC { sb: u32, _0: u32, _1: u32, _2: u32 }
            let sc = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&SC { sb: gl.scale.to_bits(), _0: 0, _1: 0, _2: 0 }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let bg0 = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.fused_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: gl.in_proj_w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: gl.out_proj_w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.layer_dt_bias[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.layer_d_param[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.layer_b_norm_w[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.layer_b_norm_b[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: self.layer_c_norm_w[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: self.layer_c_norm_b[li].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: gl.norm_w.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: gl.norm_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: fp.as_entire_binding() },
                ],
            });
            let bg1 = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &self.fused_scale_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: sc.as_entire_binding() },
                ],
            });

            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.fused_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.dispatch_workgroups(1, 1, 1); // Single workgroup — all heads sequential
        }

        // Final norm + head
        { let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.norm_pipeline);
          pass.set_bind_group(0, &self.final_norm_bg, &[]);
          pass.dispatch_workgroups(l32, 1, 1); }
        { let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&self.gpu.matmul_pipeline);
          pass.set_bind_group(0, &self.head_matmul_bg, &[]);
          pass.dispatch_workgroups((l32 + tile - 1) / tile, (v as u32 + tile - 1) / tile, 1); }

        enc.copy_buffer_to_buffer(&self.logits_buf, 0, &self.staging, 0, (l * v * 4) as u64);
        self.gpu.queue.submit(std::iter::once(enc.finish()));
        self.map_staging(l * v)
    }

    fn encode_ssm_prep(&self, enc: &mut wgpu::CommandEncoder, proj_buf: &wgpu::Buffer, li: usize, l: u32) {
        let nh = self.cpu_model.n_heads as u32;
        let di = self.cpu_model.d_inner as u32;
        let ds = self.cpu_model.d_state as u32;
        let hd = self.cpu_model.headdim as u32;
        let dip = self.layers[li].d_in_proj as u32;
        let na = ds / 2;
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P { l: u32, di: u32, ds: u32, nh: u32, hd: u32, dip: u32, na: u32, _p: u32 }
        let p = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&P { l, di, ds, nh, hd, dip, na, _p: 0 }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.ssm_prep_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: proj_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.layer_dt_bias[li].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.layer_b_norm_w[li].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.layer_b_norm_b[li].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.layer_c_norm_w[li].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.layer_c_norm_b[li].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.inp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.decay_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.cp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.x_skip_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.z_silu_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: p.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.ssm_prep_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(nh, 1, 1);
    }

    fn encode_ssm_scan(&self, enc: &mut wgpu::CommandEncoder, li: usize, l: u32) {
        let nh = self.cpu_model.n_heads as u32;
        let hd = self.cpu_model.headdim as u32;
        let ds = self.cpu_model.d_state as u32;
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SP { b: u32, l: u32, h: u32, hd: u32, ds: u32 }
        let p = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&SP { b: 1, l, h: nh, hd, ds }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bg = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.gpu.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.inp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.decay_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.cp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.x_skip_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.z_silu_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.layer_d_param[li].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.scan_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: p.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.gpu.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(nh, 1, 1);
    }

    fn encode_norm(&self, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, w: &wgpu::Buffer, b: &wgpu::Buffer, seq_len: u32, d: u32) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P { seq_len: u32, d: u32, _p0: u32, _p1: u32 }
        let p = self.gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&P { seq_len, d, _p0: 0, _p1: 0 }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
