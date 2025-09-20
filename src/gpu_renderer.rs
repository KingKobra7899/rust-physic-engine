// gpu_renderer.rs
use winit::{
    window::{Window},
};
use wgpu::util::DeviceExt;
use std::time::Instant;

const WIDTH: usize = 1000;
const HEIGHT: usize = 1000;

// Struct to represent a particle in the GPU buffer
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 2],
    pub radius: f32,
    pub is_plant: u32,

}

pub struct GpuRenderer {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_pipeline: wgpu::RenderPipeline,
    
    particle_buffer: wgpu::Buffer,
    particle_bind_group: wgpu::BindGroup,
    particle_bind_group_layout: wgpu::BindGroupLayout, // Keep this to re-create bind group if needed
    particle_count_buffer: wgpu::Buffer,

    screen_dims_buffer: wgpu::Buffer,
    screen_dims_bind_group: wgpu::BindGroup,
    screen_dims_bind_group_layout: wgpu::BindGroupLayout, // Also store this layout

    vertex_buffer: wgpu::Buffer,

    last_frame_time: Instant,
}

impl GpuRenderer {
    pub async fn new(window: &Window) -> Self {
        let instance = wgpu::Instance::default();

        // Safety: The window will outlive the surface. This transmute is common in WGPU examples
        // with winit when needing a 'static reference for surface creation.
        let leaked_window_ref = unsafe {
            std::mem::transmute::<&Window, &'static Window>(window)
        };

        let surface = instance.create_surface(leaked_window_ref).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off
                }
            )
            .await
            .unwrap();

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync, // For faster updates
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // --- WGPU Setup for Particles (Fragment Shader access) ---

        // Initial particle data buffer. This will be updated every frame.
        // Initialize with a placeholder or empty vec. The main loop will populate it.
        // For the bind group to be valid, the buffer's initial size cannot be zero.
        // Provide at least one dummy particle.
        let initial_particle_data: Vec<GpuParticle> = vec![GpuParticle { position: [0.0, 0.0], radius: 0.0,is_plant:0}]; 
        let initial_particle_count = 0; // The count will be updated on the first render

        let particle_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Particle Buffer"),
                contents: bytemuck::cast_slice(&initial_particle_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        );

        // Placeholder for particle count (Uniform buffer for a single u32)
        let particle_count_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Particle Count Buffer"),
                contents: bytemuck::cast_slice(&[initial_particle_count as u32]), // Initial count
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        // Particle Bind Group Layout (Group 0)
        let particle_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        // Binding 0: Particle data (Storage Buffer, read-only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT, // Only fragment shader needs access
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Binding 1: Number of particles (Uniform Buffer)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer { // <--- FIX IS HERE
                                ty: wgpu::BufferBindingType::Uniform, // <--- Corrected
                                has_dynamic_offset: false,
                                min_binding_size: None, // Will be size of a u32
                            },
                            count: None,
                        }
                    ],
                    label: Some("Particle Bind Group Layout"),
                });

            // Particle Bind Group (Group 0)
            let particle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &particle_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_count_buffer.as_entire_binding(),
                    },
                ],
                label: Some("Particle Bind Group"),
            });


            // --- Screen Dimensions Buffer (for Fragment Shader) ---
            let screen_dims_buffer = device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Screen Dimensions Buffer"),
                    contents: bytemuck::cast_slice(&[WIDTH as f32, HEIGHT as f32]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                }
            );

            // Screen Dimensions Bind Group Layout (Group 1)
            let screen_dims_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        // Binding 0: Screen dimensions (Uniform Buffer)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer { // <--- FIX IS HERE
                                ty: wgpu::BufferBindingType::Uniform, // <--- Corrected
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }
                    ],
                    label: Some("Screen Dimensions Bind Group Layout"),
                });

            // Screen Dimensions Bind Group (Group 1)
            let screen_dims_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &screen_dims_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: screen_dims_buffer.as_entire_binding(),
                    }
                ],
                label: Some("Screen Dimensions Bind Group"),
            });


            // --- Fullscreen Triangle Vertex Buffer ---
            #[rustfmt::skip]
            let vertices: &[f32] = &[
                -1.0, -1.0,
                 3.0, -1.0,
                -1.0,  3.0,
            ];
            let vertex_buffer = device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Fullscreen Triangle Vertex Buffer"),
                    contents: bytemuck::cast_slice(vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }
            );

            // --- Shader and Pipeline Setup ---

            let vs_source = std::fs::read_to_string("shaders/shader.vert.wgsl")
                .expect("Failed to read vertex shader. Make sure shaders/shader.vert.wgsl exists.");
            let fs_source = std::fs::read_to_string("shaders/shader.frag.wgsl")
                .expect("Failed to read fragment shader. Make sure shaders/shader.frag.wgsl exists.");

            let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Vertex Shader"),
                source: wgpu::ShaderSource::Wgsl(vs_source.into()),
            });

            let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Fragment Shader"),
                source: wgpu::ShaderSource::Wgsl(fs_source.into()),
            });

            let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &particle_bind_group_layout,      // Group 0 for particles
                    &screen_dims_bind_group_layout,   // Group 1 for screen dimensions
                ],
                push_constant_ranges: &[],
            });

            let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vs_module,
                    entry_point: Some("vs_main"),
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            array_stride: (2 * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                        },
                    ],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),                
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        println!("WGPU initialized and surface configured!");

        Self {
            surface,
            surface_config: config,
            device,
            queue,
            render_pipeline,
            particle_buffer,
            particle_bind_group,
            particle_bind_group_layout, // Stored for potential re-creation
            particle_count_buffer,
            screen_dims_buffer,
            screen_dims_bind_group,
            screen_dims_bind_group_layout, // Stored for potential re-creation
            vertex_buffer,
            last_frame_time: Instant::now(),
        }
    }

    pub fn render(&mut self, window: &Window, particles: &[GpuParticle], num_particles: u32) {
        // --- Dynamic Buffer Sizing Check (from your original logic, now handled here) ---
        // If the number of particles grows beyond the initial buffer capacity,
        // we need to re-create the buffer and its corresponding bind group.
        let required_size = (particles.len() * std::mem::size_of::<GpuParticle>()) as wgpu::BufferAddress;
        if required_size > self.particle_buffer.size() {
            println!("Resizing particle buffer from {} to {}", self.particle_buffer.size(), required_size);
            // Recreate the buffer with a larger size. Grow by at least double, and ensure it's not zero.
            self.particle_buffer = self.device.create_buffer(
                &wgpu::BufferDescriptor {
                    label: Some("Particle Buffer (Resized)"),
                    size: required_size.max(self.particle_buffer.size() * 2).max(1), 
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            );
            // Crucially: Recreate the bind group with the new buffer
            self.particle_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.particle_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.particle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.particle_count_buffer.as_entire_binding(),
                    },
                ],
                label: Some("Particle Bind Group"),
            });
        }


        // Update GPU Buffers with Latest Physics Data
        self.queue.write_buffer(&self.particle_buffer, 0, bytemuck::cast_slice(particles));
        
        // Update the number of particles in its uniform buffer
        self.queue.write_buffer(&self.particle_count_buffer, 0, bytemuck::cast_slice(&[num_particles as u32]));

        // Update screen dimensions buffer (important if window is resizable)
        let current_size = window.inner_size();
        let dims = [current_size.width as f32, current_size.height as f32];
        self.queue.write_buffer(&self.screen_dims_buffer, 0, bytemuck::cast_slice(&dims));

        // --- Begin Render Pass ---
        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }), // Clear to black
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Set the render pipeline
            render_pass.set_pipeline(&self.render_pipeline);
            // Set the bind groups
            render_pass.set_bind_group(0, &self.particle_bind_group, &[]); // Group 0: Particles
            render_pass.set_bind_group(1, &self.screen_dims_bind_group, &[]); // Group 1: Screen Dimensions
            // Set the vertex buffer (fullscreen triangle)
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            // Draw the fullscreen triangle (3 vertices, 1 instance)
            render_pass.draw(0..3, 0..1);
        }

        // Submit the command buffer and present the output
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
            // Note: screen_dims_buffer is updated in render() method, so no need to update here.
        }
    }
}