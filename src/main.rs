use nalgebra::Vector2;
use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;
use winit::{
    event::{ElementState, MouseButton, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
    application::{ApplicationHandler}
};

mod solver;
mod gpu_renderer; // Import our new module

const WIDTH: usize = 1600;
const HEIGHT: usize = 1600;

struct App {
    window: Option<Window>,
    gpu_renderer: Option<gpu_renderer::GpuRenderer>,
    time: f32,
    physics_solver: solver::PhysicsSolver,
    frame_count: u32,
    mouse_pos: Vector2<f32>,
    last_fps_time: Instant,
    paused: bool
}

impl App {
    fn new() -> Self {
        let mut physics_solver = solver::PhysicsSolver::new(WIDTH as i32, HEIGHT as i32);
        
        
        physics_solver.init_world(3000, 0.99);
        
        Self {
            window: None,
            mouse_pos: Vector2::new(0.0,0.0),
            gpu_renderer: None,
            time: 0.0,
            physics_solver,
            frame_count: 0,
            last_fps_time: Instant::now(),
            paused: true
        }
    }
}

fn handle_keyboard_input(event: KeyEvent, app: &mut App) {
    if event.state == ElementState::Pressed && !event.repeat {
        match event.logical_key {
            Key::Named(NamedKey::Escape) => {
                
                println!("Escape key pressed!");
            }
            Key::Character(c) => {
               
            }
            _ => {
               
            }
        }
    } else if event.state == ElementState::Released {
        match event.logical_key {
            Key::Named(NamedKey::Escape) => {
                println!("Escape key released!");
            }
            Key::Named(NamedKey::Space) => {
                app.paused = !app.paused;
            }
            Key::Named(NamedKey::ArrowRight) => {
                if app.paused {
                    app.physics_solver.update(1E-3, 1, Vector2::new(0.0, 0.0));
                    app.time += 1E-4;
                }else{
                    app.paused = true;
                    app.physics_solver.update(1E-3, 1, Vector2::new(0.0, 0.0));
                    app.time += 1E-4;
                }
            }
        
            Key::Character(c) => {

            }
            _ => {
               
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("")
                .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH as u32, HEIGHT as u32));

            let window = event_loop
                .create_window(window_attributes)
                .expect("Failed to create window");

            // Initialize GpuRenderer
            self.gpu_renderer = Some(pollster::block_on(gpu_renderer::GpuRenderer::new(&window)));
            self.window = Some(window);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Poll regularly to keep the simulation and rendering active
        event_loop.set_control_flow(ControlFlow::Poll);
    
        match event {
            WindowEvent::CloseRequested => {
                println!("Close requested. Exiting.");
                event_loop.exit();
            }
    
            WindowEvent::RedrawRequested => {
                if !self.paused {
                    // Update the physics simulation
                    self.physics_solver.update(1E-3, 1, Vector2::new(0.0, 0.0));
                    self.time += 1E-3;
                }
    
                let num_physics_particles = self.physics_solver.positions.len();
                let mut gpu_particles: Vec<gpu_renderer::GpuParticle> = Vec::with_capacity(num_physics_particles);
                for i in 0..num_physics_particles {
                    gpu_particles.push(gpu_renderer::GpuParticle {
                        position: [
                            self.physics_solver.positions[i].x,
                            self.physics_solver.positions[i].y,
                        ],
                        radius: self.physics_solver.radii[i],
                        is_plant: self.physics_solver.is_plant[i] as u32
                    });
                }
    
                if let (Some(renderer), Some(window)) = (&mut self.gpu_renderer, &self.window) {
                    renderer.render(window, &gpu_particles, num_physics_particles as u32);
                }
                
                // ---
                // MODIFIED CODE: Write to CSV on every frame
                // Note: This will open and close the file for every frame, which can be inefficient.
                // For a more performant solution, consider opening the file once at the start of the simulation.
                if let Ok(mut file) = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("simulation_data.csv")
                {
                    // If the file is new (empty), write the header
                    if file.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
                        let _ = writeln!(file, "time,num_cells,num_plants,avg_speed,avg_brain_size,avg_sight_r,avg_predation,avg_age");
                    }
                    
                    let now = Instant::now();
                    // Use total elapsed time since the simulation began
                    let time_elapsed = self.time;
                    
                    let _ = writeln!(
                        file,
                        "{},{},{},{},{},{},{},{}",
                        time_elapsed,
                        self.physics_solver.num_cells,
                        self.physics_solver.num_plants,
                        self.physics_solver.avg_speed, // Averages are now per-frame, so no division needed
                        self.physics_solver.avg_brain_size,
                        self.physics_solver.avg_sight_r,
                        self.physics_solver.avg_pred,
                        self.physics_solver.avg_age
                    );
                }
                
                // Reset averages for the next frame
                self.physics_solver.reset_avgs();
                // ---
    
                // Original FPS counter logic
                self.frame_count += 1;
                let now = Instant::now();
                if now.duration_since(self.last_fps_time).as_secs() >= 1 {
                    let fps = self.frame_count as f64
                        / now.duration_since(self.last_fps_time).as_secs_f64();
                    println!("FPS: {:.1}", fps);
                    self.frame_count = 0;
                    self.last_fps_time = now;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = Vector2::new(position.x as f32, position.y as f32);
            }

            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                self.physics_solver.add_cell(self.mouse_pos, 1.0, 10.0, Vector2::new(0.0, 0.0));
            }
            
    
            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = &mut self.gpu_renderer {
                    renderer.resize(physical_size);
                }
            }
    
            WindowEvent::KeyboardInput { event, .. } => {
                handle_keyboard_input(event, self);
            }
    
            _ => {}
        }
    }

    

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Request a redraw every frame to keep the simulation moving
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create EventLoop");
    let mut app = App::new();

    event_loop.run_app(&mut app).expect("EventLoop run failed");
    app.physics_solver.save_random_creature(); // Save a random creature's brain at the end
}