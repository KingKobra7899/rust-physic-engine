use fast_poisson::Poisson2D;
use nalgebra::{clamp, Vector2};
use rand::{rngs::ThreadRng, Rng};
mod quadtree;
use quadtree::{QuadTree, Rect};
type Connection = (usize, usize, f32);
pub struct PhysicsSolver {
    pub positions: Vec<Vector2<f32>>,
    pub old_positions: Vec<Vector2<f32>>,
    pub accelerations: Vec<Vector2<f32>>,
    pub masses: Vec<f32>,
    pub radii: Vec<f32>,
    width: i32,
    height: i32,
    boundary: Rect,
    pub rng: ThreadRng,
    pub qt: QuadTree,
    center: Vector2<f32>,
    pub(crate) num_particles: usize,
    springs: Vec<Connection>
}

impl PhysicsSolver {
    pub fn new(width: i32, height: i32) -> PhysicsSolver {
        PhysicsSolver {
            positions: Vec::new(),
            num_particles: 0,
            rng: rand::rng(),
            old_positions: Vec::new(),
            accelerations: Vec::new(),
            masses: Vec::new(),
            radii: Vec::new(),
            springs: Vec::new(),
            width,
            height,
            boundary: Rect::new((width / 2) as f32, (height / 2) as f32, (width / 2) as f32, (height / 2) as f32),
            center: Vector2::new((width as f32) / 2.0, (height as f32) / 2.0),
            qt: QuadTree::new(Rect::new((width as f32) / 2.0, (height as f32) / 2.0, (width as f32) / 2.0, (height as f32) / 2.0), 5),
        }
    }
    pub fn connect_particles(&mut self, p1: usize, p2:usize){
        let dist = (self.positions[p1] - self.positions[p2]);

        self.springs.push((p1, p2, 0.0f32));
    }
    pub fn apply_springs(&mut self, k: f32) {
        let springs = self.springs.clone();
        for spring in &springs {
            let p1 = spring.0;
            let p2 = spring.1;
            let d = spring.2;
            let axis: Vector2<f32> = self.positions[p1] - self.positions[p2];
            let dist = axis.magnitude();

            let norm = axis / dist;

            let force = k * (dist - d) * norm;
            self.accelerate_particle(p1, -force);
            self.accelerate_particle(p2, force);
        }
    }
    pub fn add_particle_grid(
        &mut self,
        num_x: usize,
        num_y: usize,
        start_pos: Vector2<f32>,
        radius: f32,
        spacing: f32,
        mass: f32,
        random_r: bool,
        vel: Vector2<f32>
    ) {
        for y in 0..num_y {
            for x in 0..num_x {
                let pos_x = start_pos.x + x as f32 * spacing;
                let pos_y = start_pos.y + y as f32 * spacing;
                if !random_r {
                    self.add_particle(Vector2::new(pos_x, pos_y), mass, radius, vel);
                } else {
                    let random_mult: f32 = self.rng.random_range(0.75..1.25);
                    self.add_particle(Vector2::new(pos_x, pos_y), mass * random_mult * random_mult, radius * random_mult, vel);
                }
            }
        }
    }

    pub fn apply_newtonian_grav(&mut self, grav: f32) {
        for i in 0..self.num_particles {
            let pos: Vector2<f32> = self.positions[i];
            let r: f32 = self.radii[i];
            let col_box = Rect::new(pos.x, pos.y, 10.0 * r, 10.0 * r);

            let indices: Vec<i32> = self.qt.query(&col_box);

            for n in indices {
                if n == (i as i32) {
                    continue;
                }

                let norm: Vector2<f32> = self.positions[i] - self.positions[n as usize];
                let dist: f32 = norm.magnitude();

                if dist > 0.0 {
                    let grav_mag: f32 = (grav * self.masses[i] * self.masses[n as usize]) / (dist * dist);
                    let force: Vector2<f32> = (norm / dist) * (grav_mag);

                    self.accelerate_particle(i, -1.0 * force / 2.0);
                    self.accelerate_particle(n as usize, force / 2.0);
                }
            }
        }
    }

    pub fn add_particle(&mut self, pos: Vector2<f32>, mass: f32, radius: f32, vel: Vector2<f32>) {
        self.positions.push(pos);
        self.old_positions.push(pos - vel);
        self.accelerations.push(Vector2::new(0.0, 0.0));
        self.masses.push(mass);
        self.radii.push(radius);
        self.num_particles += 1;
    }


    pub fn accelerate_particle(&mut self, index: usize, force: Vector2<f32>) {
        if index < self.accelerations.len() && index < self.masses.len() {
            self.accelerations[index] += force / self.masses[index];
        }
    }

    pub fn integrate_forces(&mut self, dt: f32, grav: Vector2<f32>, friction: f32, drag: f32) {
        for i in 0..self.num_particles as usize {
            let pos = self.positions[i];
            let old_pos = self.old_positions[i];
    

            let displacement = pos - old_pos;

            let acc = self.accelerations[i] + grav;

            let new_pos = pos + displacement + acc * dt * dt;

            self.old_positions[i] = pos;
            self.positions[i] = new_pos;

            self.accelerations[i] = Vector2::new(0.0, 0.0);
        }
    }

    pub fn update_quadtree(&mut self) {
        self.qt.clear();
        
        for i in 0..self.num_particles as usize {
            self.qt.insert(&self.positions[i], i as i32);
        }
    }

    pub fn inter_particle_collisions(&mut self) {
        for i in 0..(self.num_particles as usize) {
            let pos: Vector2<f32> = self.positions[i];
            let r: f32 = self.radii[i];
            let col_box = Rect::new(pos.x, pos.y, 3.0 * r, 3.0 * r);
    
            let indices = self.qt.query(&col_box);
    
            for n in indices {
                if n == (i as i32) {
                    continue;
                }
                
                // Add bounds check before accessing
                if (n as usize) >= self.num_particles as usize {
                    continue;
                }
                
                let other_pos: Vector2<f32> = self.positions[n as usize];
                let other_r: f32 = self.radii[n as usize];
    
                let col_axis: Vector2<f32> = pos - other_pos;
                let dist = col_axis.magnitude();
    
                if dist < (r + other_r) && dist > 0.0 {
                    let norm: Vector2<f32> = col_axis / dist;
                    let overlap: f32 = (r + other_r) - dist;
    
                    let sep1: f32 = overlap * (other_r / (r + other_r));
                    let sep2: f32 = overlap * (r / (r + other_r));
                    
                    self.positions[i] += 0.5 * norm * sep1;
                    self.positions[n as usize] -= 0.5 * norm * sep2;
                }
            }
        }
    }

    pub fn apply_circular_constraint(&mut self, con_radius: f32) {
        for i in 0..(self.num_particles as usize) {
            let pos: Vector2<f32> = self.positions[i];
            let radius: f32 = self.radii[i];

            let to_obj: Vector2<f32> = pos - self.center;
            let dist: f32 = to_obj.magnitude();

            if dist > (con_radius - radius) {
                let n: Vector2<f32> = to_obj / dist;
                let new_pos: Vector2<f32> = self.center + n * (con_radius - radius);
                self.positions[i] = new_pos;
            }
        }
    }

    pub fn apply_rect_constraint(&mut self, rectangle: Rect) {
        for i in 0..self.num_particles {
            let pos = &mut self.positions[i as usize];
            let r = self.radii[i as usize];
    
            let y_top = rectangle.y - rectangle.h;
            let y_bottom = rectangle.y + rectangle.h;
            let x_left = rectangle.x - rectangle.w;
            let x_right = rectangle.x + rectangle.w;
    
            // Top boundary
            if pos.y - r < y_top {
                pos.y = y_top + r;
            // Bottom boundary
            } else if pos.y + r > y_bottom {
                pos.y = y_bottom - r;
            }
    
            // Left boundary
            if pos.x - r < x_left {
                pos.x = x_left + r;
            } else if pos.x + r > x_right {
                pos.x = x_right - r;
            }
        }
    }

    pub fn update(&mut self, dt: f32, substeps: i32, grav: Vector2<f32>) {
        self.update_quadtree();
        for _ in 0..substeps {
            self.inter_particle_collisions();
            self.apply_rect_constraint(self.boundary);
            //self.apply_newtonian_grav(100.0);
            self.apply_springs(10.0);
            self.integrate_forces(dt / (substeps as f32), grav, 1.50, 15.0);
        }
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}