// solver.rs - Fixed version with deferred deletion system
use fast_poisson::Poisson2D;
use nalgebra::{clamp, Vector2};
use rand::{rngs::ThreadRng, Rng};
mod quadtree;
use quadtree::{QuadTree, Rect};
mod cell;
use cell::Cell;

pub struct PhysicsSolver {
    pub positions: Vec<Vector2<f32>>,
    pub old_positions: Vec<Vector2<f32>>,
    pub accelerations: Vec<Vector2<f32>>,
    pub masses: Vec<f32>,
    pub cells: Vec<Cell>,
    pub cell_indices: Vec<usize>,
    pub plants: Vec<usize>,
    pub radii: Vec<f32>,
    pub is_plant: Vec<bool>,
    pub spring_connections: Vec<Vector2<usize>>,
    pub avg_thresh: f32,
    width: i32,
    height: i32,
    pub(crate) num_cells: i32,
    pub(crate) num_plants: i32,
    boundary: Rect,
    pub rng: ThreadRng,
    pub qt: QuadTree,
    center: Vector2<f32>,
    pub avg_speed: f32,
    pub avg_brain_size: f32,
    pub avg_hunger: f32,
    pub avg_isolation: f32,
    pub avg_social: f32,
    pub avg_sight_r: f32,
    pub num_particles: i32,
    pub avg_pred: f32,
    pub avg_age: f32,
    pub pending_deletions: Vec<usize>, // New field for deferred deletions
    pub pending_additions: Vec<(Vector2<f32>, f32, i32, i32, f32, f32, f32, f32, f32, f32, Cell)>, // New field for deferred additions
}

impl PhysicsSolver {
    pub fn new(width: i32, height: i32) -> PhysicsSolver {
        return PhysicsSolver {
            positions: Vec::new(),
            spring_connections: Vec::new(),
            rng: rand::rng(),
            old_positions: Vec::new(),
            accelerations: Vec::new(),
            masses: Vec::new(),
            radii: Vec::new(),
            cells: Vec::new(),
            avg_speed: 0.0,
            avg_brain_size: 0.0,
            avg_thresh: 0.0,
            avg_hunger: 0.0,
            avg_isolation: 0.0,
            avg_social: 0.0,
            avg_sight_r: 0.0,
            avg_pred: 0.0,
            cell_indices: Vec::new(),
            plants: Vec::new(),
            is_plant: Vec::new(),
            width: width,
            height: height,
            num_particles: 0,
            num_cells: 0,
            num_plants: 0,
            avg_age: 0.0,
            boundary: Rect::new((width / 2) as f32, (height / 2) as f32, (width / 2) as f32, (height / 2) as f32),
            center: Vector2::new((width as f32) / 2.0, (height as f32) / 2.0),
            qt: QuadTree::new(Rect::new((width as f32) / 2.0, (height as f32) / 2.0, (width as f32) / 2.0, (height as f32) / 2.0), 5),
            pending_deletions: Vec::new(),
            pending_additions: Vec::new(),
        }
    }
    pub fn reset_avgs(&mut self) {
        self.avg_speed = 0.0;
        self.avg_brain_size = 0.0;
        self.avg_hunger = 0.0;
        self.avg_isolation = 0.0;
        self.avg_social = 0.0;
        self.avg_sight_r = 0.0;
        self.avg_pred = 0.0;
        self.avg_thresh = 0.0;
        self.avg_age = 0.0;
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
        for i in 0..(self.num_particles as usize) {
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
        self.is_plant.push(false); // Default to non-plant
        self.num_particles = self.num_particles + 1;
    }

    pub fn add_cell(&mut self, pos: Vector2<f32>, mass: f32, radius: f32, vel: Vector2<f32>) {
        let cell_index = self.num_particles as usize;
        
        self.positions.push(pos);
        self.old_positions.push(pos - vel);
        self.accelerations.push(Vector2::new(0.0, 0.0));
        self.masses.push(mass);
        self.radii.push(radius);
        self.is_plant.push(false);
        
        self.cells.push(Cell::random(&mut self.rng, mass, cell_index));
        self.cell_indices.push(cell_index);
        
        self.num_cells = self.num_cells + 1;
        self.num_particles = self.num_particles + 1;
    }

    pub fn add_plant(&mut self, pos: Vector2<f32>, mass: f32, radius: f32, vel: Vector2<f32>) {
        let plant_index = self.num_particles as usize;
        
        self.positions.push(pos);
        self.old_positions.push(pos - vel);
        self.accelerations.push(Vector2::new(0.0, 0.0));
        self.masses.push(mass);
        self.radii.push(radius);
        self.is_plant.push(true);
        
        self.plants.push(plant_index);
        
        self.num_plants = self.num_plants + 1;
        self.num_particles = self.num_particles + 1;
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
    
            // Current motion vector (pos - old_pos is displacement over last step)
            let mut displacement = pos - old_pos;
    
            // --- Apply drag (velocity-proportional) ---
            // Drag reduces displacement proportionally to its magnitude
            displacement *= 1.0 - drag * dt;
    
            // --- Apply friction (constant opposing motion) ---
            // Friction always reduces speed by a fixed amount per second
            let speed = displacement.magnitude();
            if speed > 0.0 {
                let friction_amount = friction * dt;
                let new_speed = (speed - friction_amount).max(0.0);
                displacement *= new_speed / speed;
            }
    
            // Acceleration = existing + gravity
            let acc = self.accelerations[i] + grav;
    
            // Verlet integration
            let new_pos = pos + displacement + acc * dt * dt;
    
            // Store old pos for next iteration
            self.old_positions[i] = pos;
            self.positions[i] = new_pos;
    
            // Clear acceleration for next step
            self.accelerations[i] = Vector2::new(0.0, 0.0);
    
            // Radius based on mass
            self.radii[i] = f32::max(self.masses[i].sqrt() * 2.0, 2.0);
        }
    }
    
    pub fn apply_springs(&mut self, _k: f32) {
        let spring_connections = self.spring_connections.clone(); // Drop the borrow
        for pair in &spring_connections {
            
            let p1 = pair[0];
            let p2 = pair[1];
    
            if p1 >= self.num_particles as usize || p2 >= self.num_particles as usize {
                continue;
            }
    
            let pos1 = self.positions[p1];
            let pos2 = self.positions[p2];
    
            let axis = pos1 - pos2;
            let dist = axis.magnitude();
    
            let target_distance: f32 = self.radii[p1] + self.radii[p2];
            
            if dist != target_distance {
                // Hard constraint: force distance to sum of radii
                let normal = if dist > 0.0 {
                    axis / dist
                } else {
                    // If particles are at same position, use arbitrary direction
                    Vector2::new(1.0, 0.0)
                };
                
                // For velocity Verlet, we need to update previous positions
                // to maintain velocity consistency
                let old_pos1 = self.old_positions[p1];
                let old_pos2 = self.old_positions[p2];
                
                // Calculate new positions to maintain exact distance
                let center = (pos1 + pos2) / 2.0;
                let half_target = target_distance / 2.0;
                
                let new_pos1 = center + normal * half_target;
                let new_pos2 = center - normal * half_target;
                
                // Calculate how much each particle moved
                let delta1 = new_pos1 - pos1;
                let delta2 = new_pos2 - pos2;
                
                // Update positions
                self.positions[p1] = new_pos1;
                self.positions[p2] = new_pos2;
                
                // Update old positions to maintain velocity in Verlet integration
                // old_pos = new_pos - velocity * dt, so we preserve the velocity component
                self.old_positions[p1] = old_pos1 + delta1;
                self.old_positions[p2] = old_pos2 + delta2;
            }
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

                    if !self.is_plant[i] && !self.is_plant[n as usize] {
                        let avg_mass = (self.masses[i] + self.masses[n as usize]) / 2.0;
                        self.masses[i] = avg_mass;
                        self.masses[n as usize] = avg_mass;
                    }
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
        let buffer = 20.0;
        let expanded_rect = Rect::new(rectangle.x, rectangle.y, rectangle.w + buffer, rectangle.h + buffer);
        let push_force = 5000.0; // Adjust this value to control push strength
    
        let mut forces_to_apply = Vec::new();
    
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
                self.masses[i as usize]*= 0.999;
                if self.rng.random_range(0.0..1.0) < 0.0 {
                    self.pending_deletions.push(i as usize);
                } else {
                    //forces_to_apply.push((i as usize, Vector2::new(0.0, push_force)));
                }
            // Bottom boundary
            } else if pos.y + r > y_bottom {
                pos.y = y_bottom - r;
                self.masses[i as usize]*= 0.999;
                if self.rng.random_range(0.0..1.0) < 0.0 {
                    self.pending_deletions.push(i as usize);
                } else {
                    //forces_to_apply.push((i as usize, Vector2::new(0.0, -push_force)));
                }
            }
    
            // Left boundary
            if pos.x - r < x_left {
                pos.x = x_left + r;
                self.masses[i as usize]*= 0.999;
                if self.rng.random_range(0.0..1.0) < 0.0 {
                    self.pending_deletions.push(i as usize);
                } else {
                    //forces_to_apply.push((i as usize, Vector2::new(push_force, 0.0)));
                }
            // Right boundary
            } else if pos.x + r > x_right {
                pos.x = x_right - r;
                self.masses[i as usize]*= 0.999;
                if self.rng.random_range(0.0..1.0) < 0.0 {
                    self.pending_deletions.push(i as usize);
                } else {
                    //forces_to_apply.push((i as usize, Vector2::new(-push_force, 0.0)));
                }
            }
        }
    
        // Apply all forces after the loop to avoid borrow conflicts
        for (index, force) in forces_to_apply {
            self.accelerate_particle(index, force);
        }
    }

    // Safe deletion system - removes duplicates and sorts in reverse order
    fn process_deletions(&mut self) {
        if self.pending_deletions.is_empty() {
            return;
        }

        // Remove duplicates and sort in descending order for safe deletion
        self.pending_deletions.sort_unstable();
        self.pending_deletions.dedup();
        self.pending_deletions.reverse();

        let deletions_to_process: Vec<usize> = self.pending_deletions.clone();
        self.pending_deletions.clear();

        for index in deletions_to_process {
        if index < self.num_particles as usize {
            self.delete_particle_immediate(index);
        }
        }

        self.pending_deletions.clear();
    }

    // Immediate deletion - should only be called from process_deletions
    fn delete_particle_immediate(&mut self, index: usize) {
        if index >= self.num_particles as usize {
            return;
        }

        // Update counters
        if self.is_plant[index] {
            self.num_plants -= 1;
        } else {
            self.num_cells -= 1;
        }

        // Remove from vectors
        self.positions.remove(index);
        self.accelerations.remove(index);
        self.old_positions.remove(index);
        self.radii.remove(index);
        self.masses.remove(index);
        self.is_plant.remove(index);


        self.spring_connections.retain_mut(|connection| {
            // Remove connections that involve the deleted particle
            if connection.x == index || connection.y == index {
                return false; // Remove this connection
            }
            
            // Update indices for particles that come after the deleted one
            if connection.x > index {
                connection.x -= 1;
            }
            if connection.y > index {
                connection.y -= 1;
            }
            
            true // Keep this connection
        });

        // Update cell indices and remove cells
        for i in (0..self.cell_indices.len()).rev() {
            let idx = self.cell_indices[i];
            if idx == index {
                self.cell_indices.remove(i);
                self.cells.remove(i);
            } else if idx > index {
                self.cell_indices[i] = idx - 1;
                if i < self.cells.len() {
                    self.cells[i].index = idx - 1;
                }
            }
        }

        // Update plant indices
        for i in (0..self.plants.len()).rev() {
            let idx = self.plants[i];
            if idx == index {
                self.plants.remove(i);
            } else if idx > index {
                self.plants[i] = idx - 1;
            }
        }

        self.num_particles -= 1;
    }

    fn process_additions(&mut self) {
        for (child_pos, child_mass, brain_delta, child_brain_size, child_sight_r, child_sight_a, child_pred, child_max_speed,child_adhesion, child_thresh, parent_cell) in self.pending_additions.drain(..) {
            
            let child_index = self.num_particles as usize;
            
            // Add physical properties
            self.positions.push(child_pos);
            self.old_positions.push(child_pos);
            self.accelerations.push(Vector2::new(0.0, 0.0));
            self.masses.push(child_mass * 0.25);
            self.radii.push(child_mass);
            self.is_plant.push(false);
            self.num_particles += 1;
            self.num_cells += 1;
    
            // Ensure child_brain_size is valid and positive
            let safe_child_brain_size = child_brain_size.max(1);
            let input_size = (65 + safe_child_brain_size) as usize;
            
            // Create child cell based on brain_delta
            
            let child_cell = Cell {
                    index: child_index,
                    brain_size: safe_child_brain_size,
                    current_energy: child_mass,
                    state_encoder: parent_cell.state_encoder.mutate(&mut self.rng),
                    social_decoder: parent_cell.social_decoder.mutate(&mut self.rng),
                    hunger_decoder: parent_cell.hunger_decoder.mutate(&mut self.rng),
                    Brain: parent_cell.Brain.mutate(&mut self.rng),
                    isolation_decoder: parent_cell.isolation_decoder.mutate(&mut self.rng),
                    metabolic_rate: 0.0,
                    current_mass: child_mass / 4.0,
                    birth_threshold: clamp(child_thresh, 0.5, 1.0),
                    max_mass: child_mass,
                    max_speed: child_max_speed,
                    old_h: 0.0,
                    old_iso: 0.0,
                    old_soc: 0.0,
                    age: 0.0,
                    generation: parent_cell.generation + 1,
                    adhesion: child_adhesion,
                    // Use safe_child_brain_size here
                    old_encoding: nalgebra::DMatrix::<f32>::zeros(safe_child_brain_size as usize, 1),
                    sight_r: child_sight_r,
                    sight_a: child_sight_a,
                    desired_energy: child_mass * 1.5,
                    predation: clamp(child_pred, 0.0, 1.0),
                    to_delete: false,
                };
    
            self.cells.push(child_cell);
            self.cell_indices.push(child_index);

            if self.rng.random_range(0.0..1.0) < parent_cell.adhesion {
                self.spring_connections.push(Vector2::new(parent_cell.index, child_index));
            }
        }
    }

    // Mark cells for deletion that have the to_delete flag set
    fn mark_dead_cells(&mut self) {
        for i in 0..self.cells.len() {
            if self.cells[i].to_delete {
                self.pending_deletions.push(self.cells[i].index);
            }
        }
    }

    pub fn init_world(&mut self, num_entities: usize, plant_ratio: f32) {
        let points = Poisson2D::new()
            .with_dimensions([(self.height - 2 * 10) as f64, (self.width - 2 * 10) as f64], 30.0)
            .iter()
            .take(num_entities);
        
        for point in points {
            let pos: Vector2<f32> = Vector2::new((point[0] + 10.0) as f32, (point[1] + 10.0) as f32);
            let mass = self.rng.random_range(6.0..20.0);
            
            if self.rng.random_range(0.0..1.0) < plant_ratio {
                self.add_plant(pos, mass, mass, Vector2::new(0.0, 0.0));
            } else {
                self.add_cell(pos, mass ,  mass, Vector2::new(0.0, 0.0));
            }
        }
    }

    pub fn random_spawn_plant_cluster(&mut self, n: usize, cluster_radius: f32) {
        // Choose a random center point for the cluster
        let center: Vector2<f32> = Vector2::new(
            self.rng.random_range(cluster_radius..(self.width as f32 - cluster_radius)),
            self.rng.random_range(cluster_radius..(self.height as f32 - cluster_radius))
        );
        
        for _ in 0..n {
            // Generate random offset within cluster radius
            let angle = self.rng.random_range(0.0..std::f32::consts::TAU);
            let distance = self.rng.random_range(0.0..cluster_radius);
            
            let offset = Vector2::new(
                angle.cos() * distance,
                angle.sin() * distance
            );
            
            let pos = center + offset;
            
            // Ensure the position stays within bounds
            let clamped_pos = Vector2::new(
                pos.x.clamp(0.0, self.width as f32),
                pos.y.clamp(0.0, self.height as f32)
            );
            
            let mass = self.rng.random_range(6.0..10.0);
            self.add_plant(clamped_pos, mass, mass, Vector2::new(0.0, 0.0));
        }
    }

    pub fn update(&mut self, dt: f32, substeps: i32, grav: Vector2<f32>) {
        self.update_quadtree();
        
        for _ in 0..substeps {
            // Move cells out to avoid borrow conflicts
            
            let mut cells = std::mem::take(&mut self.cells);
            for cell in &mut cells {
                cell.timestep(self);
            }
            self.cells = cells;
    
            // Mark dead cells for deletion
            self.mark_dead_cells();
    
            // Process all deletions first
            self.process_deletions();
    
            // Then process additions
            self.process_additions();
    
            // Update quadtree after deletions/additions
            self.update_quadtree();
    
            // Update physics
            
            

            self.inter_particle_collisions();
           
            self.apply_rect_constraint(self.boundary);
            self.integrate_forces(dt / (substeps as f32), grav, 1.50, 15.0);
            //self.apply_springs(0.0);
        }

        
        let growth_probability: f32 = 0.0075; // or use linear: base_rate * self.num_cells as f32
        if self.rng.random_range(0.0..1.0) < growth_probability.min(1.0) && self.num_cells < 200 {
            self.random_spawn_plant_cluster(10,100.0);
        }
        if self.num_cells == 0 {
            self.init_world(10, 0.0);
        }
    }

    pub fn save_random_creature(&mut self) {
        let index = self.rng.random_range(0..self.num_cells as usize);
        let cell = &self.cells[index];
        cell.save_brain_to_file("brain.json").expect("Failed to save brain");
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}