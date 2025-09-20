// cell.rs - Fixed version
use std::f32::consts::PI;

use nalgebra::{clamp, DMatrix, DVector, Vector2};
use rand::{rngs::ThreadRng, seq::SliceRandom as _, Rng};
use rand_distr::{Normal, Distribution};
use crate::solver::{quadtree::Rect, PhysicsSolver};

const MUTATION_RATE: f64 = 0.15;

use serde::Serialize;

#[derive(Serialize)]
struct BrainExport {
    brain_size: i32,
    generation: i32,
    encoder_weights: Vec<f32>,
    encoder_bias: Vec<f32>,
    decoder_weights: Vec<f32>,
    decoder_bias: Vec<f32>,
    sight_r: f32,
    sight_a: f32,
    predation: f32,
    birth_threshold: f32,
    adhesion: f32,
    max_speed: f32
}

pub struct EnvironmentalEncoder {
    pub(crate) weight_matrix: DMatrix<f32>,
    pub(crate) bias: DMatrix<f32>,
}

impl EnvironmentalEncoder {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weight_matrix = DMatrix::<f32>::zeros(output_size, input_size);
        let bias = DMatrix::<f32>::zeros(output_size, 1);
        EnvironmentalEncoder { weight_matrix, bias }
    }

    pub fn add_neurons(&self, rng: &mut ThreadRng, new_neurons: usize) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let extra_weights = DMatrix::<f32>::from_fn(new_neurons, self.weight_matrix.ncols(), |_, _| {
            normal.sample(rng)
        });

        let extra_bias = DMatrix::<f32>::from_fn(new_neurons, 1, |_, _| {
            normal.sample(rng)
        });

        let new_weight_matrix = DMatrix::from_rows(
            &[
                self.weight_matrix.row_iter().collect::<Vec<_>>(),
                extra_weights.row_iter().collect::<Vec<_>>(),
            ]
            .concat(),
        );

        let new_bias = DMatrix::from_rows(
            &[
                self.bias.row_iter().collect::<Vec<_>>(),
                extra_bias.row_iter().collect::<Vec<_>>(),
            ]
            .concat(),
        );

        EnvironmentalEncoder {
            weight_matrix: new_weight_matrix,
            bias: new_bias,
        }
    }

    pub fn remove_neurons(&self, rng: &mut ThreadRng, num_remove: usize) -> Self {
        let total = self.weight_matrix.nrows();
        if num_remove >= total {
            panic!("Cannot remove all neurons from EnvironmentalEncoder");
        }

        let mut indices: Vec<usize> = (0..total).collect();
        indices.shuffle(rng);
        indices.truncate(total - num_remove);
        indices.sort_unstable();

        let new_weight_matrix = DMatrix::<f32>::from_rows(
            &indices.iter().map(|&i| self.weight_matrix.row(i)).collect::<Vec<_>>(),
        );

        let new_bias = DMatrix::<f32>::from_rows(
            &indices.iter().map(|&i| self.bias.row(i)).collect::<Vec<_>>(),
        );

        EnvironmentalEncoder {
            weight_matrix: new_weight_matrix,
            bias: new_bias,
        }
    }

    pub fn random(input_size: usize, output_size: usize, rng: &mut ThreadRng) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let weight_matrix = DMatrix::<f32>::from_fn(output_size, input_size, |_, _| {
            normal.sample(rng)
        });
        
        let bias = DMatrix::<f32>::from_fn(output_size, 1, |_, _| {
            normal.sample(rng)
        });
        
        EnvironmentalEncoder { weight_matrix, bias }
    }

    pub fn mutate(&self, rng: &mut ThreadRng) -> Self {
        let normal = Normal::new(0.0, 0.5).unwrap();
        
        let new_weights: DMatrix<f32> = self.weight_matrix.map(|x| {
            if rng.random_bool(MUTATION_RATE) {
                x + normal.sample(rng)
            } else {
                x
            }
        });

        let new_bias: DMatrix<f32> = self.bias.map(|x| {
            if rng.random_bool(MUTATION_RATE) {
                x + normal.sample(rng)
            } else {
                x
            }
        });

        EnvironmentalEncoder { weight_matrix: new_weights, bias: new_bias }
    }

    pub fn with_weights(weight_matrix: DMatrix<f32>, bias: DMatrix<f32>) -> Self {
        EnvironmentalEncoder { weight_matrix, bias }
    }

    pub fn encode(&self, input: &DMatrix<f32>) -> DMatrix<f32> {
        DMatrix::map(&(self.weight_matrix.clone() * input + &self.bias), |x| x.tanh())
    }
}

pub struct DirMovementEncoder {
    pub(crate) encoder: EnvironmentalEncoder,
    pub(crate) decoder: EnvironmentalEncoder
}

impl DirMovementEncoder {
    pub fn random (rng: &mut ThreadRng, brain_size: usize) -> Self {
        DirMovementEncoder{
            encoder: EnvironmentalEncoder::random(71 + brain_size, brain_size, rng),
            decoder: EnvironmentalEncoder::random(brain_size, 2, rng)

        }
    }

    pub fn mutate (&self, rng: &mut ThreadRng) -> Self{
        DirMovementEncoder{
            encoder: self.encoder.mutate(rng),
            decoder: self.decoder.mutate(rng)

        }
    }

    pub fn get_movement_vector(&self, input: &DMatrix<f32>, max_speed: f32) -> Vector2<f32> {
        let base_output = self.decoder.encode(&self.encoder.encode(&input));
        
        let mut theta = (base_output[0] + 1.0) / 2.0;
        theta *= 2.0 * PI;
        let r = max_speed * (base_output[1] + 1.0) / 2.0;

        
        return Vector2::new( r * theta.cos(), r * theta.sin());
    }
}

pub struct CognitiveDecoder {
    pub(crate) weights: DVector<f32>,
}

impl CognitiveDecoder {
    pub fn new(input_size: usize) -> Self {
        let weights = DVector::<f32>::zeros(input_size);
        CognitiveDecoder { weights }
    }

    pub fn remove_neurons(&self, rng: &mut ThreadRng, num_remove: usize) -> Self {
        let total = self.weights.len();
        if num_remove >= total {
            panic!("Cannot remove all neurons from CognitiveDecoder");
        }

        let mut indices: Vec<usize> = (0..total).collect();
        indices.shuffle(rng);
        indices.truncate(total - num_remove);
        indices.sort_unstable();

        let new_weights = DVector::<f32>::from_iterator(
            indices.len(),
            indices.iter().map(|&i| self.weights[i]),
        );

        CognitiveDecoder { weights: new_weights }
    }

    pub fn with_weights(weights: DVector<f32>) -> Self {
        CognitiveDecoder { weights }
    }

    pub fn random(input_size: usize, rng: &mut ThreadRng) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let weights = DVector::from_fn(input_size, |_, _| {
            normal.sample(rng)
        });

        CognitiveDecoder { weights }
    }

    pub fn add_neurons(&self, rng: &mut ThreadRng, new_neurons: usize) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let extra_weights = DVector::<f32>::from_fn(new_neurons, |_, _| {
            normal.sample(rng)
        });

        let mut combined = DVector::<f32>::zeros(self.weights.len() + new_neurons);
        combined.rows_mut(0, self.weights.len()).copy_from(&self.weights);
        combined.rows_mut(self.weights.len(), new_neurons).copy_from(&extra_weights);

        CognitiveDecoder { weights: combined }
    }

    pub fn mutate(&self, rng: &mut ThreadRng) -> Self {
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        let new_weights: DVector<f32> = self.weights.map(|x| {
            if rng.random_bool(MUTATION_RATE) {
                x + normal.sample(rng)
            } else {
                x
            }
        });

        CognitiveDecoder { weights: new_weights }
    }

    pub fn decode(&self, input: &DVector<f32>) -> f32 {
        f32::tanh(self.weights.dot(input)) + 1.0 // Ensure output is positive
    }
}

pub struct Cell {
    pub(crate) index: usize,
    pub(crate) brain_size: i32,
    pub(crate) current_energy: f32,
    pub(crate) state_encoder: EnvironmentalEncoder,
    pub social_decoder: CognitiveDecoder,
    pub hunger_decoder: CognitiveDecoder,
    pub isolation_decoder: CognitiveDecoder,
    pub Brain: DirMovementEncoder,
    pub birth_threshold: f32,
    pub metabolic_rate: f32,
    pub current_mass: f32,
    pub max_mass: f32,
    pub max_speed: f32,
    pub age: f32,
    pub old_h: f32,
    pub old_iso: f32,
    pub adhesion: f32,
    pub old_soc: f32,
    pub old_encoding: DMatrix<f32>,
    pub sight_r: f32,
    pub sight_a: f32,
    pub desired_energy: f32,
    pub generation: i32,
    pub predation: f32,
    pub to_delete: bool, // New field to mark for deletion
}

impl Cell {
    pub fn random(rng: &mut ThreadRng, mass: f32, index: usize) -> Self {
        let sight_r_dist: Normal<f32> = Normal::new(50.0, 10.0).unwrap();
        let sight_angle_dist: Normal<f32> = Normal::new(PI / 4.0, PI / 12.0).unwrap();
        let brain_size_dist: Normal<f32> = Normal::new(6.0, 1.5).unwrap();
        let predation_dist: Normal<f32> = Normal::new(0.5, 0.1).unwrap();

        let brain_size: i32 = brain_size_dist.sample(rng) as i32;
        let max_speed = rng.random_range(0.1..0.5);

        Cell {
            index,
            brain_size,
            current_energy: mass,
            state_encoder: EnvironmentalEncoder::random((71 + brain_size) as usize, brain_size as usize, rng),
            social_decoder: CognitiveDecoder::random(brain_size as usize, rng),
            hunger_decoder: CognitiveDecoder::random(brain_size as usize, rng),
            isolation_decoder: CognitiveDecoder::random(brain_size as usize, rng),
            Brain: DirMovementEncoder::random(rng, brain_size as usize),
            metabolic_rate: 0.0,
            age: 0.0,
            old_h: 0.0,
            old_iso: 0.0,
            old_soc: 0.0,
            generation: 1,
            old_encoding: DMatrix::<f32>::zeros((brain_size) as usize, 1),
            current_mass: mass / 2.0,
            max_mass: mass,
            max_speed: max_speed,
            sight_r: sight_r_dist.sample(rng),
            sight_a: sight_angle_dist.sample(rng),
            desired_energy: mass*1.5,
            predation: predation_dist.sample(rng),
            adhesion: predation_dist.sample(rng),
            birth_threshold: clamp(predation_dist.sample(rng) + 0.25, 0.5, 1.0),
            to_delete: false,
        }
    }

    pub fn create_child(&self, world: &mut PhysicsSolver) {
        let mut child_pos: Vector2<f32> = world.positions[self.index];
        let mut child_brain_size = self.brain_size;
        let mut child_mass = self.max_mass;
        let mut child_sight_r = self.sight_r;
        let mut child_sight_a = self.sight_a;
        let mut child_pred = self.predation;
        let mut child_adhesion = self.adhesion;
        let mut child_max_speed = self.max_speed;
        let mut child_thresh = self.birth_threshold;
        let brain_delta: i32 = 0;
    
        // Reduced mutation magnitudes for stability
        if world.rng.random_range(0.0..1.0) < (MUTATION_RATE) {
            child_mass += clamp(world.rng.random_range(-0.5..0.5) as f32, 5.0, 100.0); // Smaller changes
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_sight_r += world.rng.random_range(-1.0..1.0) as f32; // Smaller changes
            child_sight_r = clamp(child_sight_r, 15.0, 60.0);
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_sight_a += world.rng.random_range(-0.02..0.02) as f32; // Smaller changes
            child_sight_a = clamp(child_sight_a, PI/12.0, PI/3.0);
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_pred += world.rng.random_range(-0.005..0.005) as f32; // Much smaller
            child_pred = clamp(child_pred, 0.1, 0.7);
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_thresh += world.rng.random_range(-0.005..0.005) as f32; // Much smaller
            child_thresh = clamp(child_thresh, 0.65, 0.9);
        }
        
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_max_speed += world.rng.random_range(-0.005..0.005); // Much smaller
            child_max_speed = clamp(child_max_speed, 0.05, 0.5);
        }
    
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_adhesion += world.rng.random_range(-0.02..0.02); // Smaller
            child_adhesion = clamp(child_adhesion, 0.1, 0.8);
        }
    
        // More conservative post-birth energy cost
        let post_birth_mass = self.current_mass * 0.6; // Less harsh
    
        let r = 2.0 * post_birth_mass.sqrt();
        child_pos += Vector2::new(r * 2.5, 0.0);
    
        world.pending_additions.push((
            child_pos, child_mass, brain_delta, child_brain_size, 
            child_sight_r, child_sight_a, child_pred, child_max_speed, 
            child_adhesion, child_thresh, self.clone()
        ));
    }

    pub fn encode_environment(&self, world: &PhysicsSolver) -> DMatrix<f32> {
        let pos: Vector2<f32> = world.positions[self.index];
        let vel: Vector2<f32> = world.positions[self.index] - world.old_positions[self.index];
        let point_idx: Vec<i32> = world.qt.query_cone(pos, vel, self.sight_r, self.sight_a);
        let mut full_env = DMatrix::<f32>::zeros((self.brain_size + 71) as usize, 1);

        let mut indices_to_shuffle: Vec<usize> = (0..point_idx.len().min(20)).collect();

        let mut rng = rand::rng();
        indices_to_shuffle.shuffle(&mut rng);

        for (i, &shuffled_i) in indices_to_shuffle.iter().enumerate() {
            let idx = point_idx[shuffled_i];
            let other_pos = world.positions[idx as usize];
            let dx = (pos.x - other_pos.x) / self.sight_r;
            let dy = (pos.y - other_pos.y) / self.sight_r;

            let base = i * 3;
            full_env[(base, 0)] = dx;
            full_env[(base + 1, 0)] = dy;
            full_env[(base + 2, 0)] = world.is_plant[idx as usize] as i32 as f32;
        }


        
        full_env[60] = (self.desired_energy - self.current_energy) / self.desired_energy;
        full_env[61] = self.metabolic_rate;
        let vel = world.positions[self.index] - world.old_positions[self.index];

        full_env[62] = vel.magnitude() / self.max_speed;
        full_env[63] = vel.angle(&Vector2::new(0.0, 1.0)) / 2.0 * PI;
        full_env[64] = self.current_mass / self.max_mass;
        

        // Only include distance if edge is within sight radius, otherwise set to max
        let pos = world.positions[self.index];

        full_env[65] = ((pos.y - (world.boundary.y - world.boundary.h)) / (2.0 * world.boundary.h)).min(1.0);
        full_env[66] = (((world.boundary.y + world.boundary.h) - pos.y) / (2.0 * world.boundary.h)).min(1.0);
        full_env[67] = ((pos.x - (world.boundary.x - world.boundary.w)) / (2.0 * world.boundary.w)).min(1.0);
        full_env[68] = (((world.boundary.x + world.boundary.w) - pos.x) / (2.0 * world.boundary.w)).min(1.0);



        full_env[69] = pos.x / world.boundary.w / 2.0;
        full_env[70] = pos.y / world.boundary.h / 2.0;
        full_env
            .view_mut((self.brain_size as usize, 0), (self.brain_size as usize, 1))
            .copy_from(&self.old_encoding);

        for val in full_env.iter_mut() {
            if !val.is_finite() {
                *val = 0.0;
            }
        }

        full_env
    }

    pub fn timestep(&mut self, world: &mut PhysicsSolver) {
        // Sync mass with world
        self.current_mass = world.masses[self.index];
        
        // Get environmental information and movement decision
        let internal_rep = self.encode_environment(world);
        let movement_vec: Vector2<f32> = self.Brain.get_movement_vector(&internal_rep, self.max_speed);
        self.old_encoding = self.Brain.encoder.encode(&internal_rep);
        // Get current position and velocity for sight calculations
        let pos: Vector2<f32> = world.positions[self.index];
        let vel: Vector2<f32> = world.positions[self.index] - world.old_positions[self.index];
        let _point_idx: Vec<i32> = world.qt.query_cone(pos, vel, self.sight_r, self.sight_a);
    
        // ----- Eating Logic -----
        let eating_range = world.radii[self.index];
        let search_radius = self.sight_r; // More reasonable search radius
        let eating_points = world.qt.query(&Rect::new(
            pos.x - search_radius, 
            pos.y - search_radius, 
            search_radius * 2.0, 
            search_radius * 2.0
        ));
    
        for &idx in &eating_points {
            if idx == self.index as i32 {
                continue;
            }
    
            let target_pos = world.positions[idx as usize];
            let distance = (pos - target_pos).magnitude();
            let contact_distance = eating_range + world.radii[idx as usize];
    
            if distance <= contact_distance {
                let is_plant = world.is_plant[idx as usize];
                let target_mass = world.masses[idx as usize];
                
                // Can eat if it's a plant, or if predation succeeds and target is smaller
                let can_eat_animal = !is_plant && 
                                    (self.predation > world.rng.random_range(0.0..1.0)) && 
                                    (self.current_mass > target_mass);
                
                // Prevent eating things that are too large relative to self
                let size_constraint = target_mass < self.current_mass * 3.0;
    
                if (is_plant || can_eat_animal) && size_constraint && world.rng.random_bool(0.333) {
                    // Energy gain based on target's mass
                    let energy_conversion_rate = if is_plant { 15.0 } else { 25.0 };
                    let energy_gain = (target_mass / 2.0) * energy_conversion_rate;

                    self.current_mass += target_mass / 3.0;
                    self.current_energy += energy_gain;
                    
                    // Mark for deletion (avoid double-processing)
                    world.pending_deletions.push(idx as usize);
                    break; // Only eat one thing per timestep
                }
            }
        }
    
        // ----- Movement -----
        world.positions[self.index] += movement_vec;
    
        // ----- Metabolic Calculations -----
        // Basal metabolic rate using Kleiber's law (3/4 power scaling)
        let basal_cost = 0.008 * self.current_mass.powf(0.7); // Reduced and gentler scaling
        let brain_cost = 0.001 * (self.brain_size as f32).powf(0.75); // Reduced brain cost
        
        let movement_magnitude = movement_vec.magnitude();
        let move_cost = 0.002 * self.current_mass * movement_magnitude.powi(2); // Reduced movement cost
        
        self.metabolic_rate = basal_cost + brain_cost + move_cost;
        self.current_energy -= self.metabolic_rate;
    
        // ----- Energy ↔ Mass Exchange -----
        let starvation_threshold = self.desired_energy * 0.1;
        let surplus_threshold = self.desired_energy * 1.2;
        
        if self.current_energy < starvation_threshold {
            // Convert mass to energy when starving
            let mass_conversion_rate = 0.005;
            let mass_loss = mass_conversion_rate * self.current_mass;
            let energy_per_mass = 25.0;
            
            self.current_mass -= mass_loss;
            self.current_energy += mass_loss * energy_per_mass;
        } else if self.current_energy > surplus_threshold {
            // Convert surplus energy to mass
            let energy_conversion_rate = 0.0005;
            let energy_to_convert = energy_conversion_rate * (self.current_energy - self.desired_energy);
            let mass_per_energy = 1.0 / 25.0; // Inverse of energy_per_mass
            
            self.current_mass += energy_to_convert * mass_per_energy;
            self.current_energy -= energy_to_convert;
        }
    
        if self.current_energy < starvation_threshold {
            // Gentler mass-to-energy conversion
            let mass_conversion_rate = 0.003; // Reduced from 0.005
            let mass_loss = mass_conversion_rate * self.current_mass;
            let energy_per_mass = 30.0; // Higher efficiency
            
            self.current_mass -= mass_loss;
            self.current_energy += mass_loss * energy_per_mass;
        } else if self.current_energy > surplus_threshold {
            // More efficient energy-to-mass conversion
            let energy_conversion_rate = 0.001; // Increased from 0.0005
            let energy_to_convert = energy_conversion_rate * (self.current_energy - self.desired_energy);
            let mass_per_energy = 1.0 / 30.0; // Match the efficiency above
            
            self.current_mass += energy_to_convert * mass_per_energy;
            self.current_energy -= energy_to_convert;
        }
    
        // ----- Reproduction (TUNED) -----
        // More achievable reproduction conditions
        if self.current_mass >= self.max_mass * self.birth_threshold && 
           self.current_energy > self.desired_energy * (self.birth_threshold * 0.8) { // Easier energy requirement
            
            self.create_child(world);
            
            // Less harsh post-reproduction costs
            self.current_energy -= self.desired_energy * 0.3; // Reduced from 0.5
            self.current_mass -= self.max_mass * 0.3; // Reduced from 0.5
        }
    
        // Ensure mass stays within reasonable bounds
        self.current_mass = self.current_mass.max(0.01).min(self.max_mass);
        
        // Sync mass back to world
        world.masses[self.index] = self.current_mass;
    
        // ----- Update World Statistics -----
        let num_cells_f32 = world.num_cells as f32;
        world.avg_speed += self.max_speed / num_cells_f32;
        world.avg_brain_size += self.brain_size as f32 / num_cells_f32;
        
        if self.predation.is_finite() {
            world.avg_pred += self.predation / num_cells_f32;
        }
        if self.birth_threshold.is_finite() {
            world.avg_thresh += self.birth_threshold / num_cells_f32;
        }
        // Only update averages if values are finite
        if self.old_h.is_finite() {
            world.avg_hunger += self.old_h / num_cells_f32;
        }
        if self.old_iso.is_finite() {
            world.avg_isolation += self.old_iso / num_cells_f32;
        }
        if self.old_soc.is_finite() {
            world.avg_social += self.old_soc / num_cells_f32;
        }
        if self.age.is_finite() {
            world.avg_age += self.age / num_cells_f32;
        }
        
        world.avg_sight_r += self.sight_r / num_cells_f32;
    
        // ----- Death Conditions -----
        let min_mass_for_survival = 0.02;
        let min_energy_for_survival = 0.0;
        
        let position_invalid = world.positions[self.index].x.is_nan() || 
                              world.positions[self.index].y.is_nan() ||
                              world.positions[self.index].x.is_infinite() ||
                              world.positions[self.index].y.is_infinite();
        
        if self.current_energy <= min_energy_for_survival || 
           self.current_mass <= min_mass_for_survival || 
           position_invalid {
            self.to_delete = true;
        }
        self.age += 1.0;
    }
    pub fn export_brain(&self) -> BrainExport {
        BrainExport {
            brain_size: self.brain_size,
            generation: self.generation,
            encoder_weights: self.Brain.encoder.weight_matrix.as_slice().to_vec(),
            encoder_bias: self.Brain.encoder.bias.as_slice().to_vec(),
            decoder_weights: self.Brain.decoder.weight_matrix.as_slice().to_vec(),
            decoder_bias: self.Brain.decoder.bias.as_slice().to_vec(),
            birth_threshold: self.birth_threshold,
            adhesion: self.adhesion,
            sight_r: self.sight_r,
            sight_a: self.sight_a,
            predation: self.predation,
            max_speed: self.max_speed,
        }
    }

    /// Export brain to JSON string
    pub fn export_brain_json(&self) -> Result<String, serde_json::Error> {
        let brain_export = self.export_brain();
        serde_json::to_string_pretty(&brain_export)
    }

    /// Save brain to file
    pub fn save_brain_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.export_brain_json()?;
        let mut file = std::fs::File::create(filename)?;
        std::io::Write::write_all(&mut file, json.as_bytes())?;
        Ok(())
    }
}

impl Clone for Cell {
    fn clone(&self) -> Self {
        Cell {
            index: self.index,
            brain_size: self.brain_size,
            generation: self.generation,
            current_energy: self.current_energy,
            state_encoder: EnvironmentalEncoder::with_weights(self.state_encoder.weight_matrix.clone(), self.state_encoder.bias.clone()),
            social_decoder: CognitiveDecoder::with_weights(self.social_decoder.weights.clone()),
            hunger_decoder: CognitiveDecoder::with_weights(self.hunger_decoder.weights.clone()),
            isolation_decoder: CognitiveDecoder::with_weights(self.isolation_decoder.weights.clone()),
            Brain: DirMovementEncoder { encoder: EnvironmentalEncoder::with_weights(self.Brain.encoder.weight_matrix.clone(), self.Brain.encoder.bias.clone()), 
            decoder: EnvironmentalEncoder::with_weights(self.Brain.decoder.weight_matrix.clone(), self.Brain.decoder.bias.clone()) },
            metabolic_rate: self.metabolic_rate,
            birth_threshold: self.birth_threshold,
            current_mass: self.current_mass,
            max_mass: self.max_mass,
            max_speed: self.max_speed,
            age: self.age,
            old_h: self.old_h,
            old_iso: self.old_iso,
            old_soc: self.old_soc,
            adhesion: self.adhesion,
            old_encoding: self.old_encoding.clone(),
            sight_r: self.sight_r,
            sight_a: self.sight_a,
            desired_energy: self.desired_energy,
            predation: self.predation,
            to_delete: self.to_delete,
        }
    }
}