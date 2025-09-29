struct Particle {
    position: vec2<f32>,
    radius: f32,
    is_plant: u32
};

@group(0) @binding(0)
var<storage, read> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> num_particles_uniform: u32;

@group(1) @binding(0)
var<uniform> screen_dims: vec2<f32>;

struct FragmentInput {
    @builtin(position) frag_coord: vec4<f32>,
};

// Gamma correction for publication-quality colors
fn gamma_correct(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / 2.2));
}

// Crisp edge falloff for circles
fn circle_falloff(dist: f32, radius: f32) -> f32 {
    let norm_dist = dist / radius;
    // Sharper than default smoothstep for cleaner edge
    return 1.0 - smoothstep(0.94, 1.0, norm_dist);
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let pixel_coord = in.frag_coord.xy;
    var field = 0.0;

    for (var i: u32 = 0u; i < num_particles_uniform; i = i + 1u) {
        let p = particles[i];
        let dist = length(pixel_coord - p.position);
        // Squared falloff for smooth metaball blending
        let r2 = (p.radius * p.radius)/5;
        let d2 = dist * dist;
        field += r2 / d2;
    }

    // Use saturate to clamp and smooth the transition
    let alpha = saturate((field - 0.5) * 10.0);

    let bg_color = vec3<f32>(1.0, 1.0, 1.0);
    let fg_color = vec3<f32>(0.34, 0.7, 0.43);
    let final_color = mix(bg_color, fg_color, alpha);

    return vec4<f32>(final_color, 1.0);
}