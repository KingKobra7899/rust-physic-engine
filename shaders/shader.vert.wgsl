// shader.vert.wgsl
// This vertex shader outputs clip-space position and UV coordinates for a full-screen quad.

struct VertexInput {
    @location(0) position: vec2<f32>, // Clip-space position from vertex buffer
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>, // UV coordinates (0 to 1) for the fullscreen quad
};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    // The input `position` is already in clip space (-1 to 1).
    // Set z to 0 and w to 1 for 2D rendering.
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0); 
    
    // Map clip space (-1 to 1) to UV space (0 to 1).
    // (pos.xy * 0.5) + 0.5 effectively transforms [-1, 1] to [0, 1].
    out.uv = (in.position * 0.5) + 0.5;
    return out;
}