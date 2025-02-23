
@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var output_texture : texture_storage_2d<rgba8unorm, write>;


 fn get_structure_tensor(coords: vec2i) -> vec3f {
    // sobel filter
    // c00 c01 c02
    // c10 c11 c12
    // c20 c21 c22
    let c00 = textureLoad(input_texture, coords + vec2(-1, -1), 0).rgb;
    let c01 = textureLoad(input_texture, coords + vec2(0,  -1), 0).rgb;
    let c02 = textureLoad(input_texture, coords + vec2(1,  -1), 0).rgb;
    
    let c10 = textureLoad(input_texture, coords + vec2(-1, 0), 0).rgb;
    // let c11 = textureLoad(input_texture, coords + vec2(0,  0), 0).rgb;
    let c12 = textureLoad(input_texture, coords + vec2(1,  0), 0).rgb;
    
    let c20 = textureLoad(input_texture, coords + vec2(-1, 1), 0).rgb;
    let c21 = textureLoad(input_texture, coords + vec2(0,  1), 0).rgb;
    let c22 = textureLoad(input_texture, coords + vec2(1 , 1), 0).rgb;
    
    let sx = 0.25 * ( (c00-c02) + 2.0*(c10-c12) + (c20-c22) );
    let sy = 0.25 * ( (c00-c20) + 2.0*(c01-c21) + (c02-c22) );
    return vec3(dot(sx, sx), dot(sy, sy), dot(sx, sy));
}


@compute
@workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) global_id : vec3<u32>,
) {
  let coords = vec2i(global_id.xy);  
  let t =  get_structure_tensor(coords);
  textureStore(output_texture, coords.xy, vec4f(t.x, t.y, t.z, 1.0));
}