
@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var output_texture : texture_storage_2d<rgba8unorm, write>;

@group(1) @binding(0) var<uniform> sigma_value : f32;

const PI: f32 = 3.14159265359;

fn gaussian_2d(x: i32, y: i32, sigma: f32) -> f32 {
    let s2 = 2.0 * sigma * sigma;
    let norm = 1.0 / (PI * s2);
    let l2 = f32 (x*x+y*y);
    return norm * exp( -l2/s2) ;
}
        
 fn blur(coords: vec2i, sigma: f32 ) -> vec3f {
    let kernel_radius = i32(sigma*4.0);
    var col = vec3f(0.0);
    var weight_sum = 0.0;
    for (var y = -kernel_radius; y <= kernel_radius; y=y+1) {
      for (var x = -kernel_radius; x <= kernel_radius; x=x+1) {
        let c = textureLoad(input_texture, coords + vec2(x, y), 0).rgb;
        let weight = gaussian_2d(x, y, sigma);
        col += c * weight;
        weight_sum += weight;
      }
    }
    if (weight_sum==0.0) {
      weight_sum=1.0;
    }
    return col / weight_sum;
}


@compute
@workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) global_id : vec3<u32>,
) {
  let coords = vec2i(global_id.xy);
  let sigma = sigma_value;
  var val :vec3f;
  if(sigma == 0.0) {
    val =  textureLoad(input_texture, coords, 0).rgb;
  }else {
    val = blur(coords, sigma);
  }
  textureStore(output_texture, coords.xy, vec4f(val.x, val.y, val.z, 1.0));
}