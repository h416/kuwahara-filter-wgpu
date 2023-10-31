
const SECTOR_COUNT = 8;

@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var output_texture : texture_storage_2d<rgba8unorm, write>;

@group(1) @binding(0) var<uniform> filter_size : u32;
@group(1) @binding(1) var<uniform> q_value : f32;
@group(1) @binding(2) var<storage, read> kernel : array<f32>;
@group(1) @binding(3) var<storage, read> sector_map : array<u32>;

@compute
@workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) global_id : vec3<u32>,
) {

  let dimensions = textureDimensions(input_texture);
  let coords = vec2<i32>(global_id.xy);
  let cx = coords.x;
  let cy = coords.y;
  let w = i32(dimensions.x);
  let h = i32(dimensions.y);
  let n = i32(filter_size);
  let q = q_value;
  let sector_map_size = 2 * n - 1;

  
  var sdvs = array<f32,SECTOR_COUNT>();
  var avg_colors = array<vec4<f32>,SECTOR_COUNT>();
  let eps = 1e-3;
    
  var sdv_min = 0.0;
  var d_value = 1u;
  for (var d : i32 = 0; d < SECTOR_COUNT; d = d + 1) {
    var sum_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var sum_color2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight_sum = 0.0;
    for (var yy : i32 = 0; yy < sector_map_size; yy = yy + 1) {
      for (var xx : i32 = 0; xx < sector_map_size; xx = xx + 1) {
      
        let x2 = xx + cx - (n - 1);
        let y2 = yy + cy - (n - 1);
        let dx = x2 - cx;
        let dy = y2 - cy;
        let dist2 = dx * dx + dy * dy;

        let map_index = xx + sector_map_size * yy;
        // let map_value = sector_map.values[map_index];
        let map_value = sector_map[map_index];
        // let in_sector = dist2 == 0 || map_value == d_value;
        let in_sector = dist2 == 0 || map_value == d_value;
        
        if in_sector && dist2 <= n * n && y2 >= 0 && y2 < h && x2 >= 0 && x2 < w {
          let weight = kernel[dist2];
          let value = textureLoad(input_texture, vec2<i32>(x2, y2), 0);
          let weited_value = value * weight;
          sum_color += weited_value;
          sum_color2 += weited_value * value;
          weight_sum += weight;
        }
      } // xx
    } // yy

    if weight_sum == 0.0 {
      weight_sum = 1.0;
    }

    let weight_sum_i = 1.0 / weight_sum;
    let avg_color = sum_color * weight_sum_i;

    let var_color = sum_color2 * weight_sum_i - avg_color * avg_color;
    let sdv = length(var_color);
    if d == 0 || sdv < sdv_min {
      sdv_min = sdv;
    }
    sdvs[d] = sdv;
    avg_colors[d] = avg_color;

    d_value = d_value + d_value;
  } // d
  
  var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var weight_sum = 0.0;
  let mq = -q;
   let mq_sdv_max = mq * log(sdv_min + eps);
  for (var d : i32 = 0; d < SECTOR_COUNT; d = d + 1) {
    let sdv = sdvs[d];
    // let sdv_q = pow(sdv, q);
    // let weight = 1.0 / (sdv_q + eps);

    // weight = 1/sdv^q = sdv ^ -q
    // = pow(sdv, -q) = exp(-q*log(sdv))
    //    pow(x, y) = exp(y*log(x))
    let mq_sdv = mq * log(sdv + eps);
    // let weight = f32::exp(mq_sdv);
    // to avoid cancelation divide weight by exp(mq_sdv_max)
    let weight = exp(mq_sdv - mq_sdv_max);

    color += weight * avg_colors[d];
    weight_sum += weight;
  }

  if weight_sum == 0.0 {
      weight_sum = 1.0;
  }
  color = color/ weight_sum;

  
  textureStore(output_texture, coords.xy, color);
}