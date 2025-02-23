
const SECTOR_COUNT = 8;
const eps = 1e-8;
    
@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var edge_texture : texture_2d<f32>;
@group(0) @binding(2) var output_texture : texture_storage_2d<rgba8unorm, write>;

@group(1) @binding(0) var<uniform> filter_size : f32;
@group(1) @binding(1) var<uniform> sharpness_value : f32;
@group(1) @binding(2) var<uniform> eccentricity_value : f32;

fn get_dominant_orientation(structure_tensor: vec3f) -> vec4f{
    let e = structure_tensor.x; 
    let f = structure_tensor.y; 
    let g = structure_tensor.z; 
    let e_g = e-g;

    let h = sqrt(e_g*e_g + 4.0 * f*f+ eps);
    let e_p_g = e+g;
    
    let lambda1 = 0.5 + (e_p_g+h) ;
    let lambda2 = 0.5 + (e_p_g-h) ;
    
    let v = vec2f(lambda1-e, -f);
    var t: vec2f;
    if (length(v) > 0.0) {
        t = normalize(v);
    } else {
        t = vec2f(0.0, 1.0);
    }
    let phi = -atan2(t.y, t.x);
    var A:f32;
    if (lambda1 + lambda2 > 0.0) {
        A = (lambda1 - lambda2) / (lambda1 + lambda2);
    } else {
        A = 0.0;
    }
    return vec4f(v, phi, A);
}

@compute
@workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) global_id : vec3<u32>,
) {

    let dimensions = textureDimensions(input_texture);
    let coords = vec2<i32>(global_id.xy);

    // params
    let q = sharpness_value;
    let kernel_radius = i32(filter_size);
    let alpha = eccentricity_value;
    let zeta = 2.0/ f32(kernel_radius);
    let zeroCross = 0.5875;//0.1; //0.58;//_ZeroCrossing; [Range(0.01f, 2.0f)] // 3.14/8 - 2*3.14/8 = 0.3925 - 0.785, 3.14/8 *1.5 = 0.5875
    
    let structure_tensor = textureLoad(edge_texture, coords, 0).rgb;
    let t = get_dominant_orientation(structure_tensor);

    //let a = f32(kernel_radius) * clamp((alpha + t.w) / alpha, 0.1, 2.0);
    //let b = f32(kernel_radius) * clamp(alpha / (alpha + t.w), 0.1, 2.0);
    let a = f32(kernel_radius) * (alpha + t.w) / alpha;
    let b = f32(kernel_radius) * alpha / (alpha + t.w);

    let cos_phi = cos(t.z);
    let sin_phi = sin(t.z);
    let R = mat2x2f(cos_phi, -sin_phi, sin_phi, cos_phi);
    let S = mat2x2f(0.5 / a, 0.0, 0.0, 0.5 / b);
    let SR = S*R;
    let max_x = i32(sqrt(a * a * cos_phi * cos_phi + b * b * sin_phi * sin_phi));
    let max_y = i32(sqrt(a * a * sin_phi * sin_phi + b * b * cos_phi * cos_phi));
                
    let sinZeroCross = sin(zeroCross);
    let eta = (zeta + cos(zeroCross)) / (sinZeroCross * sinZeroCross);
    var k: i32;

    var m = array<vec4f,SECTOR_COUNT>();
    var s = array<vec3f,SECTOR_COUNT>();
    for (k = 0; k < SECTOR_COUNT; k = k + 1) {
        m[k] = vec4f(0.0);
        s[k] = vec3f(0.0);
    }

    let sqrt2 = sqrt(2.0);
    var sum:f32 = 0.0;
    // float w[8];
    var w = array<f32,SECTOR_COUNT>();
    var z :f32;
    var vxx :f32;
    var vyy :f32;
                
    for (var y:i32 = -max_y; y <= max_y; y=y+1) {
        for (var x:i32 = -max_x; x <= max_x; x=x+1) {
            var v = SR * vec2f(f32(x), f32(y));
            if (dot(v, v) <= 0.25) {
                let c = textureLoad(input_texture, coords + vec2(x, y), 0).rgb;
                sum = 0.0;
                /* Calculate Polynomial Weights */
                vxx = zeta - eta * v.x * v.x;
                vyy = zeta - eta * v.y * v.y;
                z = max(0.0, v.y + vxx); 
                w[0] = z * z;
                sum += w[0];
                z = max(0.0, -v.x + vyy); 
                w[2] = z * z;
                sum += w[2];
                z = max(0.0, -v.y + vxx); 
                w[4] = z * z;
                sum += w[4];
                z = max(0.0, v.x + vyy); 
                w[6] = z * z;
                sum += w[6];
                v = sqrt2 / 2.0 * vec2f(v.x - v.y, v.x + v.y);
                vxx = zeta - eta * v.x * v.x;
                vyy = zeta - eta * v.y * v.y;
                z = max(0.0, v.y + vxx); 
                w[1] = z * z;
                sum += w[1];
                z = max(0.0, -v.x + vyy); 
                w[3] = z * z;
                sum += w[3];
                z = max(0.0, -v.y + vxx); 
                w[5] = z * z;
                sum += w[5];
                z = max(0.0, v.x + vyy); 
                w[7] = z * z;
                sum += w[7];
                
                let g = exp(-3.125 * dot(v,v)) / (sum+eps);
                for (k = 0; k < SECTOR_COUNT; k = k + 1) {
                    let wk = w[k] * g;
                    m[k] += vec4f(c * wk, wk);
                    s[k] += c * c * wk;
                }
            }
        }
    }

    
    var output = vec4f(0.0);
    for (k = 0; k < SECTOR_COUNT; k = k + 1) {
        var  mw = m[k].w;
        if (mw == 0.0) {
            mw = 1.0;
        }
        m[k].r /= mw;
        m[k].g /= mw;
        m[k].b /= mw;
        
        //s[k] = abs(s[k] / m[k].w - m[k].rgb * m[k].rgb);
        s[k] = s[k] / mw - m[k].rgb * m[k].rgb;

        //let sigma2 = s[k].r + s[k].g + s[k].b;
        //let sigma2 = length(s[k]);
        //let w = 1.0 / (1.0 + pow(hardness * 1000.0f * sigma2, 0.5 * q));
        let sdv = dot(sqrt(s[k]), vec3f(1.0));
        /* Compute the sector weight based on the weight function introduced in section "3.3.1
        * Single-scale Filtering" of the multi-scale paper. Use a threshold of 0.02 to avoid zero
        * division and avoid artifacts in homogeneous regions as demonstrated in the paper. */
        // Kyprianidis, Jan Eric. "Image and video abstraction by multi-scale anisotropic Kuwahara filtering." 2011.
        let w = 1.0 / pow(max(0.02f, sdv), q);
       
        output += vec4f(m[k].rgb * w, w);
    }
    output /= output.w;
    textureStore(output_texture, coords.xy, output);
}