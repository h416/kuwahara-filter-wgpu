use wgpu::util::DeviceExt;

use image::Rgba;
use image::RgbaImage;

// Giuseppe Papari, Nicolai Petkov, and Patrizio Campisi, Artistic Edge and Corner Enhancing Smoothing, IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 16, NO. 10, OCTOBER 2007, pages 2449–2461
// https://blog.redwarp.app/image-filters/
// https://github.com/redwarp/blog/blob/main/code-sample/image-filters/src/main.rs
// https://github.com/redwarp/filters
// https://en.wikipedia.org/wiki/Kuwahara_filter
// https://www.youtube.com/watch?v=LDhN-JK3U9g
// https://www.umsl.edu/~kangh/Papers/kang-tpcg2010.pdf
// https://blog.maximeheckel.com/posts/on-crafting-painterly-shaders/
// https://github.com/GarrettGunnell/Post-Processing/blob/main/Assets/Kuwahara%20Filter/AnisotropicKuwahara.shader
// https://projects.blender.org/blender/blender/pulls/110786

fn load_rgba(path: &str) -> anyhow::Result<RgbaImage> {
    let img = load_image::load_path(path)?.into_imgvec();
    match img {
        load_image::export::imgref::ImgVecKind::RGB8(img) => Ok(RgbaImage::from_fn(
            img.width() as u32,
            img.height() as u32,
            |x, y| {
                let col = img.buf()[(y * img.stride() as u32 + x) as usize];
                Rgba([col.r, col.g, col.b, 255])
            },
        )),
        load_image::export::imgref::ImgVecKind::RGBA8(img) => Ok(RgbaImage::from_fn(
            img.width() as u32,
            img.height() as u32,
            |x, y| {
                let col = img.buf()[(y * img.stride() as u32 + x) as usize];
                Rgba([col.r, col.g, col.b, col.a])
            },
        )),
        load_image::export::imgref::ImgVecKind::RGB16(img) => Ok(RgbaImage::from_fn(
            img.width() as u32,
            img.height() as u32,
            |x, y| {
                let col = img.buf()[(y * img.stride() as u32 + x) as usize];
                let r = ((col.r as u32) * 255 / 65535) as u8;
                let g = ((col.g as u32) * 255 / 65535) as u8;
                let b = ((col.b as u32) * 255 / 65535) as u8;
                Rgba([r, g, b, 255])
            },
        )),
        load_image::export::imgref::ImgVecKind::RGBA16(img) => Ok(RgbaImage::from_fn(
            img.width() as u32,
            img.height() as u32,
            |x, y| {
                let col = img.buf()[(y * img.stride() as u32 + x) as usize];
                let r = ((col.r as u32) * 255 / 65535) as u8;
                let g = ((col.g as u32) * 255 / 65535) as u8;
                let b = ((col.b as u32) * 255 / 65535) as u8;
                let a = ((col.a as u32) * 255 / 65535) as u8;
                Rgba([r, g, b, a])
            },
        )),
        _ => Err(anyhow::anyhow!(
            "image type error. only color image is supported."
        )),
    }
}

// use macro to use include_str!
macro_rules! create_compute_pipeline {
    ($device:expr, $path:expr, $name:expr) => {{
        let shader_code = include_str!($path);
        let shader_module = $device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("shader_{}", $name)),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        $device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("pipeline_{}", $name)),
            layout: None,
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }};
}

fn create_bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    input_texture: &wgpu::Texture,
    output_texture: &wgpu::Texture,
    name: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("bind_group_{}", name)),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &input_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &output_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
        ],
    })
}

fn create_bind_group_uniform(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label_f32_pairs: &[(&str, f32)],
    bind_group_name: &str,
) -> wgpu::BindGroup {
    let buffers: Vec<_> = label_f32_pairs
        .iter()
        .map(|(label, value)| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[*value]),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        })
        .collect();

    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(bind_group_name),
        layout,
        entries: &entries,
    })
}

fn create_texture(
    device: &wgpu::Device,
    texture_size: wgpu::Extent3d,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
    })
}

async fn filter(
    filter_size: i32,
    sharpness: f32,
    eccentricity: f32,
    uniformity: f32,
    src_path: &str,
    dst_path: &str,
) -> anyhow::Result<()> {
    env_logger::init();

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptionsBase {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await?;
    let (device, queue) = adapter.request_device(&Default::default()).await?;

    // Load the image
    let input_image = load_rgba(src_path)?;
    let (width, height) = input_image.dimensions();

    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let input_texture = create_texture(&device, texture_size, "input_texture");

    // copy input_image to input_texture
    queue.write_texture(
        input_texture.as_image_copy(),
        bytemuck::cast_slice(input_image.as_raw()),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: None, // Doesn't need to be specified as we are writing a single image.
        },
        texture_size,
    );

    let tmp1_texture = create_texture(&device, texture_size, "tmp1");
    let tmp2_texture = create_texture(&device, texture_size, "tmp2");
    let output_texture = create_texture(&device, texture_size, "output_texture");

    let pipeline_edge = create_compute_pipeline!(&device, "shaders/edge.wgsl", "edge");
    let bind_group_edge = create_bind_group(
        &device,
        &pipeline_edge,
        &input_texture,
        &tmp1_texture,
        "edge",
    );
    let pipeline_blur = create_compute_pipeline!(&device, "shaders/blur.wgsl", "blur");
    let bind_group_blur = create_bind_group(
        &device,
        &pipeline_blur,
        &tmp1_texture,
        &tmp2_texture,
        "blur",
    );
    let bind_group_blur_params = create_bind_group_uniform(
        &device,
        &pipeline_blur.get_bind_group_layout(1),
        &[("uniformity", uniformity)],
        "bind_group_blur_params",
    );

    let pipeline_kuwa = create_compute_pipeline!(&device, "shaders/kuwahara.wgsl", "kuwa");
    let bind_group_kuwa = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group_kuwa"),
        layout: &pipeline_kuwa.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &input_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &tmp2_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &output_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
        ],
    });

    let filter_size_f32 = filter_size as f32;
    let bind_group_kuwa_params = create_bind_group_uniform(
        &device,
        &pipeline_kuwa.get_bind_group_layout(1),
        &[
            ("filter_size_buffer", filter_size_f32),
            ("sharpness_buffer", sharpness),
            ("eccentricity_buffer", eccentricity),
        ],
        "bind_group_kuwa_params",
    );

    // Dispatch

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let (dispatch_with, dispatch_height) =
            compute_work_group_count((texture_size.width, texture_size.height), (16, 16));

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline_edge);
        compute_pass.set_bind_group(0, &bind_group_edge, &[]);
        compute_pass.dispatch_workgroups(dispatch_with, dispatch_height, 1);

        compute_pass.set_pipeline(&pipeline_blur);
        compute_pass.set_bind_group(0, &bind_group_blur, &[]);
        compute_pass.set_bind_group(1, &bind_group_blur_params, &[]);
        compute_pass.dispatch_workgroups(dispatch_with, dispatch_height, 1);

        compute_pass.set_pipeline(&pipeline_kuwa);
        compute_pass.set_bind_group(0, &bind_group_kuwa, &[]);
        compute_pass.set_bind_group(1, &bind_group_kuwa_params, &[]);
        compute_pass.dispatch_workgroups(dispatch_with, dispatch_height, 1);
    }

    // Get the result.

    let padded_bytes_per_row = padded_bytes_per_row(width);
    let unpadded_bytes_per_row = width as usize * 4;

    let output_buffer_size =
        padded_bytes_per_row as u64 * height as u64 * std::mem::size_of::<u8>() as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            aspect: wgpu::TextureAspect::All,
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row as u32),
                rows_per_image: Some(height),
            },
        },
        texture_size,
    );
    queue.submit(Some(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

    device.poll(wgpu::PollType::Wait{submission_index:None, timeout:Some(std::time::Duration::from_secs(60))})?;

    let padded_data = buffer_slice.get_mapped_range();

    let mut pixels: Vec<u8> = vec![0; unpadded_bytes_per_row * height as usize];
    for (padded, pixels) in padded_data
        .chunks_exact(padded_bytes_per_row)
        .zip(pixels.chunks_exact_mut(unpadded_bytes_per_row))
    {
        pixels.copy_from_slice(&padded[..unpadded_bytes_per_row]);
    }

    if let Some(output_image) =
        image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, &pixels[..])
    {
        output_image.save(dst_path)?;
    }

    Ok(())
}

/// Compute the amount of work groups to be dispatched for an image, based on the work group size.
/// Chances are, the group will not match perfectly, like an image of width 100, for a workgroup size of 32.
/// To make sure the that the whole 100 pixels are visited, then we would need a count of 4, as 4 * 32 = 128,
/// which is bigger than 100. A count of 3 would be too little, as it means 96, so four columns (or, 100 - 96) would be ignored.
///
/// # Arguments
///
/// * `(width, height)` - The dimension of the image we are working on.
/// * `(workgroup_width, workgroup_height)` - The width and height dimensions of the compute workgroup.
fn compute_work_group_count(
    (width, height): (u32, u32),
    (workgroup_width, workgroup_height): (u32, u32),
) -> (u32, u32) {
    let x = (width + workgroup_width - 1) / workgroup_width;
    let y = (height + workgroup_height - 1) / workgroup_height;

    (x, y)
}

/// Compute the next multiple of 256 for texture retrieval padding.
fn padded_bytes_per_row(width: u32) -> usize {
    let bytes_per_row = width as usize * 4;
    let padding = (256 - bytes_per_row % 256) % 256;
    bytes_per_row + padding
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // dbg!(&args);
    if args.len() != 7 {
        let command = &args[0];
        println!(
            "usage {} filter_size sharpness eccentricity uniformity src dst",
            &command
        );
        println!("example {} 7 10.0 4.0 2.0 src.jpg dst.png", &command);
        return Err(anyhow::anyhow!("usage error"));
    }

    let filter_size: i32 = args[1].parse().unwrap();
    if filter_size < 1 {
        println!("filter_size must be greater than or equal to 1");
        return Err(anyhow::anyhow!("argument error"));
    }
    let sharpness: f32 = args[2].parse().unwrap();
    if sharpness <= 0.0 {
        println!("sharpness must be greater than 0");
        return Err(anyhow::anyhow!("argument error"));
    }
    let eccentricity: f32 = args[3].parse().unwrap();
    if eccentricity <= 0.0 {
        println!("eccentricity must be greater than 0");
        return Err(anyhow::anyhow!("argument error"));
    }
    let uniformity: f32 = args[4].parse().unwrap();
    if uniformity <= 0.0 {
        println!("uniformity must be greater than 0");
        return Err(anyhow::anyhow!("argument error"));
    }

    let src = &args[5];
    let dst = &args[6];

    filter(filter_size, sharpness, eccentricity, uniformity, src, dst).await?;
    Ok(())
}
