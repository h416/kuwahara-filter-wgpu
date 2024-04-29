use wgpu::util::DeviceExt;

use pollster::FutureExt;

// https://blog.redwarp.app/image-filters/

// https://github.com/redwarp/blog/blob/main/code-sample/image-filters/src/main.rs
// https://github.com/redwarp/filters

// https://en.wikipedia.org/wiki/Kuwahara_filter
// https://www.youtube.com/watch?v=LDhN-JK3U9g
// Giuseppe Papari, Nicolai Petkov, and Patrizio Campisi, Artistic Edge and Corner Enhancing Smoothing, IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 16, NO. 10, OCTOBER 2007, pages 2449–2461

const SECTOR_COUNT: usize = 8;

fn filter(
    filter_size: i32,
    q: f32,
    sigma: f32,
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
        .block_on()
        .ok_or(anyhow::anyhow!("Couldn't create the adapter"))?;
    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .block_on()?;

    // Load the image

    let input_image = image::open(src_path)?.to_rgba8();

    let (width, height) = input_image.dimensions();

    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let input_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("input_texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    });

    queue.write_texture(
        input_texture.as_image_copy(),
        bytemuck::cast_slice(input_image.as_raw()),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: None, // Doesn't need to be specified as we are writing a single image.
        },
        texture_size,
    );

    // Create an output texture
    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output_texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
    });

    // Create the compute pipeline and bindings

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/kuwahara.wgsl").into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline"),
        layout: None,
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
    });

    let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("texture_bind_group"),
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
    });

    let n = filter_size;
    let filter_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("filter_size_buffer"),
        contents: bytemuck::cast_slice(&[filter_size]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let q_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("q_buffer"),
        contents: bytemuck::cast_slice(&[q]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let pi = std::f32::consts::PI;
    let pi2 = pi * 2.0_f32;
    let sigma2 = sigma * sigma;
    let zero = 0.0_f32;

    // let sector_map_size = 2 * n - 1;
    // sector_map: (2 * n - 1, 2 * n - 1)
    let sector_map = get_sector_map(filter_size);
    let sector_map_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sector_map"),
        contents: bytemuck::cast_slice(&sector_map[..]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let nn = (n * n) as usize;

    let kernel_size = nn + 1;
    let mut kernel_values = vec![zero; kernel_size];
    for i in 0..=nn {
        let weight = (-(i as f32) / (2.0_f32 * sigma2)).exp() / (pi2 * sigma2);
        kernel_values[i] = weight;
    }

    let kernel = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("kernel"),
        contents: bytemuck::cast_slice(&kernel_values[..]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let compute_constants = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_constants"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: filter_size_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: q_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: kernel.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sector_map_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let (dispatch_with, dispatch_height) =
            compute_work_group_count((texture_size.width, texture_size.height), (16, 16));

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Grayscale pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &texture_bind_group, &[]);
        compute_pass.set_bind_group(1, &compute_constants, &[]);
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
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: wgpu::ImageDataLayout {
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

    device.poll(wgpu::Maintain::Wait);

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

fn get_sector_map(filter_size: i32) -> Vec<u32> {
    let n = filter_size;
    let sector_map_size = 2 * n - 1;
    let mut sector_map = vec![0; (sector_map_size * sector_map_size) as usize];

    let pi = std::f32::consts::PI;
    let pi2 = pi * 2.0_f32;

    let debug_sector_map = false;
    if debug_sector_map {
        for d in 0..SECTOR_COUNT {
            let df = (d + 1) as f32;
            let min_val = df - 0.5_f32;
            let max_val = df + 0.5_f32;
            let min_val2 = (min_val) * pi2 / (SECTOR_COUNT as f32);
            let max_val2 = (max_val) * pi2 / (SECTOR_COUNT as f32);
            println!("{} {}-{} {}-{} ", d, min_val, max_val, min_val2, max_val2);
        }
    }

    for y2 in 0..sector_map_size {
        for x2 in 0..sector_map_size {
            let x = x2 - (n - 1);
            let y = y2 - (n - 1);
            let dist2 = x * x + y * y;
            let index = x2 + sector_map_size * y2;

            if dist2 == 0 {
                let mut map_value = 0;
                for d in 0..SECTOR_COUNT {
                    map_value += 1 << d;
                }
                // println!("{},{} {}", x2, y2, map_value);
                sector_map[index as usize] = map_value;
            } else {
                let mut theta = (y as f32).atan2(x as f32) + pi;
                if theta < pi / (SECTOR_COUNT as f32) {
                    theta += pi2;
                }
                let val = (SECTOR_COUNT as f32) * theta / pi2;
                if debug_sector_map {
                    println!("{},{}  {} {}", x, y, theta, val);
                }
                let mut map_value = 0;
                for d in 0..SECTOR_COUNT {
                    let df = (d + 1) as f32;
                    let min_val = df - 0.5_f32;
                    let max_val = df + 0.5_f32;
                    let in_sector = min_val <= val && val < max_val;
                    if in_sector {
                        map_value += 1 << d;
                    }
                }
                // println!("{},{} {}", x2, y2, map_value);
                sector_map[index as usize] = map_value;
            }
        }
    }
    if debug_sector_map {
        for y2 in 0..sector_map_size {
            for x2 in 0..sector_map_size {
                let index = x2 + sector_map_size * y2;
                print!("{:3} ", &sector_map[index as usize]);
            }
            println!();
        }

        for d in 0..SECTOR_COUNT {
            println!("{}", d);
            for y2 in 0..sector_map_size {
                for x2 in 0..sector_map_size {
                    let index = x2 + sector_map_size * y2;
                    let map_value = sector_map[index as usize];
                    let in_sector = map_value & (1 << d);

                    let val = if in_sector > 0 { 1 } else { 0 };

                    print!("{} ", val);
                }
                println!();
            }
        }
    }
    sector_map
}
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // dbg!(&args);
    if args.len() != 6 {
        let command = &args[0];
        println!("usage {} filter_size q sigma src dst", &command);
        println!("example {} 7 10.0 4.0 src.jpg dst.png", &command);
        return Err(anyhow::anyhow!("usage error"));
    }

    let filter_size: i32 = args[1].parse().unwrap();
    if filter_size < 1 {
        println!("filter_size must be greater than or equal to 1");
        return Err(anyhow::anyhow!("argument error"));
    }
    let q: f32 = args[2].parse().unwrap();
    if q < 0.0 {
        println!("q must be greater than or equal to 0");
        return Err(anyhow::anyhow!("argument error"));
    }
    let sigma: f32 = args[3].parse().unwrap();
    if sigma <= 0.0 {
        println!("sigma must be greater than 0");
        return Err(anyhow::anyhow!("argument error"));
    }

    let src = &args[4];
    let dst = &args[5];

    filter(filter_size, q, sigma, src, dst)?;
    Ok(())
}
