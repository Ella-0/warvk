use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState},
    device::{Device, DeviceExtensions, Queue, QueuesIter},
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::{ImageUsage, SwapchainImage},
    instance::{Instance, InstanceExtensions, PhysicalDevice, PhysicalDeviceType, QueueFamily},
    pipeline::{vertex::Vertex, viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract},
    swapchain,
    swapchain::{
        display::{Display, DisplayMode, DisplayPlane},
        AcquireError, ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform,
        Swapchain, SwapchainCreationError,
    },
    sync,
    sync::{FlushError, GpuFuture},
};

use std::sync::Arc;

fn required_extensions() -> InstanceExtensions {
    let ideal = InstanceExtensions {
        khr_surface: true,
        khr_display: true,
        khr_xlib_surface: true,
        khr_xcb_surface: true,
        khr_wayland_surface: true,
        khr_android_surface: true,
        khr_win32_surface: true,
        mvk_ios_surface: true,
        mvk_macos_surface: true,
        khr_get_physical_device_properties2: true,
        khr_get_surface_capabilities2: true,
        ..InstanceExtensions::none()
    };

    match InstanceExtensions::supported_by_core() {
        Ok(supported) => supported.intersection(&ideal),
        Err(_) => InstanceExtensions::none(),
    }
}

pub fn create_instance() -> Arc<Instance> {
    let supported_extensions =
        InstanceExtensions::supported_by_core().expect("failed to retrieve supported extensions");
    println!("╠══ Supported extensions: {:?}", supported_extensions);

    let required_extensions = required_extensions();

    Instance::new(None, &required_extensions, None).expect("Failed to create Instance")
}

pub fn choose_physical_device(instance: &Arc<Instance>) -> PhysicalDevice {
    println!("╠══ Looking for physical devices");

    let physical_devices = PhysicalDevice::enumerate(instance);
    let mut physical_device_ret = None;

    for physical_device in physical_devices {
        println!(
            "║\tFound physical device: {} (type: {:?})",
            physical_device.name(),
            physical_device.ty()
        );

        if physical_device_ret.is_none() || physical_device.ty() == PhysicalDeviceType::DiscreteGpu
        {
            physical_device_ret = Some(physical_device);
        }
    }
    let physical_device = physical_device_ret.expect("Failed to find device");

    println!(
        "╠══ Using physical device: {} (type: {:?})",
        physical_device.name(),
        physical_device.ty()
    );

    physical_device
}

pub fn choose_display(physical_device: PhysicalDevice) -> Display {
    let display = Display::enumerate(physical_device)
        .next()
        .expect("No Displays Found");
    println!("╠══ Found display: {}", display.name());
    display
}

pub fn choose_display_mode(display: Display) -> DisplayMode {
    let display_modes = display.display_modes();
    let mut display_mode_ret = None;
    for display_mode in display_modes {
        println!(
            "║ \thas mode: {}, {:?}",
            display_mode.refresh_rate(),
            display_mode.visible_region()
        );
        if display_mode_ret.is_none() {
            display_mode_ret = Some(display_mode);
        }
    }
    let display_mode = display_mode_ret.expect("Display has no modes");

    println!(
        "╠══ Using mode: {}, {:?}",
        display_mode.refresh_rate(),
        display_mode.visible_region()
    );
    display_mode
}

pub fn choose_display_plane(physical_device: PhysicalDevice) -> DisplayPlane {
    let mut display_planes = DisplayPlane::enumerate(physical_device);
    println!("║ Found {} planes", display_planes.len());

    display_planes.next().expect("No planes")
}

pub fn choose_queue_family(
    physical_device: PhysicalDevice,
    surface: Arc<Surface<()>>,
) -> QueueFamily {
    physical_device
        .queue_families()
        .find(|&q| {
            // We take the first queue that supports drawing to our window.
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        })
        .unwrap()
}

pub fn create_device(
    physical_device: PhysicalDevice,
    queue_family: QueueFamily,
) -> (Arc<Device>, QueuesIter) {
    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    Device::new(
        physical_device,
        physical_device.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap()
}

pub fn create_swapchain(
    physical_device: PhysicalDevice,
    surface: Arc<Surface<()>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    dimensions: [u32; 2],
) -> (Arc<Swapchain<()>>, Vec<Arc<SwapchainImage<()>>>) {
    // Querying the capabilities of the surface. When we create the swapchain we can only
    // pass values that are allowed by the capabilities.
    let caps = surface.capabilities(physical_device).unwrap();

    // The alpha mode indicates how the alpha value of the final image will behave. For example
    // you can choose whether the window will be opaque or transparent.
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();

    // Choosing the internal format that the images will have.
    let format = caps.supported_formats[0].0;

    // The dimensions of the window, only used to initially setup the swapchain.
    // NOTE:
    // On some drivers the swapchain dimensions are specified by `caps.current_extent` and the
    // swapchain size must use these dimensions.
    // These dimensions are always the same as the window dimensions
    //
    // However other drivers dont specify a value i.e. `caps.current_extent` is `None`
    // These drivers will allow anything but the only sensible value is the window dimensions.
    //
    // Because for both of these cases, the swapchain needs to be the window dimensions, we just use that.

    // Please take a look at the docs for the meaning of the parameters we didn't mention.
    Swapchain::new(
        device.clone(),
        surface.clone(),
        caps.min_image_count,
        format,
        dimensions,
        1,
        ImageUsage::color_attachment(),
        &queue,
        SurfaceTransform::Identity,
        alpha,
        PresentMode::Fifo,
        FullscreenExclusive::Default,
        true,
        ColorSpace::SrgbNonLinear,
    )
    .unwrap()
}

pub fn create_render_pass(
    device: Arc<Device>,
    swapchain: Arc<Swapchain<()>>,
) -> Arc<RenderPassAbstract + Send + Sync> {
    Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                // `color` is a custom name we give to the first and only attachment.
                color: {
                    // `load: Clear` means that we ask the GPU to clear the content of this
                    // attachment at the start of the drawing.
                    load: Clear,
                    // `store: Store` means that we ask the GPU to store the output of the draw
                    // in the actual image. We could also ask it to discard the result.
                    store: Store,
                    // `format: <ty>` indicates the type of the format of the image. This has to
                    // be one of the types of the `vulkano::format` module (or alternatively one
                    // of your structs that implements the `FormatDesc` trait). Here we use the
                    // same format as the swapchain.
                    format: swapchain.format(),
                    // TODO:
                    samples: 1,
                }
            },
            pass: {
                // We use the attachment named `color` as the one and only color attachment.
                color: [color],
                // No depth-stencil attachment is indicated with empty brackets.
                depth_stencil: {}
            }
        )
        .unwrap(),
    )
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
			#version 450
			layout(location = 0) in vec2 position;
			void main() {
				gl_Position = vec4(position, 0.0, 1.0);
			}
		"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
			#version 450
			layout(location = 0) out vec4 f_color;
			void main() {
				f_color = vec4(1.0, 0.0, 0.0, 1.0);
			}
		"
    }
}

pub fn create_pipeline<V: Vertex>(
    device: Arc<Device>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    Arc::new(
        GraphicsPipeline::start()
            // We need to indicate the layout of the vertices.
            // The type `SingleBufferDefinition` actually contains a template parameter corresponding
            // to the type of each vertex. But in this code it is automatically inferred.
            .vertex_input_single_buffer::<V>()
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one. The `main` word of `main_entry_point` actually corresponds to the name of
            // the entry point.
            .vertex_shader(vs.main_entry_point(), ())
            // The content of the vertex buffer describes a list of triangles.
            .triangle_list()
            // Use a resizable viewport set to draw over the entire window
            .viewports_dynamic_scissors_irrelevant(1)
            // See `vertex_shader`.
            .fragment_shader(fs.main_entry_point(), ())
            // We have to indicate which subpass of which render pass this pipeline is going to be used
            // in. The pipeline will only be usable from this particular subpass.
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            // Now that our builder is filled, we call `build()` to obtain an actual pipeline.
            .build(device.clone())
            .unwrap(),
    )
}

pub fn create_framebuffers(
    images: &[Arc<SwapchainImage<()>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}
