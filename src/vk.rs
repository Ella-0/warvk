use vulkano::{
    buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState},
    device::{Device, DeviceExtensions, Queue, QueuesIter},
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::{ImageUsage, SwapchainImage},
    instance::{Instance, InstanceExtensions, PhysicalDevice, PhysicalDeviceType, QueueFamily},
    memory::DeviceMemory,
    pipeline::{vertex::Vertex, viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract},
    swapchain,
    swapchain::{
        display::{Display, DisplayMode, DisplayPlane},
        AcquireError, ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform,
        Swapchain, SwapchainCreationError,
    },
    sync,
    sync::{FlushError, GpuFuture},
	image::immutable::{ImmutableImage}
};

use smithay::wayland::shm;
use smithay::reexports::wayland_server::{Resource, protocol::wl_buffer};

extern crate vk_sys as vk;

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

pub fn choose_queue_family<W>(
    physical_device: PhysicalDevice,
    surface: Arc<Surface<W>>,
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

pub fn create_swapchain<W>(
    physical_device: PhysicalDevice,
    surface: Arc<Surface<W>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    dimensions: [u32; 2],
) -> (Arc<Swapchain<W>>, Vec<Arc<SwapchainImage<W>>>) {
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

pub fn create_render_pass<W>(
    device: Arc<Device>,
    swapchain: Arc<Swapchain<W>>,
) -> Arc<dyn RenderPassAbstract + Send + Sync> {
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
layout(location = 0) out vec2 tex_coords;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = position/2 + vec2(0.5);
}
		"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;
layout(set = 0, binding = 0) uniform sampler2D tex;
void main() {
    f_color = texture(tex, tex_coords).bgra;
}
		"
    }
}

/*mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = position + vec2(0.5);
}
		"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;
void main() {
    f_color = tex_coords.xyxy;
}
		"
    }
}*/



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
            .triangle_strip()
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

pub fn create_framebuffers<W>(
    images: &Vec<Arc<SwapchainImage<W>>>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>>
where
    W: Send + Sync + 'static,
{
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

pub struct VkCtx<W>
where
    W: Send + Sync + 'static,
{
    surface: Arc<Surface<W>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    dimensions: [u32; 2],
    swapchain: Arc<Swapchain<W>>,
    swapchain_images: Vec<Arc<SwapchainImage<W>>>,
    dynamic_state: DynamicState,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
	sampler: Arc<vulkano::sampler::Sampler>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

#[derive(Clone)]
enum AbstractSurface {
    Window(Arc<Surface<winit::window::Window>>),
    Screen(Arc<Surface<()>>),
}

impl VkCtx<()> {
    pub fn init() -> VkCtx<()> {
        let instance = create_instance();

        let physical = choose_physical_device(&instance);

        let display = choose_display(physical);

        let display_mode = choose_display_mode(display);

        let display_plane = choose_display_plane(physical);
        VkCtx::<()>::agnostic_init(
            instance.clone(),
            physical,
            Surface::<()>::from_display_mode(&display_mode, &display_plane)
                .expect("Failed to create surface"),
            display_mode.visible_region(),
        )
    }
}

impl VkCtx<winit::window::Window> {
    pub fn init() -> VkCtx<winit::window::Window> {
        let instance = create_instance();

        let physical = choose_physical_device(&instance);

        use vulkano_win::VkSurfaceBuild;
        let event_loop = winit::event_loop::EventLoop::new();
        let surface = winit::window::WindowBuilder::new()
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();
        VkCtx::<winit::window::Window>::agnostic_init(
            instance.clone(),
            physical,
            surface.clone(),
            [
                surface.clone().window().inner_size().width,
                surface.window().inner_size().height,
            ],
        )
    }
}

pub struct TextureMetadata {
    pub texture: Arc<CpuAccessibleBuffer<[u8]>>,
    pub fragment: usize,
    pub y_inverted: bool,
    pub dimensions: (u32, u32),
    #[cfg(feature = "egl")]
    images: Option<EGLImages>,
}

impl<W> VkCtx<W>
where
    W: Send + Sync,
{
    pub fn agnostic_init(
        instance: Arc<Instance>,
        physical: PhysicalDevice,
        surface: Arc<Surface<W>>,
        dimensions: [u32; 2],
    ) -> VkCtx<W> {
        let queue_family = choose_queue_family(physical, surface.clone());

        let (device, mut queues) = create_device(physical, queue_family);

        let queue = queues.next().unwrap();

        let (mut swapchain, images) = create_swapchain(
            physical,
            surface.clone(),
            device.clone(),
            queue.clone(),
            dimensions,
        );

        #[derive(Default, Debug, Clone)]
        struct Vertex {
            position: [f32; 2],
        }
        vulkano::impl_vertex!(Vertex, position);

        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::all(),
                false,
                [
        			Vertex {
                        position: [-1f32, -1f32],
                    },
                    Vertex {
                        position: [-1f32, 1f32],
                    },
                    Vertex {
                        position: [1f32, -1f32],
                    },
                    Vertex {
                        position: [1f32, 1f32],
                    },
                ]
                .iter()
                .cloned(),
            )
            .unwrap()
        };

        let render_pass = create_render_pass(device.clone(), swapchain.clone());

        let pipeline = create_pipeline::<Vertex>(device.clone(), render_pass.clone());

        let mut dynamic_state = DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
            compare_mask: None,
            write_mask: None,
            reference: None,
        };

        let framebuffers = create_framebuffers(&images, render_pass.clone(), &mut dynamic_state);

        let recreate_swapchain = false;

		let sampler = vulkano::sampler::Sampler::new(
            device.clone(),
            vulkano::sampler::Filter::Linear,
            vulkano::sampler::Filter::Linear,
            vulkano::sampler::MipmapMode::Nearest,
            vulkano::sampler::SamplerAddressMode::Repeat,
            vulkano::sampler::SamplerAddressMode::Repeat,
            vulkano::sampler::SamplerAddressMode::Repeat,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        .unwrap();

        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        VkCtx::<W> {
            surface,
            device,
            queue,
            dimensions,
            swapchain,
            swapchain_images: images,
            vertex_buffer,
            render_pass,
			sampler,
            pipeline,
            dynamic_state,
            framebuffers,
            recreate_swapchain,
            previous_frame_end,
        }
    }

    pub fn run(&mut self) {
        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU has
        // already processed, and frees the resources that are no longer needed.
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        // Whenever the window resizes we need to recreate everything dependent on the window size.
        // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
        if self.recreate_swapchain {
            // Get the new dimensions of the window.
            let dimensions: [u32; 2] = self.dimensions;
            let (new_swapchain, new_images) =
                match self.swapchain.recreate_with_dimensions(dimensions) {
                    Ok(r) => r,
                    // This error tends to happen when the user is manually resizing the window.
                    // Simply restarting the loop is the easiest way to fix this issue.
                    Err(SwapchainCreationError::UnsupportedDimensions) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };

            self.swapchain = new_swapchain;
            // Because framebuffers contains an Arc on the old swapchain, we need to
            // recreate framebuffers as well.
            self.framebuffers = create_framebuffers(
                &new_images,
                self.render_pass.clone(),
                &mut self.dynamic_state,
            );
            self.recreate_swapchain = false;
        }

        // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
        // no image is available (which happens if you submit draw commands too quickly), then the
        // function will block.
        // This operation returns the index of the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is an optional timeout
        // after which the function call will return an error.
        let (image_num, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        // acquire_next_image can be successful, but suboptimal. This means that the swapchain image
        // will still work, but it may not display correctly. With some drivers this can be when
        // the window resizes, but it may not cause the swapchain to become out of date.
        if suboptimal {
            self.recreate_swapchain = true;
        }

        // Specify the color to clear the framebuffer with i.e. blue
        let clear_values = vec![[0.0, 0.0, 0.0, 0.0].into()];

        // In order to draw, we have to build a *command buffer*. The command buffer object holds
        // the list of commands that are going to be executed.
        //
        // Building a command buffer is an expensive operation (usually a few hundred
        // microseconds), but it is known to be a hot path in the driver and is expected to be
        // optimized.
        //
        // Note that we have to pass a queue family when we create the command buffer. The command
        // buffer will only be executable on that given queue family.
        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap();

        builder
            // Before we can draw, we have to *enter a render pass*. There are two methods to do
            // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
            // not covered here.
            //
            // The third parameter builds the list of values to clear the attachments with. The API
            // is similar to the list of attachments when building the framebuffers, except that
            // only the attachments that use `load: Clear` appear in the list.
            .begin_render_pass(self.framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            // We are now inside the first subpass of the render pass. We add a draw command.
            //
            // The last two parameters contain the list of resources to pass to the shaders.
            // Since we used an `EmptyPipeline` object, the objects have to be `()`.
            .draw(
                self.pipeline.clone(),
                &self.dynamic_state,
                vec![self.vertex_buffer.clone()],
                (),
                (),
            )
            .unwrap()
            // We leave the render pass by calling `draw_end`. Note that if we had multiple
            // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
            // next subpass.
            .end_render_pass()
            .unwrap();

        // Finish building the command buffer by calling `build`.
        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            // The color output is now expected to contain our triangle. But in order to show it on
            // the screen, we have to *present* the image by calling `present`.
            //
            // This function does not actually present the image immediately. Instead it submits a
            // present command at the end of the queue. This means that it will only be presented once
            // the GPU has finished executing the command buffer that draws the triangle.
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }

	pub fn load_shm_buffer(&mut self, buffer: &Resource<wl_buffer::WlBuffer>) -> Arc<CpuAccessibleBuffer<[u8]>> {
		shm::with_buffer_contents(buffer, |slice, data| {
			CpuAccessibleBuffer::from_iter(
           		self.device.clone(),
                BufferUsage::all(),
                false,
                slice.iter().map(|x| *x),
            )
            .unwrap()
		}).unwrap()
	}

	pub fn load_shm_buffer_to_image(&mut self, buffer: &Resource<wl_buffer::WlBuffer>) -> Arc<ImmutableImage<vulkano::format::Format>> {
		let (img, img_future) = shm::with_buffer_contents(buffer, |slice, data| {
			ImmutableImage::from_iter(
                slice.iter().map(|x| *x),
				vulkano::image::Dimensions::Dim2d { width: data.width as u32, height: data.height as u32},
				vulkano::format::Format::R8G8B8A8Srgb,
				self.queue.clone()
            )
            .unwrap()
		}).unwrap();



		self.previous_frame_end = Some(img_future.boxed());
		img
	}

    pub fn render_shm_buffer(&mut self, buffer: Arc<ImmutableImage<vulkano::format::Format>>) {
        let (image_num, _suboptimald,  acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap();

        let clear_values = vec![[0.0, 0.0, 0.0, 0.0].into()];

		let layout = self.pipeline.descriptor_set_layout(0).unwrap();
        let set = Arc::new(
            vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(layout.clone())
                .add_sampled_image(buffer.clone(), self.sampler.clone())
                .unwrap()
                .build()
                .unwrap(),
        );        
		let _ = builder
            .begin_render_pass(self.framebuffers[image_num].clone(), false, clear_values).unwrap()
            .draw(
                self.pipeline.clone(),
                &self.dynamic_state,
                vec![self.vertex_buffer.clone()],
                set.clone(),
                (),
            )
            .expect("Failed to render shm")
            .end_render_pass().expect("Failed to end render pass");

        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }
}
