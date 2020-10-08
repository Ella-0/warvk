use vulkano::{
    buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState},
    device::{Device, DeviceExtensions, Queue, QueuesIter},
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::immutable::ImmutableImage,
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
};

use smithay::reexports::wayland_server::{protocol::wl_buffer, Resource};
use smithay::wayland::shm;

extern crate vk_sys as vk;

use std::sync::Arc;

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

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
        ext_debug_utils: true,
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

fn create_debug_callback(
    instance: &Arc<Instance>,
) -> Option<vulkano::instance::debug::DebugCallback> {
    let severity = vulkano::instance::debug::MessageSeverity {
        error: true,
        warning: true,
        information: true,
        verbose: true,
    };
    let msg_types = vulkano::instance::debug::MessageType {
        general: true,
        validation: true,
        performance: true,
    };
    vulkano::instance::debug::DebugCallback::new(&instance, severity, msg_types, |msg| {
        println!("╠══ validation layer: {:?}", msg.description);
    })
    .ok()
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

        if physical_device_ret.is_none()
        //|| physical_device.ty() == PhysicalDeviceType::DiscreteGpu
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
layout(set = 0, binding = 0) uniform Data {
	mat4 matrix;
};
void main() {
    gl_Position = vec4(position, 0.0, 1.0) * matrix;
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
layout(set = 0, binding = 1) uniform sampler2D tex;

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
    debug_callback: Option<vulkano::instance::debug::DebugCallback>,
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
            .with_inner_size(winit::dpi::LogicalSize::new(1270.0f32, 720.0f32))
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
        //let debug_callback = create_debug_callback(&instance);
        let debug_callback = None;
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
            vulkano::sampler::Filter::Nearest,
            vulkano::sampler::Filter::Nearest,
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
            debug_callback,
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

    //.
    pub fn load_shm_buffer(
        &mut self,
        buffer: &Resource<wl_buffer::WlBuffer>,
        image: Option<Arc<vulkano::image::StorageImage<vulkano::format::Format>>>,
    ) -> (
        Arc<vulkano::image::StorageImage<vulkano::format::Format>>,
        Box<dyn GpuFuture>,
    ) {
        shm::with_buffer_contents(buffer, |pool, data| {
            use std::borrow::Cow;
            let pixelsize = 4;

            let offset = data.offset as usize;
            let width = data.width as usize;
            let height = data.height as usize;
            let stride = data.stride as usize;

            let slice: Cow<'_, [u8]> = if stride == width * pixelsize {
                // the buffer is cleanly continuous, use as-is
                Cow::Borrowed(&pool[offset..(offset + height * width * pixelsize)])
            } else {
                // the buffer is discontinuous or lines overlap
                // we need to make a copy as unfortunately Glium does not
                // expose the OpenGL APIs we would need to load this buffer :/
                let mut data = Vec::with_capacity(height * width * pixelsize);
                for i in 0..height {
                    data.extend(
                        &pool[(offset + i * stride)..(offset + i * stride + width * pixelsize)],
                    );
                }
                Cow::Owned(data)
            };
            //let slice = pool;

            let dim = vulkano::image::Dimensions::Dim2d {
                width: data.width as u32,
                height: data.height as u32,
            };
            let image = if image.is_none() || image.as_ref().unwrap().dimensions() == dim {
                let usage = ImageUsage {
                    transfer_source: true, // for blits
                    transfer_destination: true,
                    sampled: true,
                    ..ImageUsage::none()
                };

                vulkano::image::StorageImage::with_usage(
                    self.device.clone(),
                    dim,
                    vulkano::format::Format::R8G8B8A8Srgb,
                    usage,
                    Some(self.queue.family()),
                )
                .unwrap()
            } else {
                image.unwrap()
            };

            let buffer = unsafe {
                let uninitialized = CpuAccessibleBuffer::<[u8]>::uninitialized_array(
                    self.device.clone(),
                    slice.len(),
                    BufferUsage::all(),
                    false,
                )
                .unwrap();

                // Note that we are in panic-unsafety land here. However a panic should never ever
                // happen here, so in theory we are safe.
                // TODO: check whether that's true ^

                {
                    let mut mapping = uninitialized.write().unwrap();
                    mapping.copy_from_slice(&slice);
                }

                uninitialized
            };

            use vulkano::command_buffer::CommandBuffer;

            let mut builder =
                AutoCommandBufferBuilder::new(self.device.clone(), self.queue.family()).unwrap();

            builder
                .copy_buffer_to_image(buffer.clone(), image.clone())
                .unwrap();
            let command_buffer = builder.build().unwrap();
            let finished = command_buffer.execute(self.queue.clone()).unwrap();

            (image, finished.boxed())
        })
        .expect("Failed")
    }

    pub fn load_shm_buffer_to_image(
        &mut self,
        buffer: &Resource<wl_buffer::WlBuffer>,
    ) -> (
        Arc<ImmutableImage<vulkano::format::Format>>,
        Box<dyn GpuFuture>,
    ) {
        let (img, img_future) = shm::with_buffer_contents(buffer, |slice, data| {
            ImmutableImage::from_iter(
                slice.iter().map(|x| *x),
                vulkano::image::Dimensions::Dim2d {
                    width: data.width as u32,
                    height: data.height as u32,
                },
                vulkano::format::Format::R8G8B8A8Unorm,
                self.queue.clone(),
            )
            .unwrap()
        })
        .unwrap();

        (img, (img_future).boxed())
    }

    pub fn render_windows(
        &mut self,
        compositor_token: smithay::wayland::compositor::CompositorToken<
            crate::shell::SurfaceData,
            crate::shell::Roles,
        >,
        window_map: std::rc::Rc<
            std::cell::RefCell<
                crate::window_map::WindowMap<
                    crate::shell::SurfaceData,
                    crate::shell::Roles,
                    (),
                    (),
                    for<'r> fn(
                        &'r smithay::wayland::compositor::SurfaceAttributes<
                            crate::shell::SurfaceData,
                        >,
                    ) -> Option<(i32, i32)>,
                >,
            >,
        >,
    ) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

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

        let (image_num, suboptimal, acquire_future) =
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

        if suboptimal {
            self.recreate_swapchain = true;
        }

        let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];

        let pipeline2 = self.pipeline.clone();
        let layout = pipeline2.descriptor_set_layout(0).unwrap();

        let mut future = self.previous_frame_end.take().unwrap().join(acquire_future);

        let mut future: Box<dyn GpuFuture> = Box::new(future);

        let mut futures: Vec<Box<dyn GpuFuture>> = Vec::new();

        let _ = builder
            .begin_render_pass(self.framebuffers[image_num].clone(), false, clear_values)
            .unwrap();

        window_map.borrow().with_windows_from_bottom_to_top(
            |toplevel_surface, initial_place| {
                if let Some(wl_surface) = toplevel_surface.get_surface() {
                    let _ = compositor_token.with_surface_tree_upward(
                        wl_surface,
                        initial_place,
                        |_surface, attributes, role, &(mut x, mut y)| {
                            // there is actually something to draw !
                            if !attributes.user_data.texture {
                                if let Some(buffer) = attributes.user_data.buffer.take() {
                                	let (texture, future) = self.load_shm_buffer(&buffer.clone(), attributes.user_data.image.clone());
									//self.load_shm_buffer_to_image(&buffer.clone());
									attributes.user_data.texture = true;
									attributes.user_data.image = Some(texture);

                        			future.then_signal_fence_and_flush().unwrap()
                            			.wait(None).unwrap();


									//futures.push(future);
                                    // notify the client that we have finished reading the
                                    // buffer
                                    buffer.send(wl_buffer::Event::Release);
                                }
                            }
                            if let Some(ref metadata) = attributes.user_data.image {
                                if let Ok(subdata) = smithay::wayland::compositor::roles::Role::<smithay::wayland::compositor::SubsurfaceRole>::data(role) {
                                    x += subdata.location.0;
                                    y += subdata.location.1;
                                }
								let uniform_buffer = vulkano::buffer::CpuBufferPool::<vs::ty::Data>::new(self.device.clone(), BufferUsage::all());

								let xscale = (metadata.dimensions().width() as f32) / (self.dimensions[0] as f32);
                                let mut yscale = (metadata.dimensions().height() as f32) / (self.dimensions[1] as f32);

                                let xpos = 2.0 * (self.dimensions[0] as f32 / 2.0) / (self.dimensions[0] as f32) - 1.0;
                                let mut ypos = 1.0 - 2.0 * (self.dimensions[1] as f32 / 2.0) / (self.dimensions[1] as f32);
                                

								let uniform_data = vs::ty::Data {
									matrix: [
                                        [xscale,   0.0  , 0.0, 0.0],
                                        [  0.0 , yscale , 0.0, 0.0],
                                        [  0.0 ,   0.0  , 1.0, 0.0],
                                        [xpos  , ypos   , 0.0, 1.0]
									]
								};
								let sub_buffer = uniform_buffer.next(uniform_data).unwrap();

                                let set = Arc::new(
                                    vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(layout.clone())
                                        .add_buffer(sub_buffer).unwrap()
										.add_sampled_image(metadata.clone(), self.sampler.clone())
                                        .unwrap()
                                        .build()
                                        .unwrap(),
                                );

                                let _ = builder.draw(
                                    self.pipeline.clone(),
                                    &self.dynamic_state,
                                    vec![self.vertex_buffer.clone()],
                                    set.clone(),
                                    (),
                                ).expect("Failed to render shm");
										
                                //vk_ctx.run();
                                smithay::wayland::compositor::TraversalAction::DoChildren((x, y))
                            } else {
                                // we are not display, so our children are neither
                                smithay::wayland::compositor::TraversalAction::SkipChildren
                            }
                        },
                    );
                }
            },
        );

        let future_iter = futures.into_iter();

        let future = future_iter.fold(future, |acc, x| acc.join(x).boxed());

        builder
            .end_render_pass()
            .expect("Failed to end render pass");

        let command_buffer = builder.build().unwrap();

        let future = future
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

    pub fn render_shm_buffer(&mut self, buffer: Arc<ImmutableImage<vulkano::format::Format>>) {
        let (image_num, _suboptimald, acquire_future) =
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
            .begin_render_pass(self.framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            .draw(
                self.pipeline.clone(),
                &self.dynamic_state,
                vec![self.vertex_buffer.clone()],
                set.clone(),
                (),
            )
            .expect("Failed to render shm")
            .end_render_pass()
            .expect("Failed to end render pass");

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
            Ok(mut future) => {
                future.cleanup_finished();
                future.wait(None).unwrap();
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
