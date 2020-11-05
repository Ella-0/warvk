use std::{
    borrow::Cow,
    cell::RefCell,
    ffi::{CStr, CString},
    rc::Rc,
};

use ash::{
    extensions::{ext, khr},
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk, Entry, Instance,
};

use smithay::wayland::compositor::{CompositorToken, SurfaceAttributes};

use crate::{ctx::RenderCtx, shell::Roles, window_map::WindowMap};

pub struct AshCtx<W>
where
    W: 'static,
{
    instance: Instance,
    debug_utils_loader: ext::DebugUtils,
    debug_call_back: vk::DebugUtilsMessengerEXT,
    device: ash::Device,
    swapchain_loader: khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    present_queue: vk::Queue,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight_fence: Vec<vk::Fence>,
    image_in_flight: Vec<vk::Fence>,
    current_frame: usize,
    data: W,
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

#[derive(Clone, Debug, Copy)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = std::mem::zeroed();
            (&b.$field as *const _ as isize) - (&b as *const _ as isize)
        }
    }};
}

fn get_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for i in 0..mem_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0 {
            return i;
        }
    }
    panic!("Failed to get suitible memory type");
}

fn create_vertex_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
) -> (vk::Buffer, vk::DeviceMemory) {
    let create_info = vk::BufferCreateInfo::builder()
        .size(std::mem::size_of::<Vertex>() as u64)
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .build();

    let buffer = unsafe { device.create_buffer(&create_info, None) }.unwrap();

    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(get_memory_type(
            instance,
            physical_device,
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ));

    let mem = unsafe { device.allocate_memory(&alloc_info, None).unwrap() };

    unsafe { device.bind_buffer_memory(buffer, mem, 0) }.unwrap();

    let vertices = [
        Vertex {
            pos: [0.0, -0.5, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
        },
        Vertex {
            pos: [0.5, 0.5, 0.0, 0.0],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            pos: [-0.5, 0.5, 0.0, 0.0],
            color: [0.0, 0.0, 1.0, 1.0],
        },
    ];

    unsafe {
        let data = device
            .map_memory(mem, 0, create_info.size, vk::MemoryMapFlags::empty())
            .unwrap();
        {
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr(),
                data as *mut Vertex,
                vertices.len() * std::mem::size_of::<Vertex>(),
            );
        }
        device.unmap_memory(mem);
    }

    (buffer, mem)
}

fn create_render_pass(device: &ash::Device, swapchain_image_format: vk::Format) -> vk::RenderPass {
    let colour_attachment = [vk::AttachmentDescription::builder()
        .format(swapchain_image_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build()];

    let colour_attachment_ref = [vk::AttachmentReference::builder()
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .attachment(0)
        .build()];

    let sub_pass = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colour_attachment_ref)
        .build()];

    let dependency = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .build()];

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&colour_attachment)
        .subpasses(&sub_pass)
        .dependencies(&dependency)
        .build();

    unsafe { device.create_render_pass(&render_pass_info, None) }.unwrap()
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

impl AshCtx<winit::window::Window> {
    pub fn init() -> Self {
        let events_loop = winit::event_loop::EventLoop::new();
        let window = winit::window::WindowBuilder::new()
            .with_title("Ash - Example")
            .with_inner_size(winit::dpi::LogicalSize::new(f64::from(640), f64::from(480)))
            .build(&events_loop)
            .unwrap();

        let surface_exts = ash_window::enumerate_required_extensions(&window).unwrap();

        Self::agnostic_init(
            window,
            surface_exts,
            |data, entry, instance, physical_device| {
                (
                    unsafe { ash_window::create_surface(entry, instance, data, None).unwrap() },
                    vk::Extent2D::builder().width(640).height(480).build(),
                )
            },
        )
    }
}

impl AshCtx<()> {
    pub fn init() -> Self {
        let surface_exts = vec![khr::Surface::name(), khr::Display::name()];

        Self::agnostic_init((), surface_exts, |_, entry, instance, physical_device| {
            let display_loader = khr::Display::new(entry, instance);

            let displays = unsafe {
                display_loader
                    .get_physical_device_display_properties(physical_device)
                    .expect("Failed to enumerate displays")
            };

            for display in &displays {
                println!("{:#?}", unsafe { CStr::from_ptr(display.display_name) });
            }

            let display = displays.iter().next().expect("No displays found");

            let modes = unsafe {
                display_loader
                    .get_display_mode_properties(physical_device, display.display)
                    .expect("Failed to get display modes")
            };

            for mode in &modes {
                println!("{}", mode.parameters.refresh_rate);
            }

            let mode = modes.iter().next().expect("No mode found");

            let display_planes = unsafe {
                display_loader
                    .get_physical_device_display_plane_properties(physical_device)
                    .expect("Failed to get display planes")
            };

            let display_plane = display_planes.iter().next().expect("No plane found");

            let surface = {
                let create_info = vk::DisplaySurfaceCreateInfoKHR::builder()
                    .display_mode(mode.display_mode)
                    .plane_index(display_plane.current_stack_index)
                    .transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                    .alpha_mode(vk::DisplayPlaneAlphaFlagsKHR::GLOBAL)
                    .build();

                unsafe { display_loader.create_display_plane_surface(&create_info, None) }
                    .expect("Failed to create surface")
            };
            (surface, mode.parameters.visible_region)
        })
    }
}

impl<W> AshCtx<W> {
    fn agnostic_init<F>(data: W, mut surface_exts: Vec<&std::ffi::CStr>, init_surface: F) -> Self
    where
        F: FnOnce(&W, &Entry, &ash::Instance, vk::PhysicalDevice) -> (vk::SurfaceKHR, vk::Extent2D),
    {
        let entry = Entry::new().unwrap();
        let app_name = CString::new("WaRVk").unwrap();
        let engine_name = CString::new("Smithay").unwrap();

        let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
        let layer_names_raw = layer_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        surface_exts.push(ext::DebugUtils::name());

        let surface_exts_raw = surface_exts
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        let instance = {
            let app_info = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .engine_version(0)
                .engine_name(&engine_name)
                .api_version(vk::make_version(1, 1, 0));

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_layer_names(&layer_names_raw)
                .enabled_extension_names(&surface_exts_raw);

            unsafe {
                entry
                    .create_instance(&create_info, None)
                    .expect("Instance creation error!")
            }
        };

        let debug_utils_loader = ext::DebugUtils::new(&entry, &instance);

        let debug_call_back = {
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(vulkan_debug_callback));

            unsafe {
                debug_utils_loader
                    .create_debug_utils_messenger(&debug_info, None)
                    .unwrap()
            }
        };

        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Error enumerating physical devices")
        };
        //let display_loader = khr::Display::new(&entry, &instance);
        let surface_loader = khr::Surface::new(&entry, &instance);

        let (physical_device, queue_family_index) = unsafe {
            physical_devices
                .iter()
                .map(|physical_device| {
                    instance
                        .get_physical_device_queue_family_properties(*physical_device)
                        .iter()
                        .enumerate()
                        .filter_map(|(index, ref info)| {
                            if info.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                                Some((*physical_device, index))
                            } else {
                                None
                            }
                        })
                        .next()
                })
                .filter_map(|v| v)
                .next()
                .expect("Could not find suitable device.")
        };

        /*let displays = unsafe {
            display_loader
                .get_physical_device_display_properties(physical_device)
                .expect("Failed to enumerate displays")
        };

        for display in &displays {
            println!("{:#?}", unsafe { CStr::from_ptr(display.display_name) });
        }

        let display = displays.iter().next().expect("No displays found");

        let modes = unsafe {
            display_loader
                .get_display_mode_properties(physical_device, display.display)
                .expect("Failed to get display modes")
        };

        for mode in &modes {
            println!("{}", mode.parameters.refresh_rate);
        }

        let mode = modes.iter().next().expect("No mode found");

        let display_planes = unsafe {
            display_loader
                .get_physical_device_display_plane_properties(physical_device)
                .expect("Failed to get display planes")
        };

        let display_plane = display_planes.iter().next().expect("No plane found");

        let surface = {
            let create_info = vk::DisplaySurfaceCreateInfoKHR::builder()
                .display_mode(mode.display_mode)
                .plane_index(display_plane.current_stack_index)
                .transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .alpha_mode(vk::DisplayPlaneAlphaFlagsKHR::GLOBAL)
                .build();

            unsafe { display_loader.create_display_plane_surface(&create_info, None) }
                .expect("Failed to create surface")
        };*/

        let (surface, surface_extent) = init_surface(&data, &entry, &instance, physical_device);

        assert!(unsafe {
            surface_loader.get_physical_device_surface_support(
                physical_device,
                queue_family_index as u32,
                surface,
            )
        }
        .unwrap_or(false));

        let device_extension_names_raw = [khr::Swapchain::name().as_ptr()];

        let features = vk::PhysicalDeviceFeatures {
            shader_clip_distance: 1,
            ..Default::default()
        };

        let priorities = [1.0];

        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index as u32)
            .queue_priorities(&priorities)
            .build()];

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&features);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .unwrap()
        };

        let present_queue = unsafe { device.get_device_queue(queue_family_index as u32, 0) };

        let surface_formats =
            unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }
                .expect("Failed to get surface formats");

        let surface_format = surface_formats
            .iter()
            .map(|sfmt| match sfmt.format {
                vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
                    format: vk::Format::B8G8R8_UNORM,
                    color_space: sfmt.color_space,
                },
                _ => *sfmt,
            })
            .next()
            .expect("Unable to find suitable surface format.");

        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }

        let surface_resolution = match surface_capabilities.current_extent.width {
            std::u32::MAX => surface_extent,
            _ => surface_capabilities.current_extent,
        };

        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)
                .unwrap()
        };

        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_loader = khr::Swapchain::new(&instance, &device);

        let swapchain = {
            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(
                    vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            unsafe {
                swapchain_loader
                    .create_swapchain(&swapchain_create_info, None)
                    .unwrap()
            }
        };

        let pool = {
            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index as u32);

            unsafe { device.create_command_pool(&pool_create_info, None).unwrap() }
        };

        let present_images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };

        let present_image_views: Vec<vk::ImageView> = present_images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                unsafe { device.create_image_view(&create_view_info, None).unwrap() }
            })
            .collect();

        let command_buffers = {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(present_image_views.len() as u32)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            unsafe {
                device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .unwrap()
            }
        };

        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let pipeline_layout = {
            let create_info = vk::PipelineLayoutCreateInfo::default();

            unsafe { device.create_pipeline_layout(&create_info, None) }.unwrap()
        };

        let render_pass = create_render_pass(&device, surface_format.format);

        let pipeline = {
            let vert = include_bytes!(concat!(env!("OUT_DIR"), "/vert.spv"));
            let frag = include_bytes!(concat!(env!("OUT_DIR"), "/frag.spv"));

            let vert_source = ash::util::read_spv(&mut std::io::Cursor::new(&vert[..])).unwrap();
            let frag_source = ash::util::read_spv(&mut std::io::Cursor::new(&frag[..])).unwrap();

            let (vert_module, frag_module) = {
                let vert_create_info = vk::ShaderModuleCreateInfo::builder()
                    .code(&vert_source)
                    .build();

                let frag_create_info = vk::ShaderModuleCreateInfo::builder()
                    .code(&frag_source)
                    .build();
                unsafe {
                    (
                        device
                            .create_shader_module(&vert_create_info, None)
                            .unwrap(),
                        device
                            .create_shader_module(&frag_create_info, None)
                            .unwrap(),
                    )
                }
            };

            let shader_entry_name = CString::new("main").unwrap();

            let vert_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&shader_entry_name)
                .build();

            let frag_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&shader_entry_name)
                .build();

            let shader_create_infos = [vert_stage_create_info, frag_stage_create_info];

            let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
                binding: 0,
                stride: std::mem::size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }];

            let vertex_input_attribute_descriptions = [
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(Vertex, pos) as u32,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(Vertex, color) as u32,
                },
            ];

            let vertex_input_create_info = {
                vk::PipelineVertexInputStateCreateInfo::builder()
                    .vertex_binding_descriptions(&vertex_input_binding_descriptions)
                    .vertex_attribute_descriptions(&vertex_input_attribute_descriptions)
                    .build()
            };

            let vertex_input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            let viewport = [vk::Viewport::builder()
                .x(0.0f32)
                .y(0.0f32)
                .width(surface_resolution.width as f32)
                .height(surface_resolution.height as f32)
                .min_depth(0.0f32)
                .max_depth(1.0f32)
                .build()];

            let scissor = [vk::Rect2D::builder()
                .offset(vk::Offset2D { x: 0, y: 0 })
                .extent(surface_resolution)
                .build()];

            let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
                .viewports(&viewport)
                .scissors(&scissor)
                .build();

            let rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0f32)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false)
                .depth_bias_constant_factor(0.0f32)
                .depth_bias_clamp(0.0f32)
                .depth_bias_slope_factor(0.0f32)
                .build();

            let multi_sampler_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .min_sample_shading(1.0f32)
                .alpha_to_coverage_enable(false)
                .alpha_to_one_enable(false)
                .build();

            let colour_blend_attachment = [vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)
                .build()];

            let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(&colour_blend_attachment)
                .build();

            let pipeline_info = [vk::GraphicsPipelineCreateInfo::builder()
                .stages(&shader_create_infos)
                .vertex_input_state(&vertex_input_create_info)
                .input_assembly_state(&vertex_input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterizer_create_info)
                .multisample_state(&multi_sampler_create_info)
                .color_blend_state(&color_blending)
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0)
                .build()];

            unsafe {
                device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            }
            .unwrap()[0]
        };

        let framebuffers = {
            let mut ret = Vec::<vk::Framebuffer>::new();

            for image_view in present_image_views {
                let image_view = [image_view];
                let frame_buffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&image_view)
                    .width(surface_resolution.width)
                    .height(surface_resolution.height)
                    .layers(1)
                    .build();
                ret.push(unsafe { device.create_framebuffer(&frame_buffer_info, None) }.unwrap());
            }
            ret
        };

        let (vertex_buffer, vertex_mem) = create_vertex_buffer(&instance, &device, physical_device);

        for (command_buffer, framebuffer) in command_buffers.iter().zip(framebuffers) {
            let begin_info = vk::CommandBufferBeginInfo::builder().build();

            unsafe { device.begin_command_buffer(*command_buffer, &begin_info) }.unwrap();

            let clear_colour = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0f32, 1.0f32, 0.0f32, 0.0f32],
                },
            }];

            let render_pass_info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass)
                .framebuffer(framebuffer)
                .render_area(vk::Rect2D::builder().extent(surface_resolution).build())
                .clear_values(&clear_colour)
                .build();

            unsafe {
                device.cmd_begin_render_pass(
                    *command_buffer,
                    &render_pass_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    *command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline,
                );
                device.cmd_bind_vertex_buffers(*command_buffer, 0, &[vertex_buffer], &[0]);
                device.cmd_draw(*command_buffer, 3, 1, 0, 0);
                device.cmd_end_render_pass(*command_buffer);
                device.end_command_buffer(*command_buffer)
            }
            .unwrap()
        }

        let mut image_available = Vec::<vk::Semaphore>::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished = Vec::<vk::Semaphore>::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight_fence = Vec::<vk::Fence>::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut image_in_flight = Vec::<vk::Fence>::with_capacity(present_images.len());
        for _ in 0..present_images.len() {
            image_in_flight.push(vk::Fence::null())
        }

        let semaphore_info = vk::SemaphoreCreateInfo::builder().build();

        let fence_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED)
            .build();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                image_available.push(device.create_semaphore(&semaphore_info, None).unwrap());
                render_finished.push(device.create_semaphore(&semaphore_info, None).unwrap());
                in_flight_fence.push(device.create_fence(&fence_info, None).unwrap());
            }
        }

        AshCtx {
            instance,
            debug_utils_loader,
            debug_call_back,
            device,
            swapchain_loader,
            swapchain,
            present_queue,
            command_buffers,
            image_available,
            render_finished,
            in_flight_fence,
            image_in_flight,
            current_frame: 0,
            data,
        }
    }
}

impl<W> Drop for AshCtx<W> {
    fn drop(&mut self) {
        unsafe {
            //self.debug_utils_loader
            //    .destroy_debug_utils_messenger(self.debug_call_back, None);
            //self.instance.destroy_instance(None);
        }
    }
}

impl<W> RenderCtx for AshCtx<W> {
    fn render_windows(
        &mut self,
        token: CompositorToken<Roles>,
        window_map: Rc<
            RefCell<WindowMap<Roles, for<'r> fn(&'r SurfaceAttributes) -> Option<(i32, i32)>>>,
        >,
    ) {
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fence[self.current_frame]], true, u64::MAX)
                .unwrap();

            let image_index = self
                .swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available[self.current_frame],
                    vk::Fence::null(),
                )
                .unwrap()
                .0 as usize;

            if self.image_in_flight[image_index] != vk::Fence::null() {
                self.device
                    .wait_for_fences(&[self.image_in_flight[image_index]], true, u64::MAX)
                    .unwrap();
            }
            self.image_in_flight[image_index] = self.in_flight_fence[self.current_frame];

            let wait_semaphores = [self.image_available[self.current_frame]];
            let signal_semaphores = [self.render_finished[self.current_frame]];

            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&[self.command_buffers[image_index]])
                .signal_semaphores(&signal_semaphores)
                .build();

            self.device
                .reset_fences(&[self.in_flight_fence[self.current_frame]])
                .unwrap();
            self.device
                .queue_submit(
                    self.present_queue,
                    &[submit_info],
                    self.in_flight_fence[self.current_frame],
                )
                .unwrap();

            let swapchains = [self.swapchain];
            let indices = [image_index as u32];

            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&indices)
                .build();

            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
                .unwrap();
        };

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}
