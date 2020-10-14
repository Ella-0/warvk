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

pub struct AshCtx {
    instance: Instance,
    debug_utils_loader: ext::DebugUtils,
    debug_call_back: vk::DebugUtilsMessengerEXT,
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

impl AshCtx {
    pub fn init() -> AshCtx {
        let entry = Entry::new().unwrap();
        let app_name = CString::new("WaRVk").unwrap();
        let engine_name = CString::new("Smithay").unwrap();

        let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
        let layer_names_raw = layer_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        let surface_exts = [
            khr::Surface::name(),
            khr::Display::name(),
            ext::DebugUtils::name(),
        ];
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
        let display_loader = khr::Display::new(&entry, &instance);
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
            std::u32::MAX => mode.parameters.visible_region,
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
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
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
        let command_buffers = {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(2)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            unsafe {
                device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .unwrap()
            }
        };
        let setup_command_buffer = command_buffers[0];
        let draw_command_buffer = command_buffers[1];

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

        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let pipeline_layout = {
            let create_info = vk::PipelineLayoutCreateInfo::default();

            unsafe { device.create_pipeline_layout(&create_info, None) }
        };

        let pipeline = {
            let vert = include_bytes!(concat!(env!("OUT_DIR"), "/vert.spv"));
            let frag = include_bytes!(concat!(env!("OUT_DIR"), "/frag.spv"));

            let mut vert_source = Vec::<u32>::new();

            for i in 0..(vert.len() / 4) {
                vert_source.push(u32::from_ne_bytes([
                    vert[i],
                    vert[i + 1],
                    vert[i + 2],
                    vert[i + 3],
                ]));
            }

            let mut frag_source = Vec::<u32>::new();

            for i in 0..(frag.len() / 4) {
                frag_source.push(u32::from_ne_bytes([
                    frag[i],
                    frag[i + 1],
                    frag[i + 2],
                    frag[i + 3],
                ]));
            }

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

            let vert_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(unsafe { CStr::from_ptr("main".as_ptr() as *const i8) });

            let frag_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(unsafe { CStr::from_ptr("main".as_ptr() as *const i8) });

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
                    .build();
            };
        };

        AshCtx {
            instance,
            debug_utils_loader,
            debug_call_back,
        }
    }
}

impl Drop for AshCtx {
    fn drop(&mut self) {
        unsafe {
            //self.debug_utils_loader
            //    .destroy_debug_utils_messenger(self.debug_call_back, None);
            self.instance.destroy_instance(None);
        }
    }
}

impl RenderCtx for AshCtx {
    fn render_windows(
        &mut self,
        token: CompositorToken<Roles>,
        window_map: Rc<
            RefCell<WindowMap<Roles, for<'r> fn(&'r SurfaceAttributes) -> Option<(i32, i32)>>>,
        >,
    ) {
    }
}
