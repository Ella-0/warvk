use std::{
    cell::RefCell,
	borrow::Cow,
    ffi::{CStr, CString},
    rc::Rc,
};

use ash::{
	version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
	Entry,
	extensions::{
		khr, ext
	},
	Instance,
	vk
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

		let surface_exts = [khr::Surface::name(), khr::Display::name(), ext::DebugUtils::name()];
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
				entry.create_instance(&create_info, None)
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


        AshCtx {
			instance,
			debug_utils_loader,
			debug_call_back
		}
    }
}

impl Drop for AshCtx {
	fn drop(&mut self) {
		unsafe {
			self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_call_back, None);
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
