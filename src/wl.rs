use smithay::reexports::wayland_server::{
    protocol::{wl_compositor, wl_data_device_manager, wl_shm, wl_shm_pool, wl_subcompositor, wl_surface, wl_buffer, wl_output},
    Display, Resource
};

use smithay::reexports::wayland_server::calloop::{
    generic::Generic, LoopHandle
};

use smithay::{
	backend::{session::{auto::AutoSession, direct::DirectSession, Session}},
	wayland::{
		seat::{Seat, CursorImageStatus, XkbConfig},
		compositor::{compositor_init, CompositorToken, SurfaceAttributes, SurfaceEvent},
		shm::init_shm_global,
		data_device::{default_action_chooser, init_data_device, DataDeviceEvent, set_data_device_focus},
		output::{Mode, Output, PhysicalProperties}
	}
};

use std::{cell::RefCell, rc::Rc, sync::{Arc, Mutex}};

use crate::ctx::Ctx;

pub struct WlCtx {
    pub display: Rc<RefCell<Display>>,
	pub compositor_token: CompositorToken<crate::shell::SurfaceData, crate::shell::Roles>,
	pub window_map: Rc<RefCell<crate::window_map::WindowMap<crate::shell::SurfaceData, crate::shell::Roles, (), (), for <'r>
		fn (&'r SurfaceAttributes<crate::shell::SurfaceData>) -> Option<(i32, i32)>
	>>>
}

impl WlCtx {
    pub fn init<W>(loop_handle: LoopHandle<Ctx<W>>) -> WlCtx
    where
        W: Send + Sync + 'static,
    {
        let mut display = Display::new(loop_handle);
        let socket = display.add_socket_auto().expect("Failed to add socket");
        println!("║ Using socket {:#?}", socket);

        std::env::set_var("WAYLAND_DISPLAY", socket);

		init_shm_global(&mut display, vec![], None);

		// TODO init_shell

		// this is temporary

		let (compositor_token, _, _, window_map) = crate::shell::init_shell(&mut display);

        let dnd_icon = Arc::new(Mutex::new(None));

        let dnd_icon2 = dnd_icon.clone();

		init_data_device(
			&mut display,
			move |event| match event {
				DataDeviceEvent::DnDStarted {icon, ..} => {
					*dnd_icon2.lock().unwrap() = icon;
				}
				DataDeviceEvent::DnDDropped => {
					*dnd_icon2.lock().unwrap() = None;
				}
				_ => {}
			},
			default_action_chooser,
			compositor_token.clone(),
			None,
		);

		let (mut seat, _) = Seat::new(&mut display, "winit".into(), compositor_token.clone(), None);

		let cursor_status = Arc::new(Mutex::new(CursorImageStatus::Default));

		let cursor_status2 = cursor_status.clone();
		
		let pointer = seat.add_pointer(compositor_token.clone(), move |new_status| {
        	// TODO: hide winit system cursor when relevant
    	    *cursor_status2.lock().unwrap() = new_status
	    });

        let keyboard = seat
            .add_keyboard(XkbConfig::default(), 1000, 500, |seat, focus| {
                set_data_device_focus(seat, focus.and_then(|s| s.client()))
            })
            .expect("Failed to initialize the keyboard");
        
		let (output, _) = Output::new(
            &mut display,
            "Winit".into(),
            PhysicalProperties {
                width: 0,
                height: 0,
                subpixel: wl_output::Subpixel::Unknown,
                make: "Smithay".into(),
                model: "Winit".into(),
            },
			None
        );

		let pointer_location = Rc::new(RefCell::new((0.0, 0.0)));

        std::process::Command::new("weston-info")
            .spawn()
            .expect("Failed to spawn");
    
        //std::process::Command::new("/home/edward/Documents/Programming/hello-wayland/hello-wayland")
        //    .spawn()
        //    .expect("Failed to spawn");


	    WlCtx { 
			display: Rc::new(RefCell::new(display)), 
			compositor_token,
			window_map,
		}
    }

    pub fn run<W>(&mut self, ctx: &mut Ctx<W>)
    where
        W: Send + Sync + 'static,
    {
        //self.display.borrow_mut().flush_clients(ctx);
    }
}

fn surface_contents_update_notify() {}

struct Compositor {}

impl Compositor {}
