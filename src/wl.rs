use smithay::reexports::wayland_server::{
    protocol::{
        wl_buffer, wl_compositor, wl_data_device_manager, wl_output, wl_shm, wl_shm_pool,
        wl_subcompositor, wl_surface,
    },
    Display, Resource,
};

use calloop::{generic::Generic, LoopHandle};

use smithay::{
    backend::session::{auto::AutoSession, direct::DirectSession, Session},
    wayland::{
        compositor::{compositor_init, CompositorToken, SurfaceAttributes, SurfaceEvent},
        data_device::{
            default_action_chooser, init_data_device, set_data_device_focus, DataDeviceEvent,
        },
        output::{Mode, Output, PhysicalProperties},
        seat::{CursorImageStatus, Seat, XkbConfig},
        shm::init_shm_global,
    },
};

use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use crate::ctx::Ctx;

pub struct WlCtx {
    pub display: Rc<RefCell<Display>>,
    pub compositor_token: CompositorToken<crate::shell::Roles>,
    pub window_map: Rc<
        RefCell<
            crate::window_map::WindowMap<
                crate::shell::Roles,
                for<'r> fn(&'r SurfaceAttributes) -> Option<(i32, i32)>,
            >,
        >,
    >,
}

impl WlCtx {
    pub fn init(loop_handle: LoopHandle<Ctx>) -> WlCtx {
        let mut display = Rc::new(RefCell::new(Display::new()));

        let _wayland_event_source = loop_handle
            .insert_source(
                Generic::from_fd(
                    display.borrow().get_poll_fd(),
                    calloop::Interest::Readable,
                    calloop::Mode::Level,
                ),
                {
                    let display = display.clone();
                    let log = None::<()>;
                    move |_, _, state: &mut Ctx| {
                        let mut display = display.borrow_mut();
                        match display.dispatch(std::time::Duration::from_millis(0), state) {
                            Ok(_) => Ok(()),
                            Err(e) => {
                                println!("I/O error on the Wayland display: {}", e);
                                Err(e)
                            }
                        }
                    }
                },
            )
            .expect("Failed to init the wayland event source.");

        let socket = display
            .borrow_mut()
            .add_socket_auto()
            .expect("Failed to add socket");
        println!("║ Using socket {:#?}", socket);

        std::env::set_var("WAYLAND_DISPLAY", socket);

        init_shm_global(&mut display.borrow_mut(), vec![], None);

        let (compositor_token, _, _, window_map) =
            crate::shell::init_shell(&mut display.borrow_mut());

        let dnd_icon = Arc::new(Mutex::new(None));

        let dnd_icon2 = dnd_icon.clone();

        init_data_device(
            &mut display.borrow_mut(),
            move |event| match event {
                DataDeviceEvent::DnDStarted { icon, .. } => {
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

        let (mut seat, _) = Seat::new(
            &mut display.borrow_mut(),
            "winit".into(),
            compositor_token.clone(),
            None,
        );

        let cursor_status = Arc::new(Mutex::new(CursorImageStatus::Default));

        let cursor_status2 = cursor_status.clone();

        let pointer = seat.add_pointer(compositor_token.clone(), move |new_status| {
            // TODO: hide winit system cursor when relevant
            *cursor_status2.lock().unwrap() = new_status
        });

        /*let keyboard = seat
        .add_keyboard(XkbConfig::default(), 1000, 500, |seat, focus| {
            set_data_device_focus(seat, focus.and_then(|s| s.client()))
        })
        .expect("Failed to initialize the keyboard");*/

        let (output, _) = Output::new(
            &mut display.borrow_mut(),
            "Winit".into(),
            PhysicalProperties {
                width: 0,
                height: 0,
                subpixel: wl_output::Subpixel::Unknown,
                make: "Smithay".into(),
                model: "WaRVk".into(),
            },
            None,
        );

        output.change_current_state(
            Some(Mode {
                width: 1920 as i32,
                height: 1080 as i32,
                refresh: 60_000,
            }),
            None,
            None,
        );

        output.set_preferred(Mode {
            width: 1920 as i32,
            height: 1080 as i32,
            refresh: 60_000,
        });

        let pointer_location = Rc::new(RefCell::new((0.0, 0.0)));

        /*        std::process::Command::new("mpv")
        .arg("--vo=wlshm")
        .arg("https://youtu.be/-W6JfiC-QBk")
        .spawn()
        .expect("Failed to spawn");*/

        std::process::Command::new("alacritty")
            .arg("-e")
            .arg("btm")
            .spawn()
            .expect("Failed to spawn");
        //std::process::Command::new("weston-smoke")
        //    .spawn()
        //    .expect("Failed to spawn");

        //std::process::Command::new("/home/edward/Documents/Programming/hello-wayland/hello-wayland")
        //    .spawn()
        //    .expect("Failed to spawn");

        WlCtx {
            display,
            compositor_token,
            window_map,
        }
    }

    pub fn run(&mut self, ctx: &mut Ctx) {
        //self.display.borrow_mut().flush_clients(ctx);
    }
}

fn surface_contents_update_notify() {}

struct Compositor {}

impl Compositor {}
