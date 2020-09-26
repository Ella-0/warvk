use wayland_server::{
    protocol::{wl_compositor, wl_subcompositor},
    Display, Filter, Main,
};

use calloop::EventLoop;

struct LoopState {
    socket: std::ffi::OsString,
    display: Display,
}

pub fn implement_compositor(
    compositor: Main<wl_compositor::WlCompositor>,
) -> wl_compositor::WlCompositor {
    compositor.quick_assign(move |_compositor, request, _| match request {
        wl_compositor::Request::CreateSurface { id } => {
            println!("Creating a new wl_surface.");
        }
        wl_compositor::Request::CreateRegion { id } => {
            println!("Creating a new wl_region.");
        }
        _ => unreachable!(),
    });
    (*compositor).clone()
}

pub fn implement_sub_compositor(
    sub_compositor: Main<wl_subcompositor::WlSubcompositor>,
) -> wl_subcompositor::WlSubcompositor {
    sub_compositor.quick_assign(move |subcompositor, request, _| match request {
        wl_subcompositor::Request::GetSubsurface {
            id,
            surface,
            parent,
        } => {}
        wl_subcompositor::Request::Destroy => {}
        _ => unreachable!(),
    });
    (*sub_compositor).clone()
}

pub fn init() {
    println!("╠══ Init Wayland");
    let mut display = Display::new();
    let socket = display.add_socket_auto().expect("Failed to add socket");
    println!("║ Using socket {:#?}", socket,);

    std::env::set_var("WAYLAND_DISPLAY", socket);

    let event_loop = EventLoop::<LoopState>::new().expect("Failed to create event loop");

    let compositor = display.create_global(
        4,
        Filter::new(move |(new_compositor, _version), _, _| {
            implement_compositor(new_compositor);
        }),
    );

    let sub_compositor = display.create_global(
        1,
        Filter::new(move |(new_sub_compositor, _version), _, _| {
            implement_sub_compositor(new_sub_compositor);
        }),
    );

    std::process::Command::new("alacritty")
        .spawn()
        .expect("Failed to spawn");
}
