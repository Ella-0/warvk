use wayland_server::{protocol::wl_compositor, Display};

use std::os::unix::io::FromRawFd;

pub fn init() {
    println!("╠══ Init Wayland");
    let mut display = Display::new();
    println!(
        "║ Using socket {:#?}",
        display.add_socket_auto().expect("Failed to add socket")
    );
}
