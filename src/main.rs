mod kbd;
mod wl;

mod vk;

mod ctx;

mod input_handler;
mod pool;
mod shell;
mod window_map;

use std::time::Duration;

use std::{cell::RefCell, rc::Rc};

use vk::VkCtx;
use wl::WlCtx;

use std::env;

use smithay::reexports::wayland_server::{calloop::EventLoop, Display};

use slog::Drain;

fn warvk<W>(vk_ctx: VkCtx<W>)
where
    W: Send + Sync + 'static,
{
    let mut event_loop = EventLoop::new().expect("Failed to create EventLoop");

    let display = Rc::new(RefCell::new(Display::new(event_loop.handle())));

    let kbd_rx = kbd::init(event_loop.handle());

    let mut should_close = false;

    let vk_ctx = Rc::new(RefCell::new(vk_ctx));
    let wl_ctx = Rc::new(RefCell::new(WlCtx::init(event_loop.handle())));

    let mut ctx = ctx::Ctx {
        vk_ctx: vk_ctx.clone(),
        wl_ctx: wl_ctx.clone(),
    };

    while !should_close {
        if let Ok(event) = kbd_rx.try_recv() {
            if kbd::is_key_press(event.value) {
                let text = kbd::get_key_text(event.code, 0);
                if text == "<ESC>" {
                    should_close = true;
                }
            }
        }

        let _ = event_loop.dispatch(Some(Duration::from_millis(16)), &mut ctx);
        ctx.run();
        //wl_ctx.borrow_mut().run(&mut ctx);
        //vk_ctx.borrow_mut().run();
    }
}

fn main() {
    println!("╔══ WaRVk\n║ A Vulkan based Wayland compositor\n║ Written in Rust");

    let args: Vec<String> = env::args().collect();

    if let Some(arg) = args.get(1) {
        if arg == "--winit" {
            warvk(VkCtx::<winit::window::Window>::init());
        } else {
            panic!("Unsupported");
        }
    } else {
        warvk(VkCtx::<()>::init());
    }
}
