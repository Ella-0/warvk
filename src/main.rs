mod kbd;
mod wl;

mod vk;

mod ctx;

use std::time::Duration;

use calloop::EventLoop;

use std::{cell::RefCell, rc::Rc};

use vk::VkCtx;
use wl::WlCtx;

fn main() {
    println!("╔══ WaRVk\n║ A Vulkan based Wayland compositor\n║ Written in Rust");

    let kbd_rx = kbd::init();

    let mut event_loop = EventLoop::new().expect("Failed to create EventLoop");

    let mut should_close = false;

    let mut vk_ctx = Rc::new(RefCell::new(VkCtx::init()));
    let mut wl_ctx = Rc::new(RefCell::new(WlCtx::init(event_loop.handle())));

    while !should_close {
        if let Ok(event) = kbd_rx.try_recv() {
            if kbd::is_key_press(event.value) {
                let text = kbd::get_key_text(event.code, 0);
                if text == "<ESC>" {
                    should_close = true;
                }
            }
        }

        let _ = event_loop.dispatch(
            Duration::from_millis(20),
            &mut ctx::Ctx {
                vk_ctx: vk_ctx.clone(),
                wl_ctx: wl_ctx.clone(),
            },
        );
        wl_ctx.borrow_mut().run(&mut ctx::Ctx {
            vk_ctx: vk_ctx.clone(),
            wl_ctx: wl_ctx.clone(),
        });
        vk_ctx.borrow_mut().run();
    }
}
