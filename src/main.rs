mod kbd;
mod wl;

mod ash;
mod ctx;
mod vk;

mod shell;
mod window_map;

use std::time::Duration;

use std::{cell::RefCell, rc::Rc};

use vk::VkCtx;
use wl::WlCtx;

use std::env;

use calloop::EventLoop;
use calloop::LoopHandle;
use smithay::reexports::wayland_server::Display;

fn warvk<W>(vk_ctx: Box<dyn ctx::RenderCtx>)
where
    W: Send + Sync + 'static,
{
    let mut event_loop = EventLoop::new().expect("Failed to create EventLoop");

    let display = Rc::new(RefCell::new(Display::new()));

    let kbd_rx = kbd::init(event_loop.handle());

    let mut should_close = false;

    //let vk_ctx: Box<dyn crate::ctx::RenderCtx> = Box::new(vk_ctx);

    let vk_ctx = Rc::new(RefCell::new(vk_ctx));
    let wl_ctx = Rc::new(RefCell::new(WlCtx::init(event_loop.handle())));

    let mut ctx = ctx::Ctx {
        vk_ctx: vk_ctx.clone(),
        wl_ctx: wl_ctx.clone(),
    };

    let start_time = std::time::Instant::now();

    while !should_close {
        if let Ok(event) = kbd_rx.try_recv() {
            if kbd::is_key_press(event.value) {
                let text = kbd::get_key_text(event.code, 0);
                if text == "<ESC>" {
                    should_close = true;
                }
            }
        }

        wl_ctx
            .clone()
            .borrow_mut()
            .window_map
            .borrow_mut()
            .send_frames(start_time.elapsed().as_millis() as u32);
        wl_ctx
            .clone()
            .borrow_mut()
            .display
            .borrow_mut()
            .flush_clients(&mut ctx);
        ctx.run();

        wl_ctx
            .clone()
            .borrow_mut()
            .display
            .borrow_mut()
            .flush_clients(&mut ctx);

        let _ = event_loop.dispatch(Some(Duration::from_millis(16)), &mut ctx);

        //wl_ctx.borrow_mut().run(&mut ctx);
        //vk_ctx.borrow_mut().run();
    }
}

fn main() {
    println!("╔══ WaRVk\n║ A Vulkan based Wayland compositor\n║ Written in Rust");

    let args: Vec<String> = env::args().collect();

    let mut prefer_discrete = false;
    let mut winit = false;

    for arg in args {
        match arg.as_str() {
            "--winit" => {
                winit = true;
            }
            "--discrete" => {
                prefer_discrete = true;
            }
            _ => {
                println!("Unsupported option {}", arg);
            }
        }
    }

    let prefer_discrete = prefer_discrete;
    let winit = winit;

    //let ash_ctx = ash::AshCtx::init();

    warvk::<()>(if winit {
        Box::new(VkCtx::<winit::window::Window>::init(prefer_discrete))
    } else {
        Box::new(VkCtx::<()>::init(prefer_discrete))
    });

    //warvk::<()>(Box::new(ash_ctx));
}
