use std::cell::RefCell;
use std::rc::Rc;

use crate::vk::VkCtx;
use crate::wl::WlCtx;

#[derive(Clone)]
pub struct Ctx<W>
where
    W: Send + Sync + 'static,
{
    pub vk_ctx: Rc<RefCell<VkCtx<W>>>,
    pub wl_ctx: Rc<RefCell<WlCtx>>,
}

impl<W> Ctx<W> where W: Send + Sync + 'static {
	pub fn run(&mut self) {
		// self.wl_ctx.input.dspatch_new_events().unwrap()

		{
			// TODO replace with vulkan shit
            //use glium::Surface;
            //let mut frame = drawer.draw();
            //frame.clear(None, Some((0.8, 0.8, 0.9, 1.0)), false, Some(1.0), None);


			// self.vk_ctx.drawer.draw_windows() or some shit

			let (x, y) = ((), ()); // self.wl_ctx.pointer_location.borrow();

			{
				let guard = (); //self.wl_ctx.dnd_icon.lock().unwrap();
				//draw the icon, see https://github.com/Smithay/smithay/blob/0.2.x/anvil/src/winit.rs#L150
			}

			{
				// drawe the cursor as relevant
			}
		}

		//event_loop.dispatch

		self.wl_ctx.borrow_mut().display.borrow_mut().flush_clients();

		self.wl_ctx.borrow_mut().window_map.borrow_mut().refresh();
	}
}
