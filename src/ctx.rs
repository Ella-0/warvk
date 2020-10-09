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

use smithay::reexports::wayland_server::protocol::wl_buffer::{self, WlBuffer};
use smithay::wayland::compositor::{roles::Role, SubsurfaceRole, TraversalAction};
impl<W> Ctx<W>
where
    W: Send + Sync + 'static,
{
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

            let wl_ctx = self.wl_ctx.borrow();
            let mut vk_ctx = self.vk_ctx.borrow_mut();

            /*wl_ctx.window_map.borrow().with_windows_from_bottom_to_top(
                |toplevel_surface, initial_place| {
                    if let Some(wl_surface) = toplevel_surface.get_surface() {
                        wl_ctx.compositor_token.with_surface_tree_upward(
                            wl_surface,
                            initial_place,
                            |_surface, attributes, role, &(mut x, mut y)| {
                                // there is actually something to draw !
                                if attributes.user_data.texture.is_none() {
                                    if let Some(buffer) = attributes.user_data.buffer.take() {
                                        attributes.user_data.texture =
                                            Some(vk_ctx.load_shm_buffer_to_image(&buffer.clone()));
                                        // notify the client that we have finished reading the
                                        // buffer
                                        buffer.send(wl_buffer::Event::Release);
                                    }
                                }
                                if let Some(ref metadata) = attributes.user_data.texture {
                                    if let Ok(subdata) = Role::<SubsurfaceRole>::data(role) {
                                        x += subdata.location.0;
                                        y += subdata.location.1;
                                    }
                                    vk_ctx.render_shm_buffer(metadata.clone());
                                    //vk_ctx.run();
                                    TraversalAction::DoChildren((x, y))
                                } else {
                                    // we are not display, so our children are neither
                                    TraversalAction::SkipChildren
                                }
                            },
                        );
                    }
                },
            );*/
            vk_ctx.render_windows(wl_ctx.compositor_token, wl_ctx.window_map.clone());
        }

        //event_loop.dispatch

		let mut m = ();

        /*self.wl_ctx
            .borrow_mut()
            .display
            .borrow_mut()
            .flush_clients(&mut m);*/

        self.wl_ctx.borrow_mut().window_map.borrow_mut().refresh();
    }
}
