
use smithay::wayland::compositor::{CompositorToken, SurfaceAttributes};

use std::{
	rc::Rc,
	cell::RefCell,
};

use crate::{
	ctx::RenderCtx,
	window_map::WindowMap,
	shell::Roles,
};

struct AshCtx {


}

impl RenderCtx for AshCtx {
    fn render_windows(
		&mut self,
        token: CompositorToken<Roles>,
        window_map: Rc<RefCell<WindowMap<
            Roles,
            for<'r> fn(&'r SurfaceAttributes) -> Option<(i32, i32)>,
        >>>,
    ) {

	}
}
