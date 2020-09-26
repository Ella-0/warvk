use std::cell::RefCell;
use std::rc::Rc;

use crate::vk::VkCtx;
use crate::wl::WlCtx;

pub struct Ctx {
    pub vk_ctx: Rc<RefCell<VkCtx>>,
    pub wl_ctx: Rc<RefCell<WlCtx>>,
}
