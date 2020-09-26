use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState},
    device::{Device, DeviceExtensions},
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::{ImageUsage, SwapchainImage},
    instance::{Instance, InstanceExtensions, PhysicalDevice, PhysicalDeviceType, QueueFamily},
    pipeline::{viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract},
    swapchain,
    swapchain::{
        display::{Display, DisplayMode, DisplayPlane},
        AcquireError, ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform,
        Swapchain, SwapchainCreationError,
    },
    sync,
    sync::{FlushError, GpuFuture},
};

use std::sync::Arc;

mod kbd;
mod wlr;

mod vk;

use vk::VkCtx;

fn main() {
    println!("╔══ WaRVk\n║ A Vulkan based Wayland compositor\n║ Written in Rust");

    let kbd_rx = kbd::init();

    wlr::init();

    let mut should_close = false;

	let mut vk_ctx = VkCtx::init();

    while !should_close {
        if let Ok(event) = kbd_rx.try_recv() {
            if kbd::is_key_press(event.value) {
                let text = kbd::get_key_text(event.code, 0);
                if text == "<ESC>" {
                    should_close = true;
                }
            }
        }

		vk_ctx.run();
    }
}
