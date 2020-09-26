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

use vk::*;

fn main() {
    println!("╔══ WaRVk\n║ A Vulkan based Wayland compositor\n║ Written in Rust");

    let kbd_rx = kbd::init();

    wlr::init();

    let instance = create_instance();

    let physical = choose_physical_device(&instance);

    let display = choose_display(physical);

    let display_mode = choose_display_mode(display);

    let display_plane = choose_display_plane(physical);

    let surface = Surface::<()>::from_display_mode(&display_mode, &display_plane)
        .expect("Failed to create surface");

    let queue_family = choose_queue_family(physical, surface.clone());

    let (device, mut queues) = create_device(physical, queue_family);

    let queue = queues.next().unwrap();

    let dimensions: [u32; 2] = display_mode.visible_region();

    let (mut swapchain, images) = create_swapchain(
        physical,
        surface.clone(),
        device.clone(),
        queue.clone(),
        dimensions,
    );

    #[derive(Default, Debug, Clone)]
    struct Vertex {
        position: [f32; 2],
    }
    vulkano::impl_vertex!(Vertex, position);

    let vertex_buffer = {
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            [
                Vertex {
                    position: [-0.5, 0.5],
                },
                Vertex {
                    position: [0.0, -0.5],
                },
                Vertex {
                    position: [0.5, 0.5],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap()
    };

    let render_pass = create_render_pass(device.clone(), swapchain.clone());

    let pipeline = create_pipeline::<Vertex>(device.clone(), render_pass.clone());

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
        compare_mask: None,
        write_mask: None,
        reference: None,
    };

    let mut framebuffers = create_framebuffers(&images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut should_close = false;

    while !should_close {
        if let Ok(event) = kbd_rx.try_recv() {
            if kbd::is_key_press(event.value) {
                let text = kbd::get_key_text(event.code, 0);
                if text == "<ESC>" {
                    should_close = true;
                }
            }
        }

        // It is important to call this function from time to time, otherwise resources will keep
        // accumulating and you will eventually reach an out of memory error.
        // Calling this function polls various fences in order to determine what the GPU has
        // already processed, and frees the resources that are no longer needed.
        previous_frame_end.as_mut().unwrap().cleanup_finished();

        // Whenever the window resizes we need to recreate everything dependent on the window size.
        // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
        if recreate_swapchain {
            // Get the new dimensions of the window.
            let dimensions: [u32; 2] = display_mode.visible_region();
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                Ok(r) => r,
                // This error tends to happen when the user is manually resizing the window.
                // Simply restarting the loop is the easiest way to fix this issue.
                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };

            swapchain = new_swapchain;
            // Because framebuffers contains an Arc on the old swapchain, we need to
            // recreate framebuffers as well.
            framebuffers =
                create_framebuffers(&new_images, render_pass.clone(), &mut dynamic_state);
            recreate_swapchain = false;
        }

        // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
        // no image is available (which happens if you submit draw commands too quickly), then the
        // function will block.
        // This operation returns the index of the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is an optional timeout
        // after which the function call will return an error.
        let (image_num, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        // acquire_next_image can be successful, but suboptimal. This means that the swapchain image
        // will still work, but it may not display correctly. With some drivers this can be when
        // the window resizes, but it may not cause the swapchain to become out of date.
        if suboptimal {
            recreate_swapchain = true;
        }

        // Specify the color to clear the framebuffer with i.e. blue
        let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];

        // In order to draw, we have to build a *command buffer*. The command buffer object holds
        // the list of commands that are going to be executed.
        //
        // Building a command buffer is an expensive operation (usually a few hundred
        // microseconds), but it is known to be a hot path in the driver and is expected to be
        // optimized.
        //
        // Note that we have to pass a queue family when we create the command buffer. The command
        // buffer will only be executable on that given queue family.
        let mut builder =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap();

        builder
            // Before we can draw, we have to *enter a render pass*. There are two methods to do
            // this: `draw_inline` and `draw_secondary`. The latter is a bit more advanced and is
            // not covered here.
            //
            // The third parameter builds the list of values to clear the attachments with. The API
            // is similar to the list of attachments when building the framebuffers, except that
            // only the attachments that use `load: Clear` appear in the list.
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            // We are now inside the first subpass of the render pass. We add a draw command.
            //
            // The last two parameters contain the list of resources to pass to the shaders.
            // Since we used an `EmptyPipeline` object, the objects have to be `()`.
            .draw(
                pipeline.clone(),
                &dynamic_state,
                vec![vertex_buffer.clone()],
                (),
                (),
            )
            .unwrap()
            // We leave the render pass by calling `draw_end`. Note that if we had multiple
            // subpasses we could have called `next_inline` (or `next_secondary`) to jump to the
            // next subpass.
            .end_render_pass()
            .unwrap();

        // Finish building the command buffer by calling `build`.
        let command_buffer = builder.build().unwrap();

        let future = previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            // The color output is now expected to contain our triangle. But in order to show it on
            // the screen, we have to *present* the image by calling `present`.
            //
            // This function does not actually present the image immediately. Instead it submits a
            // present command at the end of the queue. This means that it will only be presented once
            // the GPU has finished executing the command buffer that draws the triangle.
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
        }
    }
}
