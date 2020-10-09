use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, Mutex},
};

use smithay::{
    reexports::wayland_server::{
        protocol::{wl_buffer, wl_callback, wl_shell_surface, wl_surface},
        Display, Resource,
    },
    wayland::{
		Serial,
        compositor::{compositor_init, CompositorToken, SurfaceAttributes, SurfaceEvent, BufferAssignment},
        data_device::DnDIconRole,
        seat::CursorImageRole,
        shell::{
            legacy::{
                wl_shell_init, ShellRequest, ShellState as WlShellState, ShellSurfaceKind,
                ShellSurfaceRole,
            },
            xdg::{
                xdg_shell_init, PopupConfigure, ShellState as XdgShellState, ToplevelConfigure,
                XdgRequest, XdgSurfaceRole,
            },
        },
    },
};

use smithay::define_roles;

use crate::window_map::{Kind as SurfaceKind, WindowMap};

define_roles!(Roles =>
    [ XdgSurface, XdgSurfaceRole ]
    [ ShellSurface, ShellSurfaceRole]
    [ DnDIcon, DnDIconRole ]
    [ CursorImage, CursorImageRole ]
);

pub type MyWindowMap = WindowMap<
    Roles,
    fn(&SurfaceAttributes) -> Option<(i32, i32)>,
>;

pub type MyCompositorToken = CompositorToken<Roles>;

pub fn init_shell(
    display: &mut Display,
) -> (
    CompositorToken<Roles>,
    Arc<Mutex<XdgShellState<Roles>>>,
    Arc<Mutex<WlShellState<Roles>>>,
    Rc<RefCell<MyWindowMap>>,
) {
    // Create the compositor
    let (compositor_token, _, _) = compositor_init(
        display,
        move |request, surface, ctoken| match request {
            SurfaceEvent::Commit => surface_commit(&surface, ctoken)
        },
        None,
    );

    // Init a window map, to track the location of our windows
    let window_map = Rc::new(RefCell::new(WindowMap::<_, _>::new(
        compositor_token,
        get_size as _,
    )));

    // init the xdg_shell
    let xdg_window_map = window_map.clone();
    let (xdg_shell_state, _, _) = xdg_shell_init(
        display,
        compositor_token,
        move |shell_event| match shell_event {
            XdgRequest::NewToplevel { surface } => {
                // place the window at a random location in the [0;800]x[0;800] square
                let x = 0;
                let y = 0;
                surface.send_configure(ToplevelConfigure {
                    size: None,
                    states: vec![],
                    serial: Serial::from(42),
                });
                xdg_window_map
                    .borrow_mut()
                    .insert(SurfaceKind::Xdg(surface), (x, y));
            }
            XdgRequest::NewPopup { surface } => surface.send_configure(PopupConfigure {
                size: (10, 10),
                position: (10, 10),
                serial: Serial::from(42),
            }),
            _ => (),
        },
        None,
    );

    // init the wl_shell
    let shell_window_map = window_map.clone();
    let (wl_shell_state, _) = wl_shell_init(
        display,
        compositor_token,
        move |req: ShellRequest<_>| {
            if let ShellRequest::SetKind {
                surface,
                kind: ShellSurfaceKind::Toplevel,
            } = req
            {
                // place the window at a random location in the [0;300]x[0;300] square
                let x = 0;
                let y = 0;
                surface.send_configure((0, 0), wl_shell_surface::Resize::None);
                shell_window_map
                    .borrow_mut()
                    .insert(SurfaceKind::Wl(surface), (x, y));
            }
        },
        None,
    );

    (
        compositor_token,
        xdg_shell_state,
        wl_shell_state,
        window_map,
    )
}

#[derive(Default)]
pub struct SurfaceData {
    pub buffer: Option<wl_buffer::WlBuffer>,
    // make vulkan texture data
    pub texture: bool,
    pub image: Option<Arc<vulkano::image::StorageImage<vulkano::format::Format>>>,
}

fn surface_commit(
    surface: &wl_surface::WlSurface,
    token: CompositorToken<Roles>,
) {
    // we retrieve the contents of the associated buffer and copy it 

    token.with_surface_data(surface, |attributes| {
        attributes
            .user_data
            .insert_if_missing(|| RefCell::new(SurfaceData::default()));

        let mut data = attributes
                .user_data
                .get::<RefCell<SurfaceData>>()
                .unwrap()
                .borrow_mut();

        match attributes.buffer.take() {
            Some(BufferAssignment::NewBuffer {buffer, ..}) => {
                // new contents
                // TODO: handle hotspot coordinates
                data.buffer = Some(buffer);
                data.texture = false;
            }
            Some(BufferAssignment::Removed) => {
                // erase the contents
                data.buffer = None;
                data.texture = false;
            }
            None => {}
        }
    });
}

fn get_size(attrs: &SurfaceAttributes) -> Option<(i32, i32)> {

    let mut data = attrs
            .user_data
            .get::<RefCell<SurfaceData>>()?
            .borrow_mut();

    if data.texture {
        let d = data.image.as_ref().unwrap().dimensions();
        Some((d.width() as i32, d.height() as i32))
    } else {
        None
    }
}
