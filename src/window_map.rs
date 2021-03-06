use smithay::{
    reexports::wayland_server::{protocol::wl_surface, Resource},
    utils::Rectangle,
    wayland::{
        compositor::{
            roles::Role, CompositorToken, SubsurfaceRole, SurfaceAttributes, TraversalAction,
        },
        shell::{
            legacy::{ShellSurface, ShellSurfaceRole},
            xdg::{ToplevelSurface, XdgSurfaceRole},
        },
    },
};

pub enum Kind<R> {
    Xdg(ToplevelSurface<R>),
    Wl(ShellSurface<R>),
}

impl<R> Kind<R>
where
    R: Role<SubsurfaceRole> + Role<XdgSurfaceRole> + Role<ShellSurfaceRole> + 'static,
{
    pub fn alive(&self) -> bool {
        match *self {
            Kind::Xdg(ref t) => t.alive(),
            Kind::Wl(ref t) => t.alive(),
        }
    }
    pub fn get_surface(&self) -> Option<&wl_surface::WlSurface> {
        match *self {
            Kind::Xdg(ref t) => t.get_surface(),
            Kind::Wl(ref t) => t.get_surface(),
        }
    }
}

struct Window<R> {
    location: (i32, i32),
    surface: Rectangle,
    toplevel: Kind<R>,
}

impl<R> Window<R>
where
    R: Role<SubsurfaceRole> + Role<XdgSurfaceRole> + Role<ShellSurfaceRole> + 'static,
{
    // Find the topmost surface under this point if any and the location of this surface

    fn matching<F>(
        &self,
        point: (f64, f64),
        ctoken: CompositorToken<R>,
        get_size: F,
    ) -> Option<(wl_surface::WlSurface, (f64, f64))>
    where
        F: Fn(&SurfaceAttributes) -> Option<(i32, i32)>,
    {
        if !self.surface.contains((point.0 as i32, point.1 as i32)) {
            return None;
        }
        // need to check more carefully
        let found = std::cell::RefCell::new(None);
        if let Some(wl_surface) = self.toplevel.get_surface() {
            ctoken.with_surface_tree_downward(
                wl_surface,
                self.location,
                |wl_surface, attributes, role, &(mut x, mut y)| {
                    let data = attributes
                        .user_data
                        .get::<std::cell::RefCell<crate::shell::SurfaceData>>();
                    if let Some((w, h)) = get_size(attributes) {
                        if let Ok(subdata) = Role::<SubsurfaceRole>::data(role) {
                            x += subdata.location.0;
                            y += subdata.location.1;
                        }

                        let my_rect = Rectangle {
                            x,
                            y,
                            width: w,
                            height: h,
                        };

                        if my_rect.contains((point.0 as i32, point.1 as i32)) {
                            *found.borrow_mut() =
                                Some((wl_surface.clone(), (my_rect.x as f64, my_rect.y as f64)));
                        }

                        TraversalAction::DoChildren((x, y))
                    } else {
                        TraversalAction::SkipChildren
                    }
                },
                |_, _, _, _| {},
                |_, _, _, _| {
                    // only continue if the point is not found
                    found.borrow().is_none()
                },
            );
        }
        found.into_inner()
    }

    fn self_update<F>(&mut self, ctoken: CompositorToken<R>, get_size: F)
    where
        F: Fn(&SurfaceAttributes) -> Option<(i32, i32)>,
    {
        let (base_x, base_y) = self.location;
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (base_x, base_y, base_x, base_y);
        if let Some(wl_surface) = self.toplevel.get_surface() {
            let _ = ctoken.with_surface_tree_downward(
                wl_surface,
                (base_x, base_y),
                |_, attributes, role, &(mut x, mut y)| {
                    if let Some((w, h)) = get_size(attributes) {
                        if let Ok(subdata) = Role::<SubsurfaceRole>::data(role) {
                            x += subdata.location.0;
                            y += subdata.location.1;
                        }
                        // update the bounding box
                        if x < min_x {
                            min_x = x;
                        }
                        if y < min_y {
                            min_y = y;
                        }
                        if x + w > max_x {
                            max_x = x + w;
                        }
                        if y + h > max_y {
                            max_y = y + w;
                        }
                        TraversalAction::DoChildren((x, y))
                    } else {
                        TraversalAction::SkipChildren
                    }
                },
                |_, _, _, _| {},
                |_, _, _, _| true,
            );
        }
        self.surface = Rectangle {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        };
    }

    pub fn send_frame(&self, time: u32, ctoken: CompositorToken<R>) {
        if let Some(wl_surface) = self.toplevel.get_surface() {
            ctoken.with_surface_tree_downward(
                wl_surface,
                (),
                |_, _, _, &()| TraversalAction::DoChildren(()),
                |_, attributes, _, &()| {
                    // the surface may not have any user_data if it is a subsurface and has not
                    // yet been commited
                    if let Some(data) = attributes
                        .user_data
                        .get::<std::cell::RefCell<crate::shell::SurfaceData>>()
                    {
                        data.borrow_mut().send_frame(time)
                    }
                },
                |_, _, _, &()| true,
            );
        }
    }
}

pub struct WindowMap<R, F> {
    ctoken: CompositorToken<R>,
    windows: Vec<Window<R>>,
    get_size: F,
}

impl<R, F> WindowMap<R, F>
where
    F: Fn(&SurfaceAttributes) -> Option<(i32, i32)>,
    R: Role<SubsurfaceRole> + Role<XdgSurfaceRole> + Role<ShellSurfaceRole> + 'static,
{
    pub fn new(ctoken: CompositorToken<R>, get_size: F) -> WindowMap<R, F> {
        WindowMap {
            ctoken,
            windows: Vec::new(),
            get_size,
        }
    }

    pub fn insert(&mut self, toplevel: Kind<R>, location: (i32, i32)) {
        let mut window = Window {
            location,
            surface: Rectangle {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            toplevel,
        };
        window.self_update(self.ctoken, &self.get_size);
        self.windows.insert(0, window);
    }

    pub fn get_surface_under(
        &self,
        point: (f64, f64),
    ) -> Option<(wl_surface::WlSurface, (f64, f64))> {
        for w in &self.windows {
            if let Some(surface) = w.matching(point, self.ctoken, &self.get_size) {
                return Some(surface);
            }
        }
        None
    }

    pub fn get_surface_and_bring_to_top(
        &mut self,
        point: (f64, f64),
    ) -> Option<(wl_surface::WlSurface, (f64, f64))> {
        let mut found = None;
        for (i, w) in self.windows.iter().enumerate() {
            if let Some(surface) = w.matching(point, self.ctoken, &self.get_size) {
                found = Some((i, surface));
                break;
            }
        }
        if let Some((i, surface)) = found {
            let winner = self.windows.remove(i);
            self.windows.insert(0, winner);
            Some(surface)
        } else {
            None
        }
    }

    pub fn with_windows_from_bottom_to_top<Func>(&self, mut f: Func)
    where
        Func: FnMut(&Kind<R>, (i32, i32)),
    {
        for w in self.windows.iter().rev() {
            f(&w.toplevel, w.location)
        }
    }

    pub fn refresh(&mut self) {
        self.windows.retain(|w| w.toplevel.alive());
        for w in &mut self.windows {
            w.self_update(self.ctoken, &self.get_size);
        }
    }

    pub fn clear(&mut self) {
        self.windows.clear();
    }

    pub fn send_frames(&self, time: u32) {
        for window in &self.windows {
            window.send_frame(time, self.ctoken);
        }
    }
}
