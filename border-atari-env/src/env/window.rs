use anyhow::Result;
use crate::atari_env::AtariEnv;
// use atari_env::{AtariAction, AtariEnv, EmulatorConfig};
use pixels::{Pixels, SurfaceTexture};
use winit::{
    event_loop::EventLoop,
    // platform::run_return::EventLoopExtRunReturn,
    window::{Window, WindowBuilder},
};

pub(super) struct AtariWindow {
    pub(super) event_loop: EventLoop<()>,
    window: Window,
    pixels: Pixels<Window>,
}

impl AtariWindow {
    pub fn new(env: &AtariEnv) -> Result<Self> {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("A fantastic window!")
            .with_inner_size(winit::dpi::LogicalSize::new(128.0, 128.0))
            .build(&event_loop)?;
        let surface_texture = SurfaceTexture::new(128, 128, &window);
        let pixels = Pixels::new(
            env.width() as u32,
            env.height() as u32,
            surface_texture,
        )
        .unwrap();
        // event_loop.run_return(move |_event, _, _control_flow| {});

        Ok(Self {
            event_loop: event_loop,
            window,
            pixels,
        })
    }

    pub fn get_frame(&mut self) -> &mut [u8]{
        self.pixels.get_frame()
    }

    pub fn render_and_request_redraw(&mut self) {
        self.pixels.render().unwrap();
        self.window.request_redraw();
    }
}
