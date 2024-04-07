use crate::atari_env::AtariEnv;
use anyhow::Result;
#[cfg(feature = "atari-env-sys")]
use {
    pixels::{Pixels, SurfaceTexture},
    winit::{
        event_loop::EventLoop,
        // platform::run_return::EventLoopExtRunReturn,
        window::{Window, WindowBuilder},
    },
};

pub(super) struct AtariWindow {
    #[cfg(feature = "atari-env-sys")]
    pub(super) event_loop: EventLoop<()>,
    #[cfg(feature = "atari-env-sys")]
    window: Window,
    #[cfg(feature = "atari-env-sys")]
    pixels: Pixels<Window>,
}

impl AtariWindow {
    pub fn new(env: &AtariEnv) -> Result<Self> {
        #[cfg(feature = "atari-env-sys")]
        {
            let event_loop = EventLoop::new();
            let window = WindowBuilder::new()
                .with_title("A fantastic window!")
                .with_inner_size(winit::dpi::LogicalSize::new(128.0, 128.0))
                .build(&event_loop)?;
            let surface_texture = SurfaceTexture::new(128, 128, &window);
            let pixels =
                Pixels::new(env.width() as u32, env.height() as u32, surface_texture).unwrap();
            // event_loop.run_return(move |_event, _, _control_flow| {});

            Ok(Self {
                event_loop: event_loop,
                window,
                pixels,
            })
        }

        #[cfg(not(feature = "atari-env-sys"))]
        unimplemented!();
    }

    pub fn get_frame(&mut self) -> &mut [u8] {
        #[cfg(feature = "atari-env-sys")]
        {
            self.pixels.get_frame()
        }

        #[cfg(not(feature = "atari-env-sys"))]
        unimplemented!();
    }

    pub fn render_and_request_redraw(&mut self) {
        #[cfg(feature = "atari-env-sys")]
        {
            self.pixels.render().unwrap();
            self.window.request_redraw();
        }

        #[cfg(not(feature = "atari-env-sys"))]
        unimplemented!();
    }
}
