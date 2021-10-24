mod config;
mod window;
use super::{BorderAtariAct, BorderAtariObs};
use anyhow::Result;
use atari_env::{AtariAction, AtariEnv, EmulatorConfig};
use border_core::{record::Record, Env, Info, Obs, Step};
use config::BorderAtariEnvConfig;
use image::{
    imageops::{grayscale, resize, FilterType::Triangle},
    ImageBuffer, Luma, Rgb,
};
use std::default::Default;
use window::AtariWindow;
use winit::{event_loop::ControlFlow, platform::run_return::EventLoopExtRunReturn};
use std::ptr::copy;

/// Empty struct.
pub struct NullInfo;

impl Info for NullInfo {}

fn env() -> AtariEnv {
    AtariEnv::new(
        dirs::home_dir()
            .unwrap()
            .join(".local/lib/python3.9/site-packages/atari_py/atari_roms/space_invaders.bin"),
            // .join(".local/lib/python3.9/site-packages/atari_py/atari_roms/pong.bin"),
        EmulatorConfig {
            // display_screen: true,
            // sound: true,
            frame_skip: 1,
            color_averaging: false,
            ..EmulatorConfig::default()
        },
    )
}

/// A wrapper of atari learning environment.
///
/// Preprocessing is the same in the link:
/// https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html#stable_baselines3.common.atari_wrappers.AtariWrapper.
pub struct BorderAtariEnv {
    // True for training mode, it affects preprocessing at every steps.
    train: bool,

    // Environment
    env: AtariEnv,

    // Window for displaying the current game state
    window: Option<AtariWindow>,

    // Observation buffer for frame skipping
    obs_buffer: [Vec<u8>; 2],

    // Lives in the game
    lives: usize,

    // If the game was done.
    was_real_done: bool,

    // Buffer for stacking frames
    frames: Vec<u8>,
}

impl BorderAtariEnv {
    /// Opens window for display.
    pub fn open(&mut self) -> Result<()> {
        // Do nothing if a window is already opened.
        if !self.window.is_none() {
            return Ok(());
        }

        self.window = Some(AtariWindow::new(&self.env)?);

        Ok(())
    }

    /// Returns the number of actions.
    pub fn get_num_actions_atari(&self) -> i64 {
        self.env.available_actions().len() as i64
    }

    fn episodic_life_env_step(&mut self, a: &BorderAtariAct) -> (Vec<u8>, f32, i8) {
        let actions = self.env.available_actions();
        let ix = a.act;
        let reward = self.env.step(actions[ix as usize]) as f32;
        let mut done = self.env.is_game_over();
        self.was_real_done = done;
        let lives = self.env.lives();

        if self.train && lives < self.lives && lives > 0 {
            done = true;
            self.lives = lives;
        }

        let done = if done { 1 } else { 0 };
        let (w, h) = (self.env.width(), self.env.height());
        let mut obs = vec![0u8; w * h * 3];
        self.env.render_rgb24(&mut obs);

        (obs, reward, done)
    }

    fn skip_and_max(&mut self, a: &BorderAtariAct) -> (Vec<u8>, f32, Vec<i8>) {
        let mut total_reward = 0f32;
        let mut done = 0;

        for i in 0..4 {
            let (obs, reward, done_) = self.episodic_life_env_step(a);
            total_reward += reward;
            done = done_;
            if i == 2 {
                self.obs_buffer[0] = obs;
            } else if i == 3 {
                self.obs_buffer[1] = obs;
            }
            if done_ == 1 {
                break;
            }
        }

        // Max pooling
        let obs = self.obs_buffer[0]
            .iter()
            .zip(self.obs_buffer[1].iter())
            .map(|(&a, &b)| a.max(b))
            .collect::<Vec<_>>();

        (obs, total_reward, vec![done])
    }

    fn clip_reward(&self, r: f32) -> Vec<f32> {
        if self.train {
            vec![r]
        } else {
            vec![r.signum()]
        }
    }

    fn warp_and_grayscale(w: u32, h: u32, obs: Vec<u8>) -> Vec<u8> {
        // `obs.len()` is w * h * 3 where (w, h) is the size of the frame.
        let img = ImageBuffer::<Rgb<_>, _>::from_vec(w, h, obs).unwrap();
        let img = resize(&img, 84, 84, Triangle);
        let img: ImageBuffer::<Luma<u8>, _> = grayscale(&img);
        let buf = img.to_vec();
        assert_eq!(buf.len(), 84 * 84);
        buf
    }

    fn stack_frame(&mut self, obs: Vec<u8>) {
        unsafe {
            let src: *const u8 = &self.frames[1 * 84 * 84];
            let dst: *mut u8 = &mut self.frames[0];
            copy(src, dst, 3 * 84 * 84);

            let src: *const u8 = &obs[0];
            let dst: *mut u8 = &mut self.frames[3 * 84 * 84];
            copy(src, dst, 1 * 84 * 84);
        }
    }
}

impl Default for BorderAtariEnv {
    fn default() -> Self {
        Self {
            train: false,
            env: env(),
            window: None,
            obs_buffer: [vec![], vec![]],
            lives: 0,
            was_real_done: true,
            frames: vec![0; 4 * 84 * 84],
        }
    }
}

impl Env for BorderAtariEnv {
    type Config = BorderAtariEnvConfig;
    type Obs = BorderAtariObs;
    type Act = BorderAtariAct;
    type Info = NullInfo;

    fn build(_config: &Self::Config, _seed: i64) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            train: false,
            env: env(),
            window: None,
            obs_buffer: [vec![], vec![]],
            lives: 0,
            was_real_done: true,
            frames: vec![0; 4 * 84 * 84],
        })
    }

    fn reset(&mut self, _is_done: Option<&Vec<i8>>) -> Result<Self::Obs> {
        if self.was_real_done {
            self.env.reset();
            // println!("RESET");
        } else {
            // no-op step to advance from terminal/lost life state
            self.env.step(AtariAction::Noop);
        }

        // TODO: noop random steps (?)

        self.was_real_done = false;
        self.lives = self.env.lives();

        let (w, h) = (self.env.width(), self.env.height());
        let mut obs = vec![0u8; w * h * 3];
        self.env.render_rgb24(&mut obs);
        self.obs_buffer[0] = obs.clone();
        self.obs_buffer[1] = obs.clone();

        let obs = Self::warp_and_grayscale(w as u32, h as u32, obs);

        unsafe {
            let src: *const u8 = &obs[0];
            for i in 0..4 {
                let dst: *mut u8 = &mut self.frames[i * 84 * 84];
                copy(src, dst, 84 * 84);
            }
        }

        Ok(self.frames.clone().into())
    }

    fn step_with_reset(
        &mut self,
        a: &Self::Act,
    ) -> (border_core::Step<Self>, border_core::record::Record)
    where
        Self: Sized,
    {
        unimplemented!();
    }

    fn step(&mut self, act: &Self::Act) -> (border_core::Step<Self>, border_core::record::Record)
    where
        Self: Sized,
    {
        let (obs, reward, is_done) = self.skip_and_max(act);
        let (w, h) = (self.env.width() as u32, self.env.height() as u32);
        let obs = Self::warp_and_grayscale(w, h, obs);
        let reward = self.clip_reward(reward); // in training
        self.stack_frame(obs);
        let step = Step::new(
            self.frames.clone().into(),
            act.clone(),
            reward,
            is_done,
            NullInfo,
            Self::Obs::dummy(1),
        );
        let record = Record::empty();

        if let Some(window) = self.window.as_mut() {
            window.event_loop.run_return(|_event, _, control_flow| {
                *control_flow = ControlFlow::Exit;
            });
            self.env.render_rgb32(window.get_frame());
            window.render_and_request_redraw();
        }

        (step, record)
    }
}
