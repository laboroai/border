mod config;
mod window;
use super::BorderAtariAct;
use super::{BorderAtariActFilter, BorderAtariObsFilter};
use crate::atari_env::{AtariAction, AtariEnv, EmulatorConfig};
use anyhow::Result;
use border_core::{record::Record, Act, Env, Info, Obs, Step};
pub use config::BorderAtariEnvConfig;
use image::{
    imageops::{/*grayscale,*/ resize, FilterType::Triangle},
    ImageBuffer, /*Luma,*/ Rgb,
};
use itertools::izip;
use std::ptr::copy;
use std::{default::Default, marker::PhantomData};
use window::AtariWindow;
#[cfg(feature = "atari-env-sys")]
use winit::{event_loop::ControlFlow, platform::run_return::EventLoopExtRunReturn};

/// Empty struct.
pub struct NullInfo;

impl Info for NullInfo {}

fn env(rom_dir: &str, name: &str) -> AtariEnv {
    AtariEnv::new(
        rom_dir.to_string() + format!("/{}.bin", name).as_str(),
        EmulatorConfig {
            // display_screen: true,
            // sound: true,
            frame_skip: 1,
            color_averaging: false,
            repeat_action_probability: 0.0,
            ..EmulatorConfig::default()
        },
    )
}

/// A wrapper of atari learning environment.
///
/// Preprocessing is the same in the link:
/// <https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html#stable_baselines3.common.atari_wrappers.AtariWrapper>.
pub struct BorderAtariEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: BorderAtariObsFilter<O>,
    AF: BorderAtariActFilter<A>,
{
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

    // Filters
    obs_filter: OF,
    act_filter: AF,
    phantom: PhantomData<(O, A)>,
}

impl<O, A, OF, AF> BorderAtariEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: BorderAtariObsFilter<O>,
    AF: BorderAtariActFilter<A>,
{
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
        // self.env.available_actions().len() as i64
        self.env.minimal_actions().len() as i64
    }

    fn episodic_life_env_step(&mut self, a: &BorderAtariAct) -> (Vec<u8>, f32, i8) {
        let actions = self.env.minimal_actions();
        let ix = a.act;
        let reward = self.env.step(actions[ix as usize]) as f32;
        let mut done = self.env.is_game_over();
        self.was_real_done = done;
        let lives = self.env.lives();

        if self.train && lives < self.lives && lives > 0 {
            done = true;
        }
        self.lives = lives;

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
            if r == 0.0 {
                vec![0.0]
            } else {
                vec![r.signum()]
            }
        } else {
            vec![r]
        }
    }

    fn warp_and_grayscale(w: u32, h: u32, obs: Vec<u8>) -> Vec<u8> {
        // `obs.len()` is w * h * 3 where (w, h) is the size of the frame.
        let img = ImageBuffer::<Rgb<_>, _>::from_vec(w, h, obs).unwrap();
        let img = resize(&img, 84, 84, Triangle);
        let buf = {
            let buf = img.to_vec();
            let i1 = buf.iter().step_by(3);
            let i2 = buf.iter().skip(1).step_by(3);
            let i3 = buf.iter().skip(2).step_by(3);
            izip![i1, i2, i3]
                .map(|(&b, &g, &r)| {
                    ((0.299 * r as f32) + (0.587 * g as f32) + (0.114 * b as f32)) as u8
                })
                .collect::<Vec<_>>()
        };
        // let buf = {
        //     let img: ImageBuffer<Luma<u8>, _> = grayscale(&img);
        //     img.to_vec()
        // };
        assert_eq!(buf.len(), 84 * 84);
        buf
    }

    fn stack_frame(&mut self, obs: Vec<u8>) {
        unsafe {
            let src: *const u8 = &self.frames[0];
            let dst: *mut u8 = &mut self.frames[1 * 84 * 84];
            copy(src, dst, 3 * 84 * 84);

            let src: *const u8 = &obs[0];
            let dst: *mut u8 = &mut self.frames[0];
            copy(src, dst, 1 * 84 * 84);
        }
    }
}

impl<O, A, OF, AF> Default for BorderAtariEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: BorderAtariObsFilter<O>,
    AF: BorderAtariActFilter<A>,
{
    fn default() -> Self {
        let config = BorderAtariEnvConfig::<O, A, OF, AF>::default();

        Self {
            train: false,
            env: env(config.rom_dir.as_str(), "pong"),
            window: None,
            obs_buffer: [vec![], vec![]],
            lives: 0,
            was_real_done: true,
            frames: vec![0; 4 * 84 * 84],
            obs_filter: OF::build(&config.obs_filter_config).unwrap(),
            act_filter: AF::build(&config.act_filter_config).unwrap(),
            phantom: PhantomData,
        }
    }
}

impl<O, A, OF, AF> Env for BorderAtariEnv<O, A, OF, AF>
where
    O: Obs,
    A: Act,
    OF: BorderAtariObsFilter<O>,
    AF: BorderAtariActFilter<A>,
{
    type Config = BorderAtariEnvConfig<O, A, OF, AF>;
    type Obs = O;
    type Act = A;
    type Info = NullInfo;

    fn build(config: &Self::Config, _seed: i64) -> Result<Self>
    where
        Self: Sized,
    {
        let mut env = Self {
            train: config.train,
            env: env(config.rom_dir.as_str(), config.name.as_str()),
            window: None,
            obs_buffer: [vec![], vec![]],
            lives: 0,
            was_real_done: true,
            frames: vec![0; 4 * 84 * 84],
            obs_filter: OF::build(&config.obs_filter_config)?,
            act_filter: AF::build(&config.act_filter_config)?,
            phantom: PhantomData,
        };

        if config.render {
            let _ = env.open();
        }

        Ok(env)
    }

    fn reset(&mut self, _is_done: Option<&Vec<i8>>) -> Result<Self::Obs> {
        if self.was_real_done {
            self.env.reset();
            // println!("RESET");
        } else {
            // no-op step to advance from terminal/lost life state
            self.env.step(AtariAction::Noop);

            let n = fastrand::u8(0..=30);
            for _ in 0..n {
                self.env.step(AtariAction::Noop);
            }
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

        Ok(self.obs_filter.filt(self.frames.clone().into()).0)
    }

    fn reset_with_index(&mut self, ix: usize) -> Result<Self::Obs> {
        self.env.seed(ix as i32);
        self.reset(None)
    }

    /// Currently it supports non-vectorized environment.
    fn step_with_reset(
        &mut self,
        a: &Self::Act,
    ) -> (border_core::Step<Self>, border_core::record::Record)
    where
        Self: Sized,
    {
        let (step, record) = self.step(a);
        assert_eq!(step.is_done.len(), 1);
        let step = if step.is_done[0] == 1 {
            let init_obs = self.reset(None).unwrap();
            Step {
                act: step.act,
                obs: step.obs,
                reward: step.reward,
                is_done: step.is_done,
                info: step.info,
                init_obs,
            }
        } else {
            step
        };

        (step, record)
    }

    fn step(&mut self, act: &Self::Act) -> (border_core::Step<Self>, border_core::record::Record)
    where
        Self: Sized,
    {
        #[cfg(feature = "atari-env-sys")]
        {
            let act_org = act.clone();
            let (act, _record) = self.act_filter.filt(act_org.clone());
            let (obs, reward, is_done) = self.skip_and_max(&act);
            let (w, h) = (self.env.width() as u32, self.env.height() as u32);
            let obs = Self::warp_and_grayscale(w, h, obs);
            let reward = self.clip_reward(reward); // in training
            self.stack_frame(obs);
            let (obs, _record) = self.obs_filter.filt(self.frames.clone().into());
            let step = Step::new(obs, act_org, reward, is_done, NullInfo, Self::Obs::dummy(1));
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

        #[cfg(not(feature = "atari-env-sys"))]
        unimplemented!();
    }
}
