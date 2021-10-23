mod config;
mod window;
use super::{BorderAtariAct, BorderAtariObs};
use anyhow::Result;
use atari_env::{AtariAction, AtariEnv, EmulatorConfig};
use border_core::{record::Record, Env, Info, Obs, Step};
use config::BorderAtariEnvConfig;
use rand::{seq::SliceRandom, Rng};
use std::default::Default;
use window::AtariWindow;
use winit::{event_loop::ControlFlow, platform::run_return::EventLoopExtRunReturn};

/// Empty struct.
pub struct NullInfo;

impl Info for NullInfo {}

fn env() -> AtariEnv {
    AtariEnv::new(
        dirs::home_dir()
            .unwrap()
            // .join(".local/lib/python3.9/site-packages/atari_py/atari_roms/space_invaders.bin"),
            .join(".local/lib/python3.9/site-packages/atari_py/atari_roms/pong.bin"),
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
        let mut obs = vec![0u8 ; w * h * 3];
        self.env.render_rgb24(&mut obs);

        (obs, reward, done)
    }

    fn skip_and_max(&mut self, a: &BorderAtariAct) -> (Vec<u8>, f32, Vec<i8>) {
        let mut total_reward = 0f32;
        let mut done = 0;

        for i in 0..4 {
            let (obs, reward, done_) = self.episodic_life_env_step(a);
            total_reward += reward;
            done_ = done;
            if i == 2 {
                self.obs_buffer[0] = obs;
            } else if i == 3 {
                self.obs_buffer[1] = obs;
            }
            if done_ == 1 {
                break;
            }
        }

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
}

impl Env for BorderAtariEnv {
    type Config = BorderAtariEnvConfig;
    type Obs = BorderAtariObs;
    type Act = BorderAtariAct;
    type Info = NullInfo;

    fn build(config: &Self::Config, seed: i64) -> Result<Self>
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
        })
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

    fn reset(&mut self, is_done: Option<&Vec<i8>>) -> Result<Self::Obs> {
        // TODO: noop random steps
        unimplemented!();
    }

    fn step(&mut self, act: &Self::Act) -> (border_core::Step<Self>, border_core::record::Record)
    where
        Self: Sized,
    {
        let (obs, reward, is_done) = self.skip_and_max(act);
        let obs = Self::warp_obs(obs);
        let obs = Self::scale_obs(obs);
        let reward = self.clip_reward(reward); // in training
        let obs = self.stack_frame(obs);
        let step = Step::new(obs, act.clone(), reward, is_done, NullInfo, Self::Obs::dummy(1));
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
