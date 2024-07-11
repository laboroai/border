//! Atari environment for reinforcement learning.
pub mod ale;
use std::path::Path;

pub use ale::Ale;
pub use ale::AleAction as AtariAction;
pub use ale::AleConfig as EmulatorConfig;
use gym_core::{ActionSpace, CategoricalActionSpace, GymEnv};

use anyhow::{Context, Result};
use ndarray::{Array1, ArrayD, ArrayView3, Ix0, Ix1, Ix3};
use num_traits::cast::FromPrimitive;

pub struct AtariEnv {
    ale: Ale,
}

impl AtariEnv {
    /// about frame-skipping and action-repeat,
    /// see <https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/>
    pub fn new<P: AsRef<Path>>(rom_path: P, emulator_config: EmulatorConfig) -> Self {
        Self {
            ale: Ale::new(rom_path.as_ref(), emulator_config),
        }
    }
    pub fn width(&self) -> usize {
        self.ale.width() as usize
    }
    pub fn height(&self) -> usize {
        self.ale.height() as usize
    }
    pub fn available_actions(&self) -> Vec<AtariAction> {
        self.ale.available_actions()
    }
    pub fn minimal_actions(&self) -> Vec<AtariAction> {
        self.ale.minimal_actions()
    }
    pub fn available_difficulty_settings(&self) -> Vec<i32> {
        self.ale.available_difficulty_settings()
    }
    pub fn lives(&self) -> usize {
        self.ale.lives() as usize
    }
    pub fn is_game_over(&self) -> bool {
        self.ale.is_game_over()
    }
    pub fn reset(&mut self) {
        self.ale.reset()
    }
    pub fn step(&mut self, action: AtariAction) -> i32 {
        self.ale.take_action(action)
    }
    pub fn rgb32_size(&self) -> usize {
        self.ale.rgb32_size()
    }
    pub fn rgb24_size(&self) -> usize {
        self.ale.rgb24_size()
    }
    pub fn ram_size(&self) -> usize {
        self.ale.ram_size()
    }
    pub fn render_rgb32(&self, buf: &mut [u8]) {
        self.ale.rgb32(buf);
    }
    pub fn render_rgb24(&self, buf: &mut [u8]) {
        self.ale.rgb24(buf);
    }
    pub fn render_ram(&self, buf: &mut [u8]) {
        self.ale.ram(buf);
    }
    pub fn into_ram_env(self) -> AtariRamEnv {
        AtariRamEnv::new(self)
    }
    pub fn into_rgb_env(self) -> AtariRgbEnv {
        AtariRgbEnv::new(self)
    }
    pub fn seed(&self, seed: i32) {
        self.ale.seed(seed);
    }
}

pub struct AtariRamEnv {
    buf1: Array1<u8>,
    inner: AtariEnv,
    available_actions: Vec<AtariAction>,
}

pub struct AtariRgbEnv {
    buf1: Array1<u8>,
    inner: AtariEnv,
    available_actions: Vec<AtariAction>,
}

impl AtariRamEnv {
    pub fn new(env: AtariEnv) -> Self {
        Self {
            buf1: Array1::zeros(env.ram_size()),
            available_actions: env.minimal_actions(),
            inner: env,
        }
    }
}

impl GymEnv<i32> for AtariRamEnv {
    fn state_size(&self) -> Vec<usize> {
        vec![self.inner.ram_size()]
    }
    fn action_space(&self) -> ActionSpace<i32> {
        Box::new(CategoricalActionSpace::new(self.available_actions.len()))
    }
    fn state(&self, out: ndarray::ArrayViewMut<f32, ndarray::IxDyn>) -> Result<()> {
        let mut out = out.into_dimensionality::<Ix1>()?;
        ndarray::parallel::par_azip!((a in &mut out, &b in &self.buf1) {*a = b as f32 / 255.0;});
        Ok(())
    }
    fn step(&mut self, action: ArrayD<i32>) -> Result<i32> {
        let action = AtariAction::from_i32(action.into_dimensionality::<Ix0>()?.into_scalar())
            .context("action out of range")?;
        let reward = self.inner.step(action);
        self.inner.render_ram(self.buf1.as_slice_mut().unwrap());
        Ok(reward)
    }
    fn is_over(&self) -> bool {
        self.inner.is_game_over()
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

impl AtariRgbEnv {
    pub fn new(env: AtariEnv) -> Self {
        Self {
            buf1: Array1::zeros(env.rgb24_size()),
            available_actions: env.minimal_actions(),
            inner: env,
        }
    }
}

impl GymEnv<i32> for AtariRgbEnv {
    fn state_size(&self) -> Vec<usize> {
        vec![self.inner.height(), self.inner.width(), 3]
    }
    fn action_space(&self) -> ActionSpace<i32> {
        Box::new(CategoricalActionSpace::new(self.available_actions.len()))
    }
    fn state(&self, out: ndarray::ArrayViewMut<f32, ndarray::IxDyn>) -> Result<()> {
        let mut out = out.into_dimensionality::<Ix3>()?;
        let from: ArrayView3<_> = self
            .buf1
            .view()
            .into_shape(self.state_size())?
            .into_dimensionality()?;
        ndarray::parallel::par_azip!((a in &mut out, &b in &from) {*a = b as f32 / 255.0;});
        // ndarray::parallel::par_azip!((a in &mut out, &b in &self.buf1) {*a = b as f32 / 255.0;});
        Ok(())
    }
    // fn state(&self) -> ArrayView<f32, IxDyn>{ self.buf2.view().into_dyn() }
    fn step(&mut self, action: ArrayD<i32>) -> Result<i32> {
        let action = self.available_actions
            [(action.into_dimensionality::<Ix0>()?.into_scalar() - 1) as usize];
        let reward = self.inner.step(action);
        self.inner.render_rgb24(self.buf1.as_slice_mut().unwrap());
        Ok(reward)
    }
    fn is_over(&self) -> bool {
        self.inner.is_game_over()
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}
