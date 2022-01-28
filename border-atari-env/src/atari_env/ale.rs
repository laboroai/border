use c_str_macro::c_str;
use std::ffi::CString;
use std::path::Path;
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, num_derive::FromPrimitive)]
#[repr(i32)]
pub enum AleAction {
    Noop = 0,
    Fire = 1,
    Up = 2,
    Right = 3,
    Left = 4,
    Down = 5,
    UpRight = 6,
    UpLeft = 7,
    DownRight = 8,
    DownLeft = 9,
    UpFire = 10,
    RightFire = 11,
    LeftFire = 12,
    DownFire = 13,
    UpRightFire = 14,
    UpLeftFire = 15,
    DownRightFire = 16,
    DownLeftFire = 17,
    // admin actions
    // Reset = 40,
    // SaveState = 43,
    // LoadState = 44,
    // SystemReset = 45,
}

pub struct AleConfig {
    pub random_seed: i32, // if 0, set to time
    pub display_screen: bool,
    pub sound: bool,
    pub color_averaging: bool, // average the last 2 frames
    ///
    pub frame_skip: i32, // 1 is no skip
    pub repeat_action_probability: f32,
    pub record_screen_dir: Option<PathBuf>,
    ///
    pub difficulty_setting: i32,
}

impl Default for AleConfig {
    fn default() -> Self {
        Self {
            random_seed: 0,
            display_screen: false,
            sound: false,
            color_averaging: false, // true is recommended
            frame_skip: 1,
            repeat_action_probability: 0.25,
            record_screen_dir: None,
            difficulty_setting: 0,
        }
    }
}

pub struct Ale {
    inner: *mut atari_env_sys::ALEInterface,
}

// TODO: it seems to work, but needs to be verified
unsafe impl Send for Ale {}

impl Drop for Ale {
    fn drop(&mut self) {
        unsafe {
            atari_env_sys::ALE_del(self.inner);
        }
    }
}

impl Ale {
    pub fn new(rom_path: &Path, config: AleConfig) -> Self {
        let ale = unsafe { atari_env_sys::ALE_new() };
        unsafe {
            atari_env_sys::setInt(ale, c_str!("random_seed").as_ptr(), config.random_seed);
            atari_env_sys::setBool(
                ale,
                c_str!("display_screen").as_ptr(),
                config.display_screen,
            );
            atari_env_sys::setBool(ale, c_str!("sound").as_ptr(), config.sound);
            atari_env_sys::setBool(
                ale,
                c_str!("color_averaging").as_ptr(),
                config.color_averaging,
            );
            atari_env_sys::setInt(ale, c_str!("frame_skip").as_ptr(), config.frame_skip);
            atari_env_sys::setFloat(
                ale,
                c_str!("repeat_action_probability").as_ptr(),
                config.repeat_action_probability,
            );

            if let Some(path) = config.record_screen_dir {
                let path = CString::new(path.to_str().unwrap()).unwrap();
                atari_env_sys::setString(ale, c_str!("record_screen_dir").as_ptr(), path.as_ptr());
            }
            let rom_path = CString::new(rom_path.to_str().unwrap()).unwrap();
            atari_env_sys::loadROM(ale, rom_path.as_ptr());
        }
        unsafe {
            atari_env_sys::setDifficulty(ale, config.difficulty_setting);
            atari_env_sys::reset_game(ale);
        }

        Self { inner: ale }
    }
    pub fn available_actions(&self) -> Vec<AleAction> {
        let n = unsafe { atari_env_sys::getLegalActionSize(self.inner) } as usize;
        let mut buf = vec![AleAction::Noop; n];
        unsafe {
            atari_env_sys::getLegalActionSet(self.inner, buf.as_mut_ptr() as *mut i32);
        }
        buf
    }

    pub fn minimal_actions(&self) -> Vec<AleAction> {
        let n = unsafe { atari_env_sys::getMinimalActionSize(self.inner) } as usize;
        let mut buf = vec![AleAction::Noop; n];
        unsafe {
            atari_env_sys::getMinimalActionSet(self.inner, buf.as_mut_ptr() as *mut i32);
        }
        buf
    }

    pub fn is_game_over(&self) -> bool {
        unsafe { atari_env_sys::game_over(self.inner) }
    }

    /// frame number since rom loading (Ale::new)
    pub fn rom_frame_number(&self) -> i32 {
        unsafe { atari_env_sys::getFrameNumber(self.inner) }
    }

    /// frame number of the current episode
    pub fn episode_frame_number(&self) -> i32 {
        unsafe { atari_env_sys::getEpisodeFrameNumber(self.inner) }
    }

    pub fn reset(&mut self) {
        unsafe {
            atari_env_sys::reset_game(self.inner);
        }
    }

    /// returns reward
    pub fn take_action(&mut self, action: AleAction) -> i32 {
        let ret: ::std::os::raw::c_int = unsafe { atari_env_sys::act(self.inner, action as i32) };
        ret.into()
    }

    pub fn lives(&self) -> u32 {
        unsafe { atari_env_sys::lives(self.inner) as u32 }
    }

    pub fn available_difficulty_settings(&self) -> Vec<i32> {
        let n = unsafe { atari_env_sys::getAvailableDifficultiesSize(self.inner) } as usize;
        let mut buf = vec![0i32; n];
        unsafe {
            atari_env_sys::getAvailableDifficulties(self.inner, buf.as_mut_ptr() as *mut i32);
        }
        buf
    }

    pub fn width(&self) -> u32 {
        unsafe { atari_env_sys::getScreenWidth(self.inner) as u32 }
    }
    pub fn height(&self) -> u32 {
        unsafe { atari_env_sys::getScreenHeight(self.inner) as u32 }
    }

    pub fn rgb24_size(&self) -> usize {
        return (self.width() as usize) * (self.height() as usize) * 3;
    }
    /// bgr on little-endian, rgb on big-endian
    pub fn rgb24_native_endian(&self, buf: &mut [u8]) {
        unsafe {
            atari_env_sys::getScreenRGB(self.inner, buf.as_mut_ptr());
        }
    }
    /// always rgb in regardless of endianness
    pub fn rgb24(&self, buf: &mut [u8]) {
        unsafe {
            atari_env_sys::getScreenRGB2(self.inner, buf.as_mut_ptr());
        }
    }

    pub fn rgb32_size(&self) -> usize {
        return (self.width() as usize) * (self.height() as usize) * 4;
    }
    /// always rgb in regardless of endianness
    pub fn rgb32(&self, buf: &mut [u8]) {
        let n = buf.len() / 4;
        self.rgb24(&mut buf[n..]);
        for i in 0..n {
            buf[i * 4 + 0] = buf[n + (i * 3) + 0];
            buf[i * 4 + 1] = buf[n + (i * 3) + 1];
            buf[i * 4 + 2] = buf[n + (i * 3) + 2];
            buf[i * 4 + 3] = 0;
        }
    }

    pub fn ram_size(&self) -> usize {
        unsafe { atari_env_sys::getRAMSize(self.inner) as usize }
    }
    pub fn ram(&self, buf: &mut [u8]) {
        unsafe {
            atari_env_sys::getRAM(self.inner, buf.as_mut_ptr());
        }
    }

    pub fn save_png<P: AsRef<Path>>(&self, path: P) {
        use std::os::unix::ffi::OsStrExt;
        let path = path.as_ref();
        let path = CString::new(path.as_os_str().as_bytes()).unwrap();
        unsafe {
            atari_env_sys::saveScreenPNG(self.inner, path.as_ptr());
        }
    }
}
