use border_core::Obs;

#[derive(Debug, Clone)]
pub struct BorderAtariObs {
    // 4 * 84 * 84
    frames: Vec<u8>,
}

impl From<Vec<u8>> for BorderAtariObs {
    fn from(frames: Vec<u8>) -> Self {
        Self { frames }
    }
}

impl Obs for BorderAtariObs {
    fn dummy(_n: usize) -> Self {
        Self {
            frames: vec![0; 4 * 84 * 84],
        }
    }

    fn merge(self, _obs_reset: Self, _is_done: &[i8]) -> Self {
        unimplemented!();
    }

    fn len(&self) -> usize {
        1
    }
}
