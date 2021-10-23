use border_core::Obs;

#[derive(Debug, Clone)]
pub struct BorderAtariObs {
}

impl Obs for BorderAtariObs {
    fn dummy(_n: usize) -> Self {
        unimplemented!();
    }

    fn merge(self, _obs_reset: Self, _is_done: &[i8]) -> Self {
        unimplemented!();
    }

    fn len(&self) -> usize {
        1
    }
}
