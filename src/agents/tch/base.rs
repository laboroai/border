use std::fmt::Debug;

pub trait Shape: Clone + Debug {
    fn shape() -> &'static [usize];
}
