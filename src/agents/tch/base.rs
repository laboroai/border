pub trait Shape: Clone {
    fn shape() -> &'static [usize];
}
