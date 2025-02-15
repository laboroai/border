use ndarray::ArrayD;

#[derive(Clone, Debug)]
/// Action.
pub enum NdarrayAct {
    Continuous(ArrayD<f32>),
    Discrete(ArrayD<i64>),
}

impl border_core::Act for NdarrayAct {}

impl NdarrayAct {
    /// Creates continuous action from a vector of [`ArrayD`]`<f32>`.
    pub fn new_cont(a: ArrayD<f32>) -> Self {
        Self::Continuous(a)
    }
}
