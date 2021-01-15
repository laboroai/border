use std::fmt::Debug;

pub trait Shape: Clone + Debug {
    fn shape() -> &'static [usize];

    /// Return true if you would like to squeeze the first dimension of the array
    /// before conversion into an numpy array in Python. The first dimension may
    /// correspond to process indices for vectorized environments. However, this
    /// dimension is not compatible with PyGymEnv (non-vectorized environment).
    /// This method is used in
    /// [here][crate::agents::tch::py_gym_env::act_c::TchPyGymEnvContinuousAct#impl-Into<Py<PyAny>>].
    fn squeeze_first_dim() -> bool {
        false
    }
}
