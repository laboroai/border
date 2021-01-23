use std::fmt::Debug;
use std::cell::RefCell;
use num_traits::cast::AsPrimitive;
use ndarray::{s, ArrayD, Axis};
use pyo3::PyObject;
use numpy::PyArrayDyn;
use crate::agents::tch::Shape;
use crate::py_gym_env::ObsFilter;
use crate::agents::tch::py_gym_env::TchPyGymEnvObs;

// Returns ArrayD<u8>, not ArrayD<f32>
fn pyobj_to_arrayd<S: Shape>(obs: PyObject) -> ArrayD<u8> {
    pyo3::Python::with_gil(|py| {
        let obs: &PyArrayDyn<u8> = obs.extract(py).unwrap();
        let obs = obs.to_owned_array();
        let obs = obs.mapv(|elem| elem.as_());
        let obs = {
            if obs.shape().len() == S::shape().len() + 1 {
                // In this case obs has a dimension for n_procs
                obs
            }
            else if obs.shape().len() == S::shape().len() {
                // add dimension for n_procs
                obs.insert_axis(Axis(0))
            }
            else {
                panic!();
            }
        };
        obs
    })
}

/// Apply a filter to image observations in Pong.
///
/// Code is adapted from [TensorFlow reinforcement learning Pong agent](https://github.com/mrahtz/tensorflow-rl-pong/blob/master/pong.py).
fn filt_pong(img: ArrayD<u8>) -> ArrayD<f32> {
    img.slice(s![35..195, .., ..]).slice(s![..;2, ..;2, ..]).index_axis(Axis(2), 0)
    .mapv(|x| { match x {
        144 => 0f32,
        109 => 0f32,
        0 => 0f32,
        _ => 1f32
    }})
    .into_dyn()
}

#[derive(Clone, Debug)]
pub struct TchPyGymPongObsShape;

/// Shape of image observation of Pong.
///
/// It is after preprovessing by TchPyGymEnvObsPongFilter.
impl Shape for TchPyGymPongObsShape {
    fn shape() -> &'static [usize] {
        &[80, 80]
    }
}

type PongObsShape = TchPyGymPongObsShape;

pub struct TchPyGymEnvObsPongFilter {
    obs_prev: RefCell<ArrayD<f32>>,
}

impl TchPyGymEnvObsPongFilter {
    pub fn new() -> Self {
        Self {
            obs_prev: RefCell::new(ArrayD::<f32>::zeros(PongObsShape::shape())),
        }
    }
}

type PongObs = TchPyGymEnvObs<PongObsShape, u8>;

impl ObsFilter<PongObs> for TchPyGymEnvObsPongFilter {
    fn filt(&self, obs: PyObject) -> PongObs {
        let obs = pyobj_to_arrayd::<TchPyGymPongObsShape>(obs);
        let obs = filt_pong(obs);
        let obs_prev = self.obs_prev.replace(obs);
        let obs = self.obs_prev.borrow().clone() - obs_prev;
        PongObs::new(obs)
    }
}
