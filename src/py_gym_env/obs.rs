use std::fmt::Debug;
use std::marker::PhantomData;
use log::trace;
use num_traits::cast::AsPrimitive;
use pyo3::PyObject;
use ndarray::{ArrayD, Axis, IxDyn, stack};
use numpy::Element;
use crate::core::Obs;
use crate::agents::tch::Shape;
use crate::py_gym_env::PyGymEnvObsFilter;
use crate::agents::tch::py_gym_env::util::pyobj_to_arrayd;

fn any(is_done: &[f32]) -> bool {
    is_done.iter().fold(0, |x, v| x + *v as i32) > 0
}

pub struct PyGymEnvObsRawFilter<S, T> {
    phantom: PhantomData<(S, T)>
}

impl <S, T>PyGymEnvObsRawFilter<S, T> where
    S: Shape,
    T: Element,
{
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<S, T> PyGymEnvObsFilter<PyGymEnvObs<S, T>> for PyGymEnvObsRawFilter<S, T> where
    S: Shape,
    T: Element + Debug + num_traits::identities::Zero + AsPrimitive<f32>,
{
    fn filt(&mut self, obs: PyObject) -> PyGymEnvObs<S, T> {
        PyGymEnvObs {
            obs: pyobj_to_arrayd::<S, T>(obs),
            phantom: PhantomData,
        }
    }

    /// Stack filtered observation objects in the given vector.
    fn stack(filtered: Vec<PyGymEnvObs<S, T>>) -> PyGymEnvObs<S, T> {
        let arrays: Vec<_> = filtered.iter().map(|o| {
            debug_assert_eq!(&o.obs.shape()[1..], S::shape());
            o.obs.clone().remove_axis(Axis(0))
        }).collect();
        let arrays_view: Vec<_> = arrays.iter().map(|a| a.view()).collect();
        PyGymEnvObs::<S, T> {
            obs: stack(Axis(0), arrays_view.as_slice()).unwrap(),
            phantom: PhantomData
        }
    }
}

/// Represents observation.
#[derive(Clone, Debug)]
pub struct PyGymEnvObs<S, T> where
    S: Shape,
    T: Element + Debug,
{
    pub(crate) obs: ArrayD<f32>,
    phantom: PhantomData<(S, T)>
}

impl<S, T> PyGymEnvObs<S, T> where
    S: Shape,
    T: Element + Debug,
{
    pub fn new(obs: ArrayD<f32>) -> Self {
        Self {
            obs,
            phantom: PhantomData
        }
    }
}

impl<S, T> Obs for PyGymEnvObs<S, T> where
    S: Shape,
    T: Element + Debug + num_traits::identities::Zero,
{
    fn zero(n_procs: usize) -> Self {
        let shape = &mut S::shape().to_vec();
        shape.insert(0, n_procs as _);
        trace!("Shape of TchPyGymEnvObs: {:?}", shape);
        Self {
            obs: ArrayD::zeros(IxDyn(&shape[..])),
            phantom: PhantomData
        }
    }

    fn merge(mut self, obs_reset: Self, is_done: &[f32]) -> Self {
        if any(is_done) {
            for (i, is_done_i) in is_done.iter().enumerate() {
                if *is_done_i != 0.0 as f32 {
                    self.obs.index_axis_mut(Axis(0), i)
                        .assign(&obs_reset.obs.index_axis(Axis(0), i));
                }
            }
        };
        self
    }
}
