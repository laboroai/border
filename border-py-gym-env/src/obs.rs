//! Observation for [`super::PyGymEnv`] and [`super::PyVecGymEnv`].
use crate::base::{PyGymEnvObsFilter, Shape};
use border_core::{
    record::{Record, RecordValue},
    Obs,
};
use log::trace;
use ndarray::{stack, ArrayD, Axis, IxDyn};
use num_traits::cast::AsPrimitive;
use numpy::{Element, PyArrayDyn};
use pyo3::{types::PyList, Py, PyObject};
use std::default::Default;
use std::fmt::Debug;
use std::marker::PhantomData;

fn any(is_done: &[i8]) -> bool {
    is_done.iter().fold(0, |x, v| x + *v as i32) > 0
}

/// Convert PyObject to ArrayD.
///
/// If the shape of the PyArray has the number of axes equal to the shape of
/// observation, i.e., `S.shape().len()`, it is considered an observation from a
/// non-vectorized environment, an axis will be appended before the leading dimension.
/// in order for the array to meet the shape of the array in [`PyGymEnvObs`].
pub fn pyobj_to_arrayd<S, T1, T2>(obs: PyObject) -> ArrayD<T2>
where
    S: Shape,
    T1: Element + AsPrimitive<T2>,
    T2: 'static + Copy,
{
    pyo3::Python::with_gil(|py| {
        let obs: &PyArrayDyn<T1> = obs.extract(py).unwrap();
        let obs = obs.to_owned_array();
        // let obs = obs.mapv(|elem| elem as f32);
        let obs = obs.mapv(|elem| elem.as_());
        let obs = {
            if obs.shape().len() == S::shape().len() + 1 {
                // In this case obs has a dimension for n_procs
                obs
            } else if obs.shape().len() == S::shape().len() {
                // add dimension for n_procs
                obs.insert_axis(Axis(0))
            } else {
                panic!();
            }
        };
        obs
    })
}

/// Observation represented by an [ndarray::ArrayD].
///
/// `S` is the shape of an observation, except for batch and process dimensions.
/// `T` is the dtype of ndarray in the Python gym environment.
/// For some reason, the dtype of observations in Python gym environments seems to
/// vary, f32 or f64. To get observations in Rust side, the dtype is specified as a
/// type parameter, instead of checking the dtype of Python array at runtime.
#[derive(Clone, Debug)]
pub struct PyGymEnvObs<S, T1, T2>
where
    S: Shape,
    T1: Element + Debug,
    T2: 'static + Copy,
{
    pub obs: ArrayD<T2>,
    pub(crate) phantom: PhantomData<(S, T1)>,
}

// // TODO: consider remove this item.
// impl<S, T1, T2> PyGymEnvObs<S, T1, T2> where
//     S: Shape,
//     T1: Element + Debug,
// {
//     // pub fn new(obs: ArrayD<f32>) -> Self {
//     //     Self {
//     //         obs,
//     //         phantom: PhantomData
//     //     }
//     // }
// }

// impl<S, T1, T2> Obs for PyGymEnvObs<S, T1, T2> where
//     S: Shape,
//     T1: Element + Debug + num_traits::identities::Zero,
// {
impl<S, T1, T2> Obs for PyGymEnvObs<S, T1, T2>
where
    S: Shape,
    T1: Debug + Element,
    T2: 'static + Copy + Debug + num_traits::Zero,
{
    fn dummy(n_procs: usize) -> Self {
        let shape = &mut S::shape().to_vec();
        shape.insert(0, n_procs as _);
        trace!("Shape of TchPyGymEnvObs: {:?}", shape);
        Self {
            obs: ArrayD::zeros(IxDyn(&shape[..])),
            phantom: PhantomData,
        }
    }

    fn merge(mut self, obs_reset: Self, is_done: &[i8]) -> Self {
        if any(is_done) {
            for (i, is_done_i) in is_done.iter().enumerate() {
                if *is_done_i != 0 {
                    self.obs
                        .index_axis_mut(Axis(0), i)
                        .assign(&obs_reset.obs.index_axis(Axis(0), i));
                }
            }
        };
        self
    }

    // TODO: consider to remove this method
    fn n_procs(&self) -> usize {
        self.obs.shape()[0]
    }

    fn batch_size(&self) -> usize {
        self.obs.shape()[0]
    }
}

/// An observation filter without any postprocessing.
///
/// The filter works with [super::PyGymEnv] or [super::PyVecGymEnv].
pub struct PyGymEnvObsRawFilter<S, T1, T2> {
    /// If the environment is vectorized.
    pub vectorized: bool,
    /// Marker.
    pub phantom: PhantomData<(S, T1, T2)>,
}

impl<S, T1, T2> PyGymEnvObsRawFilter<S, T1, T2> {
    /// Constructs a raw filter for vectorized environments.
    pub fn vectorized() -> Self {
        Self {
            vectorized: true,
            phantom: PhantomData,
        }
    }
}

impl<S, T1, T2> Default for PyGymEnvObsRawFilter<S, T1, T2>
where
    S: Shape,
    T1: Element,
{
    fn default() -> Self {
        Self {
            vectorized: false,
            phantom: PhantomData,
        }
    }
}

impl<S, T1, T2> PyGymEnvObsFilter<PyGymEnvObs<S, T1, T2>> for PyGymEnvObsRawFilter<S, T1, T2>
where
    S: Shape,
    T1: Element + Debug + num_traits::identities::Zero + AsPrimitive<T2>,
    T2: 'static + Copy + Debug + num_traits::Zero + AsPrimitive<f32>,
{
    /// Convert `PyObject` to [ndarray::ArrayD].
    ///
    /// No filter is applied after conversion.
    /// The shape of the observation is `S` in [crate::env::py_gym_env::PyGymEnv].
    ///
    /// For [crate::env::py_gym_env::PyVecGymEnv], which is a vectorized environments,
    /// the shape becomes `[n_procs, S]`, where `n_procs` is the number of processes
    /// of the vectorized environment.
    ///
    /// [Record] in the returned value has `obs`, which is a flattened array of
    /// observation, for either of single and vectorized environments.
    fn filt(&mut self, obs: PyObject) -> (PyGymEnvObs<S, T1, T2>, Record) {
        if self.vectorized {
            let obs = pyo3::Python::with_gil(|py| {
                debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "list");
                let obs: Py<PyList> = obs.extract(py).unwrap();

                // Iterate over the list of observations of the environments in the
                // vectorized environment.
                let filtered = obs
                    .as_ref(py)
                    .iter()
                    .map(|o| {
                        // `NoneType` means the element will be ignored in the following processes.
                        // This can appears in partial reset of the vectorized environment.
                        if o.get_type().name().unwrap() == "NoneType" {
                            ArrayD::zeros(IxDyn(S::shape()))
                        }
                        // Processes the partial observation in the vectorized environment.
                        else {
                            debug_assert_eq!(o.get_type().name().unwrap(), "ndarray");
                            let obs: &PyArrayDyn<T1> = o.extract().unwrap();
                            let obs = obs.to_owned_array();
                            let obs = obs.mapv(|elem| elem.as_());
                            debug_assert_eq!(obs.shape(), S::shape());
                            obs
                        }
                    })
                    .collect::<Vec<_>>();
                let arrays_view: Vec<_> = filtered.iter().map(|a| a.view()).collect();
                PyGymEnvObs::<S, T1, T2> {
                    obs: stack(Axis(0), arrays_view.as_slice()).unwrap(),
                    phantom: PhantomData,
                }
            });
            let record = {
                let vec = obs.obs.iter().map(|x| x.as_()).collect();
                Record::from_slice(&[("obs", RecordValue::Array1(vec))])
            };
            // let record = Record::from_slice(
            //     &[("obs", RecordValue::Array1(
            //         Vec::<_>::from_iter(obs.obs.iter().map(|x| x.as_()).cloned()))
            //     )]);
            (obs, record)
        } else {
            let obs = pyo3::Python::with_gil(|py| {
                if obs.as_ref(py).get_type().name().unwrap() == "NoneType" {
                    // TODO: consider panic!() if the environment returns None
                    PyGymEnvObs::<S, T1, T2>::dummy(1)
                } else {
                    PyGymEnvObs {
                        obs: pyobj_to_arrayd::<S, T1, T2>(obs),
                        phantom: PhantomData,
                    }
                }
            });
            // let array1: Vec<T2> = obs.obs.iter().cloned().collect();
            // let record = Record::from_slice(&[("obs", RecordValue::Array1(array1))]);
            let record = {
                // let vec = Vec::<f32>::from_iter(obs.obs.iter().map(|x| x.as_()));
                let vec = obs.obs.iter().map(|x| x.as_()).collect();
                Record::from_slice(&[("obs", RecordValue::Array1(vec))])
            };
            (obs, record)
        }
    }
}

/// Defines a newtype of [PyGymEnvObs] and [PyGymEnvObsRawFilter].
///
/// The example below defines `ObsShape` implementing [Shape](border_py_gym_env::Shape),
/// `Obs` as a newtype of [PyGymEnvObs], and `ObsFilter` as a newtype of [PyGymEnvObsRawFilter].
///
/// ```no_run
/// use border_py_gym_env::{newtype_obs, shape};
///
/// shape!(ObsShape, [1, 2, 3]);
/// newtype_obs!(Obs, ObsFilter, ObsShape, f32, f32);
///
/// fn func() {
///    // TODO: run test
///    let obs = Obs::new();
///    let obs_filter = ObsFilter::new();
/// }
/// ```
#[macro_export]
macro_rules! newtype_obs {
    ($struct_:ident, $shape_:ty, $t1_:ty, $t2_:ty) => {
        #[derive(Clone, Debug)]
        struct $struct_(border_py_gym_env::PyGymEnvObs<$shape_, $t1_, $t2_>);

        impl border_core::Obs for $struct_{
            fn dummy(n_procs: usize) -> Self {
                $struct_(border_py_gym_env::PyGymEnvObs::dummy(n_procs))
            }

            fn merge(mut self, obs_reset: Self, is_done: &[i8]) -> Self {
                $struct_(self.0.merge(obs_reset.0, is_done))
            }

            fn n_procs(&self) -> usize {
                unimplemented!();
            }

            fn batch_size(&self) -> usize {
                self.0.batch_size()
            }
        }
    };

    ($struct_:ident, $struct2_:ident, $shape_:ty, $t1_:ty, $t2_:ty) => {
        newtype_obs!($struct_, $shape_, $t1_, $t2_);

        struct $struct2_(border_py_gym_env::PyGymEnvObsRawFilter<$shape_, $t1_, $t2_>);

        impl<$shape_, $t1_, $t2_> border_py_gym_env::PyGymEnvObsFilter<$shape_, $t1_, $t2_> for $struct2_ {
            fn filt(&mut self, obs: pyo3::PyObject) -> ($struct_, border_core::record::Record) {
                let (obs, record) = self.0.filt(obs);
                ($struct_(obs), record)
            }
        }
    };
}
