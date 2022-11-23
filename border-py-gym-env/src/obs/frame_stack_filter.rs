//! An observation filter with stacking observations (frames).
use super::PyGymEnvObs;
use crate::PyGymEnvObsFilter;
use border_core::Shape;
use border_core::{
    record::{Record, RecordValue},
    Obs,
};
use ndarray::{ArrayD, Axis, SliceInfoElem}; //, SliceOrIndex};
                                            // use ndarray::{stack, ArrayD, Axis, IxDyn, SliceInfo, SliceInfoElem};
use num_traits::cast::AsPrimitive;
use numpy::{Element, PyArrayDyn};
use pyo3::{PyAny, PyObject};
// use pyo3::{types::PyList, Py, PyAny, PyObject};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};
// use std::{convert::TryFrom, fmt::Debug, marker::PhantomData};

#[derive(Debug, Serialize, Deserialize)]
/// Configuration of [FrameStackFilter].
#[derive(Clone)]
pub struct FrameStackFilterConfig {
    n_procs: i64,
    n_stack: i64,
    vectorized: bool,
}

impl Default for FrameStackFilterConfig {
    fn default() -> Self {
        Self {
            n_procs: 1,
            n_stack: 4,
            vectorized: false,
        }
    }
}

/// An observation filter with stacking sequence of original observations.
///
/// The first element of the shape `S` denotes the number of stacks (`n_stack`) and the following elements
/// denote the shape of the partial observation, which is the observation of each environment
/// in the vectorized environment.
#[derive(Debug)]
pub struct FrameStackFilter<S, T1, T2, U>
where
    S: Shape,
    T1: Element + Debug + num_traits::identities::Zero + AsPrimitive<T2>,
    T2: 'static + Copy + num_traits::Zero,
    U: Obs + From<PyGymEnvObs<T1, T2>>,
{
    // Each element in the vector corresponds to a process.
    buffers: Vec<Option<ArrayD<T2>>>,

    #[allow(dead_code)]
    n_procs: i64,

    n_stack: i64,

    shape: Option<Vec<usize>>,

    // Verctorized environment is not supported
    vectorized: bool,

    phantom: PhantomData<(S, T1, U)>,
}

impl<S, T1, T2, U> FrameStackFilter<S, T1, T2, U>
where
    S: Shape,
    T1: Element + Debug + num_traits::identities::Zero + AsPrimitive<T2>,
    T2: 'static + Copy + num_traits::Zero,
    U: Obs + From<PyGymEnvObs<T1, T2>>,
{
    /// Returns the default configuration.
    pub fn default_config() -> FrameStackFilterConfig {
        FrameStackFilterConfig::default()
    }

    /// Create slice for a dynamic array: equivalent to arr[j:(j+1), ::] in numpy.
    ///
    /// See https://github.com/rust-ndarray/ndarray/issues/501
    fn s(shape: &Option<Vec<usize>>, j: usize) -> Vec<SliceInfoElem> {
        // The first index of self.shape corresponds to stacking dimension,
        // specific index.
        let mut slicer = vec![SliceInfoElem::Index(j as isize)];

        // For remaining dimensions, all elements will be taken.
        let n = shape.as_ref().unwrap().len() - 1;
        let (start, end, step) = (0, None, 1);

        slicer.extend(vec![SliceInfoElem::Slice { start, end, step }; n]);
        slicer
    }

    /// Update the buffer of the stacked observations.
    ///
    /// * `i` - Index of process.
    fn update_buffer(&mut self, i: i64, obs: &ArrayD<T2>) {
        let arr = if let Some(arr) = &mut self.buffers[i as usize] {
            arr
        } else {
            let mut shape = obs.shape().to_vec();
            self.shape = Some(shape.clone());
            shape.insert(0, self.n_stack as _);
            self.buffers[i as usize] = Some(ArrayD::zeros(shape));
            self.buffers[i as usize].as_mut().unwrap()
        };

        // Shift stacks frame(j) <- frame(j - 1) for j=1,..,(n_stack - 1)
        for j in (1..self.n_stack as usize).rev() {
            let dst_slice = Self::s(&self.shape, j);
            let src_slice = Self::s(&self.shape, j - 1);
            let (mut dst, src) = arr.multi_slice_mut((dst_slice.as_slice(), src_slice.as_slice()));
            dst.assign(&src);
        }
        arr.slice_mut(Self::s(&self.shape, 0).as_slice())
            .assign(obs)
    }

    /// Fill the buffer, invoked when resetting
    fn fill_buffer(&mut self, i: i64, obs: &ArrayD<T2>) {
        if let Some(arr) = &mut self.buffers[i as usize] {
            for j in (0..self.n_stack as usize).rev() {
                let mut dst = arr.slice_mut(Self::s(&self.shape, j).as_slice());
                dst.assign(&obs);
            }
        } else {
            unimplemented!("fill_buffer() was called before receiving the first sample.");
        }
    }

    /// Get ndarray from pyobj
    fn get_ndarray(o: &PyAny) -> ArrayD<T2> {
        debug_assert_eq!(o.get_type().name().unwrap(), "ndarray");
        let o: &PyArrayDyn<T1> = o.extract().unwrap();
        let o = o.to_owned_array();
        let o = o.mapv(|elem| elem.as_());
        o
    }
}

impl<S, T1, T2, U> PyGymEnvObsFilter<U> for FrameStackFilter<S, T1, T2, U>
where
    S: Shape,
    T1: Element + Debug + num_traits::identities::Zero + AsPrimitive<T2>,
    T2: 'static + Copy + num_traits::Zero + Into<f32>,
    U: Obs + From<PyGymEnvObs<T1, T2>>,
{
    type Config = FrameStackFilterConfig;

    fn build(config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        Ok(FrameStackFilter {
            buffers: vec![None; config.n_procs as usize],
            n_procs: config.n_procs,
            n_stack: config.n_stack,
            shape: None,
            vectorized: config.vectorized,
            phantom: PhantomData,
        })
    }

    fn filt(&mut self, obs: PyObject) -> (U, Record) {
        if self.vectorized {
            unimplemented!();
            // // Processes the input observation to update `self.buffer`
            // pyo3::Python::with_gil(|py| {
            //     debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "list");

            //     let obs: Py<PyList> = obs.extract(py).unwrap();

            //     for (i, o) in (0..self.n_procs).zip(obs.as_ref(py).iter()) {
            //         let o = Self::get_ndarray(o);
            //         self.update_buffer(i, &o);
            //     }
            // });

            // // Returned values
            // let array_views: Vec<_> = self.buffer.iter().map(|a| a.view()).collect();
            // let obs = PyGymEnvObs::from(stack(Axis(0), array_views.as_slice()).unwrap());
            // let obs = U::from(obs);

            // // TODO: add contents in the record
            // let record = Record::empty();

            // (obs, record)
        } else {
            // Update the buffer with obs
            pyo3::Python::with_gil(|py| {
                debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "ndarray");
                let o = Self::get_ndarray(obs.as_ref(py));
                self.update_buffer(0, &o);
            });

            // Returns stacked observation in the buffer
            // img.shape() = [1, 4, 1, 84, 84]
            // [batch_size, n_stack, color_ch, width, height]
            let img = self.buffers[0].clone().unwrap().insert_axis(Axis(0));
            let data = img.iter().map(|&e| e.into()).collect::<Vec<_>>();
            let shape = [img.shape()[3] * self.n_stack as usize, img.shape()[4]];

            let obs = PyGymEnvObs::from(img);
            let obs = U::from(obs);

            // TODO: add contents in the record
            let mut record = Record::empty();
            record.insert("frame_stack_filter_out", RecordValue::Array2(data, shape));

            (obs, record)
        }
    }

    fn reset(&mut self, obs: PyObject) -> U {
        if self.vectorized {
            unimplemented!();
            // pyo3::Python::with_gil(|py| {
            //     debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "list");

            //     let obs: Py<PyList> = obs.extract(py).unwrap();

            //     for (i, o) in (0..self.n_procs).zip(obs.as_ref(py).iter()) {
            //         if o.get_type().name().unwrap() != "NoneType" {
            //             let o = Self::get_ndarray(o);
            //             self.fill_buffer(i, &o);
            //         }
            //     }
            // });

            // // Returned values
            // let array_views: Vec<_> = self.buffer.iter().map(|a| a.view()).collect();
            // O::from(stack(Axis(0), array_views.as_slice()).unwrap())
        } else {
            // Update the buffer if obs is not None, otherwise do nothing
            pyo3::Python::with_gil(|py| {
                if obs.as_ref(py).get_type().name().unwrap() != "NoneType" {
                    debug_assert_eq!(obs.as_ref(py).get_type().name().unwrap(), "ndarray");
                    let o = Self::get_ndarray(obs.as_ref(py));
                    self.fill_buffer(0, &o);
                }
            });

            // Returns stacked observation in the buffer
            let frames = self.buffers[0].clone().unwrap().insert_axis(Axis(0));
            U::from(PyGymEnvObs::from(frames))
        }
    }
}
