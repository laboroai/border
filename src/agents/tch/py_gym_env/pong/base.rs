use std::fmt::Debug;
use std::cell::RefCell;
use std::{path::Path, error::Error};
use log::{info, trace};
use num_traits::cast::AsPrimitive;
use ndarray::{s, ArrayD, Axis};
use pyo3::PyObject;
use numpy::PyArrayDyn;
use tch::{Tensor, nn, nn::Module, Device, nn::OptimizerConfig};
use crate::py_gym_env::ObsFilter;
use crate::agents::tch::Shape;
use crate::agents::tch::model::{ModelBase, Model1};
use crate::agents::tch::py_gym_env::TchPyGymEnvObs;

/// Returns ArrayD<u8>, not ArrayD<f32>
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

/// Action value function.
///
/// Code is adapted from [TensorFlow reinforcement learning Pong agent](https://github.com/mrahtz/tensorflow-rl-pong/blob/master/pong.py).
#[derive(Debug)]
pub struct PongNet {
    var_store: nn::VarStore,
    network: nn::Sequential,
    device: Device,
    opt: nn::Optimizer<nn::Adam>,
    learning_rate: f64
}

impl Clone for PongNet {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.learning_rate);
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl PongNet {
    pub fn new(learning_rate: f64) -> Self {
        let in_dim = (80 * 80) as _;
        let out_dim = 6 as _;
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let p = &vs.root();

        let network = nn::seq()
            .add_fn(move |xs| xs.reshape(&[-1, in_dim]))
            .add(nn::linear(p / "cl1", in_dim, 200, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "cl2", 200, out_dim, Default::default()));
        let opt = nn::Adam::default().build(&vs, learning_rate).unwrap();

        Self {
            network,
            device: p.device(),
            var_store: vs,
            opt,
            learning_rate,
        }
    }
}

impl ModelBase for PongNet {
    fn backward_step(&mut self, loss: &Tensor) {
        self.opt.backward_step(loss);
    }

    fn get_var_store(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.save(&path)?;
        info!("Save pongnet to {:?}", path.as_ref());
        let vs = self.var_store.variables();
        for (name, _) in vs.iter() {
            trace!("Save variable {}", name);
        };
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.load(&path)?;
        info!("Load pongnet from {:?}", path.as_ref());
        Ok(())
    }
}

impl Model1 for PongNet {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, xs: &Tensor) -> Tensor {
        self.network.forward(xs)
    }
}