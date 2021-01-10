fn main() {}
// use std::error::Error;
// use ndarray::{ArrayD, Array2};
// use pyo3::{PyObject, IntoPy};
// use numpy::{PyArrayDyn, PyArray2};
// use tch::Tensor;
// use lrr::core::{Obs, Act, Env, Trainer, Agent, util};
// use lrr::py_gym_env::{PyGymInfo, vec::PyVecGymEnv};
// use lrr::agents::tch::{DQN, QNetwork, ReplayBuffer, TchBuffer};

// #[derive(Clone, Debug)]
// pub struct CartPoleVecObs (pub Array2<f32>);

// impl Obs for CartPoleVecObs {}

// impl From<PyObject> for CartPoleVecObs {
//     fn from(obs: PyObject) -> Self {
//         pyo3::Python::with_gil(|py| {
//             let obs: &PyArray2<f64> = obs.extract(py).unwrap();
//             let obs = obs.to_owned_array();
//             let obs = obs.mapv(|elem| elem as f32);
//             Self {
//                 0: obs
//             }
//         })
//     }
// }

// impl Into<Tensor> for CartPoleVecObs {
//     fn into(self) -> Tensor {
//         let obs = self.0.view().to_slice().unwrap();
//         Tensor::of_slice(obs)
//     }
// }

// struct CartPoleVecObsBuffer {
//     obs: Tensor
// }

// impl TchBuffer for CartPoleVecObsBuffer {
//     type Item = CartPoleVecObs;
//     type SubBatch = Tensor;

//     fn new(capacity: usize) -> Self {
//         Self {
//             obs: Tensor::zeros(&[capacity as _, 4], tch::kind::FLOAT_CPU),
//         }
//     }

//     fn push(&mut self, index: i64, item: &CartPoleVecObs) {
//         let obs = item.0.view().to_slice().unwrap();
//         let obs = Tensor::of_slice(obs);
//         self.obs.get(index).copy_(&obs);
//     }

//     fn batch(&self, batch_indexes: &Tensor) -> Tensor {
//         self.obs.index_select(0, &batch_indexes)
//     }
// }

// #[derive(Clone, Debug)]
// pub struct CartPoleAct (u32);

// impl Act for CartPoleAct {}

// impl Into<PyObject> for CartPoleAct {
//     fn into(self) -> PyObject {
//         pyo3::Python::with_gil(|py| {
//             self.0.into_py(py)
//         })
//     }
// }

// impl From<Tensor> for CartPoleAct {
//     fn from(t: Tensor) -> Self {
//         let a: i32 = t.into();
//         Self::new(a as u32)
//     }
// }

// impl CartPoleAct {
//     pub fn new(v: u32) -> Self {
//         CartPoleAct { 0: v }
//     }
// }

// struct CartPoleActBuffer {
//     act: Tensor
// }

// impl TchBuffer for CartPoleActBuffer {
//     type Item = CartPoleAct;
//     type SubBatch = Tensor;

//     fn new(capacity: usize) -> Self {
//         Self {
//             act: Tensor::zeros(&[capacity as _, 1], tch::kind::INT64_CPU),
//         }
//     }

//     fn push(&mut self, index: i64, item: &CartPoleAct) {
//         let act = (item.0 as i32).into();
//         self.act.get(index).copy_(&act);
//     }

//     fn batch(&self, batch_indexes: &Tensor) -> Tensor {
//         self.act.index_select(0, &batch_indexes)
//     }
// }

// type CartpoleEnv = PyVecGymEnv::<CartPoleVecObs, CartPoleAct>;

// fn main() -> Result<(), Box<dyn Error>> {
//     std::env::set_var("RUST_LOG", "info");
//     env_logger::init();
//     tch::manual_seed(42);

//     let env = CartpoleEnv::new("CartPole-v0", None, 2)?;

//     let obs = env.reset();
//     println!("{:?}", obs);

//     env.close(); // TODO: add drop

//     Ok(())
// }
