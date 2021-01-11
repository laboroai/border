use std::error::Error;
use log::trace;
use ndarray::{Axis, ArrayD, IxDyn};
use pyo3::{PyObject, IntoPy};
use numpy::PyArrayDyn;
use tch::Tensor;
use lrr::core::{Obs, Act, Trainer, Agent, util};
use lrr::py_gym_env::PyGymEnv;
use lrr::agents::tch::{DQN, QNetwork, ReplayBuffer, TchBuffer, util::try_from};

#[derive(Clone, Debug)]
pub struct CartPoleObs (pub ArrayD<f32>);

impl Obs for CartPoleObs {
    fn zero(n_procs: usize) -> Self {
        Self(ArrayD::zeros(IxDyn(&[n_procs, 4])))
    }

    fn merge(mut self, obs_reset: Self, is_done: &[f32]) -> Self {
        let any = is_done.iter().fold(0, |x, v| x + *v as i32);
        if any > 0 {
            #[allow(clippy::needless_range_loop)]
            for i in 0..is_done.len() {
                if is_done[i] != 0.0 {
                    self.0.index_axis_mut(Axis(0), i).assign(&obs_reset.0.index_axis(Axis(0), i));
                }
            }
            self
        }
        else {
            self
        }
    }
}

impl From<PyObject> for CartPoleObs {
    fn from(obs: PyObject) -> Self {
        pyo3::Python::with_gil(|py| {
            let obs: &PyArrayDyn<f64> = obs.extract(py).unwrap();
            let obs = obs.to_owned_array();
            let obs = obs.mapv(|elem| elem as f32);
            let obs = obs.insert_axis(Axis(0));
            trace!("obs from PyObject: {:?}", obs);
            Self(obs)
        })
    }
}

impl Into<Tensor> for CartPoleObs {
    fn into(self) -> Tensor {
        try_from(self.0).unwrap()
    }
}

struct CartPoleObsBuffer {
    obs: Tensor
}

impl TchBuffer for CartPoleObsBuffer {
    type Item = CartPoleObs;
    type SubBatch = Tensor;

    fn new(capacity: usize, n_procs: usize) -> Self {
        let capacity = capacity as _;
        let n_procs = n_procs as _;
        Self {
            obs: Tensor::zeros(&[capacity, n_procs, 4], tch::kind::FLOAT_CPU),
        }
    }

    fn push(&mut self, index: i64, item: &CartPoleObs) {
        let obs: Tensor = item.clone().into();
        self.obs.get(index).copy_(&obs);
    }

    /// Create minibatch.
    /// The second axis is squeezed, thus the batch size is
    /// `batch_indexes.len()` times `n_procs`.
    fn batch(&self, batch_indexes: &Tensor) -> Tensor {
        let batch = self.obs.index_select(0, &batch_indexes);
        batch.flatten(0, 1)
    }
}

#[derive(Clone, Debug)]
pub struct CartPoleAct (u32);

impl Act for CartPoleAct {}

impl Into<PyObject> for CartPoleAct {
    fn into(self) -> PyObject {
        pyo3::Python::with_gil(|py| {
            self.0.into_py(py)
        })
    }
}

impl From<Tensor> for CartPoleAct {
    fn from(t: Tensor) -> Self {
        let a: i32 = t.into();
        Self::new(a as u32)
    }
}

impl CartPoleAct {
    pub fn new(v: u32) -> Self {
        CartPoleAct { 0: v }
    }
}

struct CartPoleActBuffer {
    act: Tensor
}

impl TchBuffer for CartPoleActBuffer {
    type Item = CartPoleAct;
    type SubBatch = Tensor;

    fn new(capacity: usize, n_procs: usize) -> Self {
        Self {
            act: Tensor::zeros(&[capacity as _, n_procs as _, 1], tch::kind::INT64_CPU),
        }
    }

    fn push(&mut self, index: i64, item: &CartPoleAct) {
        let act = (item.0 as i32).into();
        self.act.get(index).copy_(&act);
    }

    /// Create minibatch.
    /// The second axis is squeezed, thus the batch size is
    /// `batch_indexes.len()` times `n_procs`.
    fn batch(&self, batch_indexes: &Tensor) -> Tensor {
        self.act.index_select(0, &batch_indexes).flatten(0, 1)
    }
}

fn create_agent() -> impl Agent<PyGymEnv<CartPoleObs, CartPoleAct>> {
    type E = PyGymEnv<CartPoleObs, CartPoleAct>;
    type O = CartPoleObsBuffer;
    type A = CartPoleActBuffer;

    let qnet = QNetwork::new(4, 2, 0.001);
    let replay_buffer 
        = ReplayBuffer::<E, O, A>::new(10000, 1);
    let agent: DQN<E, _, _, _> = DQN::new(
        qnet,
        replay_buffer)
        .n_samples_per_opt(50)
        .n_updates_per_opt(1)
        .n_opts_per_soft_update(1)
        .min_transitions_warmup(100)
        .batch_size(64)
        .discount_factor(0.99)
        .tau(0.005);
    agent
}

fn main() -> Result<(), Box<dyn Error>> {
    std::env::set_var("RUST_LOG", "info");
    // std::env::set_var("RUST_LOG", "trace");
    env_logger::init();
    tch::manual_seed(42);

    let env = PyGymEnv::<CartPoleObs, CartPoleAct>::new("CartPole-v0")?;
    let env_eval = PyGymEnv::<CartPoleObs, CartPoleAct>::new("CartPole-v0")?;
    let agent = create_agent();
    let mut trainer = Trainer::new(
        env,
        env_eval,
        agent)
        .max_opts(1000)
        .n_opts_per_eval(50)
        .n_episodes_per_eval(5);

    trainer.train();
    trainer.get_agent().save("./examples/model/dqn_cartpole")?;

    let mut env = PyGymEnv::<CartPoleObs, CartPoleAct>::new("CartPole-v0")?;
    let mut agent = create_agent();
    env.set_render(true);
    agent.load("./examples/model/dqn_cartpole")?;
    agent.eval();
    util::eval(&env, &agent, 5, None);

    Ok(())
}
