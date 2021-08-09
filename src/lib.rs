#![warn(missing_docs)]
//! Border is a reinforcement learning library in Rust.
//! For reusability of both RL environments and agents, this library provides
//! a reference implementation of environments and agents,
//! both of which are independent on each other.
//! For this purpose, this library consists of crates below:
//!
//! * [border-core](border_core) provides base components in RL such as
//!   [Env](border_core::Env), [Policy](border_core::Policy), [Agent](border_core::Agent),
//!   [Trainer](border_core::Trainer), [eval](border_core::eval) (function for evaluation).
//!   It also has utilities like [Record](border_core::record::Record).
//! * [border-py-gym-env](border_py_gym_env) provides a wrapper of OpenAI gym implemented in Python.
//!   Its observation ([PyGymEnvObs](border_py_gym_env::PyGymEnvObs)) action
//!   ([PyGymEnvDiscreteAct](border_py_gym_env::PyGymEnvDiscreteAct) or
//!   [PyGymEnvContinuousAct](border_py_gym_env::PyGymEnvContinuousAct)) are based on [ndarray].
//! * [border-tch-agent](border_tch_agent) is a set of RL agents implemented with [tch].
//!
//! To bridge between environments and agents, you may need to implement some conversions of
//! observation and action data. In `dqn_cartpole` example in `example` directory,
//! which uses [border-py-gym-env](border_py_gym_env) and
//! [border-tch-agent](border_tch_agent), the following conversions are defined:
//!
//! ```no_run
//! use anyhow::Result;
//! use border::try_from;
//! use border_core::{
//!     record::{BufferedRecorder, Record, TensorboardRecorder},
//!     shape, util, Agent, TrainerBuilder,
//! };
//! use border_py_gym_env::{newtype_act_d, newtype_obs, PyGymEnv, PyGymEnvDiscreteAct};
//! use std::convert::TryFrom;
//! use tch::Tensor;
//!
//! const DIM_OBS: i64 = 4;
//!
//! shape!(ObsShape, [DIM_OBS as usize]);
//! shape!(ActShape, [1]);
//! newtype_obs!(Obs, ObsFilter, ObsShape, f64, f32);
//! newtype_act_d!(Act, ActFilter);
//!
//! impl From<Obs> for Tensor {
//!     fn from(obs: Obs) -> Tensor {
//!         try_from(obs.0.obs).unwrap()
//!     }
//! }
//!
//! impl From<Act> for Tensor {
//!     fn from(act: Act) -> Tensor {
//!         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//!         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//!
//!         // The first dimension of the action tensor is the number of processes,
//!         // which is 1 for the non-vectorized environment.
//!         t.unsqueeze(0)
//!     }
//! }
//!
//! impl From<Tensor> for Act {
//!     /// `t` must be a 1-dimentional tensor of `f32`.
//!     fn from(t: Tensor) -> Self {
//!         let data: Vec<i64> = t.into();
//!         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//!         Act(PyGymEnvDiscreteAct::new(data))
//!     }
//! }
//! ```
//!
//! After importing modules, [shape](border_core::shape) macro is used to define the shape of observation and action.
//! It is followed by macros [newtype_obs](border_py_gym_env::newtype_obs) and
//! [newtype_act_d](border_py_gym_env::newtype_act_d). These macros define tuple structs named `Obs` and `Act`,
//! which wraps [PyGymEnvObs](border_py_gym_env::PyGymEnvObs) and
//! [PyGymEnvDiscreteAct](border_py_gym_env::PyGymEnvDiscreteAct), respectively. These tuple structs are required
//! because we implement [From] traits on them. `ObsFilter` and `ActFilter` are
//! wrappers of [PyGymEnvObsRawFilter](border_py_gym_env::PyGymEnvObsRawFilter) and
//! [PyGymEnvDiscreteActRawFilter](border_py_gym_env::PyGymEnvDiscreteActRawFilter), which pass through
//! observation and action without any processing. The 4-th argument of `newtype_obs!` macro, `f64`,
//! means that the cartpole environment outputs its observation as `f64` array. Replacing it with `f32`
//! will result in a runtime error. The last argument, `f32`, is the type of array in
//! [PyGymEnvObs](border_py_gym_env::PyGymEnvObs).
//!
//! Once `Obs` and `Act` are defined, some [From] traits are implemented on these types.
//! These are conversions between `Obs`, `Act` and [Tensor](tch::Tensor)s, as [border-tch-agent](border_tch_agent)
//! are based on [tch].
//!
//! Optionally, you may define type aliases for basic components of reinforcement learning:
//!
//! ```no_run
//! # use anyhow::Result;
//! # use border::try_from;
//! # use border_core::{
//! #     record::{BufferedRecorder, Record, TensorboardRecorder},
//! #     shape, util, Agent, TrainerBuilder,
//! # };
//! # use border_py_gym_env::{newtype_act_d, newtype_obs, PyGymEnv, PyGymEnvDiscreteAct};
//! # use border_tch_agent::{
//! #     dqn::{DQNBuilder, DQNExplorer, EpsilonGreedy},
//! #     replay_buffer::TchTensorBuffer,
//! #     util::OptInterval,
//! # };
//! # use std::convert::TryFrom;
//! # use tch::Tensor;
//! # 
//! # const DIM_OBS: i64 = 4;
//! #
//! # shape!(ObsShape, [DIM_OBS as usize]);
//! # shape!(ActShape, [1]);
//! # newtype_obs!(Obs, ObsFilter, ObsShape, f64, f32);
//! # newtype_act_d!(Act, ActFilter);
//! # 
//! # impl From<Obs> for Tensor {
//! #     fn from(obs: Obs) -> Tensor {
//! #         try_from(obs.0.obs).unwrap()
//! #     }
//! # }
//! # 
//! # impl From<Act> for Tensor {
//! #     fn from(act: Act) -> Tensor {
//! #         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//! #         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//! # 
//! #         // The first dimension of the action tensor is the number of processes,
//! #         // which is 1 for the non-vectorized environment.
//! #         t.unsqueeze(0)
//! #     }
//! # }
//! # 
//! # impl From<Tensor> for Act {
//! #     /// `t` must be a 1-dimentional tensor of `f32`.
//! #     fn from(t: Tensor) -> Self {
//! #         let data: Vec<i64> = t.into();
//! #         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//! #         Act(PyGymEnvDiscreteAct::new(data))
//! #     }
//! # }
//! #
//! type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! type ObsBuffer = TchTensorBuffer<f32, ObsShape, Obs>;
//! type ActBuffer = TchTensorBuffer<i64, ActShape, Act>;
//! ```
//!
//! `Env` requires type parameters `Obs`, `Act`, `ObsFilter` and `ActFilter`, which were defined previously.
//! `ObsBuffer` and `ActBuffer` are [TchTensorBuffer](border_tch_agent::replay_buffer::TchTensorBuffer), where
//! whose data type of the buffer (`f32` or `i64`), shape (`ObsShape` or `ActShape`) are given as type parameters.
//! The last type parameter (`Obs` or `Act`) are types of data those are converted and pushed into the buffer.
//!
//! Functions creating the agent and the environment look like below:
//!
//! ```no_run
//! # use anyhow::Result;
//! # use border::{try_from, dqn_model};
//! # use border_core::{
//! #     record::{BufferedRecorder, Record, TensorboardRecorder},
//! #     shape, util, Agent, TrainerBuilder,
//! # };
//! # use border_py_gym_env::{newtype_act_d, newtype_obs, PyGymEnv, PyGymEnvDiscreteAct};
//! # use border_tch_agent::{
//! #     dqn::{DQNBuilder, DQNExplorer, EpsilonGreedy},
//! #     replay_buffer::TchTensorBuffer,
//! #     util::OptInterval,
//! # };
//! # use std::convert::TryFrom;
//! # use tch::Tensor;
//! #
//! # const DIM_OBS: i64 = 4;
//! # 
//! # shape!(ObsShape, [DIM_OBS as usize]);
//! # shape!(ActShape, [1]);
//! # newtype_obs!(Obs, ObsFilter, ObsShape, f64, f32);
//! # newtype_act_d!(Act, ActFilter);
//! # 
//! # impl From<Obs> for Tensor {
//! #     fn from(obs: Obs) -> Tensor {
//! #         try_from(obs.0.obs).unwrap()
//! #     }
//! # }
//! # 
//! # impl From<Act> for Tensor {
//! #     fn from(act: Act) -> Tensor {
//! #         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//! #         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//! # 
//! #         // The first dimension of the action tensor is the number of processes,
//! #         // which is 1 for the non-vectorized environment.
//! #         t.unsqueeze(0)
//! #     }
//! # }
//! # 
//! # impl From<Tensor> for Act {
//! #     /// `t` must be a 1-dimentional tensor of `f32`.
//! #     fn from(t: Tensor) -> Self {
//! #         let data: Vec<i64> = t.into();
//! #         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//! #         Act(PyGymEnvDiscreteAct::new(data))
//! #     }
//! # }
//! #
//! # type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! # type ObsBuffer = TchTensorBuffer<f32, ObsShape, Obs>;
//! # type ActBuffer = TchTensorBuffer<i64, ActShape, Act>;
//!
//! const DIM_ACT: i64 = 2;
//! const LR_CRITIC: f64 = 0.001;
//! const DISCOUNT_FACTOR: f64 = 0.99;
//! const BATCH_SIZE: usize = 64;
//! const N_TRANSITIONS_WARMUP: usize = 100;
//! const N_UPDATES_PER_OPT: usize = 1;
//! const TAU: f64 = 0.005;
//! const OPT_INTERVAL: OptInterval = OptInterval::Steps(50);
//! const MAX_OPTS: usize = 1000;
//! const EVAL_INTERVAL: usize = 50;
//! const REPLAY_BUFFER_CAPACITY: usize = 10000;
//! const N_EPISODES_PER_EVAL: usize = 5;
//! const MODEL_DIR: &str = "./examples/model/dqn_cartpole";
//!
//! fn create_agent() -> Result<impl Agent<Env>> {
//!     let device = tch::Device::cuda_if_available();
//!     let qnet = dqn_model::create_dqn_model(DIM_OBS, DIM_ACT, LR_CRITIC, device)?;
//!
//!     Ok(DQNBuilder::default()
//!         .opt_interval(OPT_INTERVAL)
//!         .n_updates_per_opt(N_UPDATES_PER_OPT)
//!         .min_transitions_warmup(N_TRANSITIONS_WARMUP)
//!         .batch_size(BATCH_SIZE)
//!         .discount_factor(DISCOUNT_FACTOR)
//!         .tau(TAU)
//!         .replay_burffer_capacity(REPLAY_BUFFER_CAPACITY)
//!         .explorer(DQNExplorer::EpsilonGreedy(EpsilonGreedy::new()))
//!         .build::<_, _, ObsBuffer, ActBuffer>(qnet, device, tch::Device::Cpu))
//! }
//!
//! fn create_env() -> Env {
//!     let obs_filter = ObsFilter::default();
//!     let act_filter = ActFilter::default();
//!     Env::new("CartPole-v0", obs_filter, act_filter, None).unwrap()
//! }
//! ```
//!
//! With the agent and environment created using the above functions, we are able to run training and evaluation
//! of the agent on the environment.
//!
//! ```no_run
//! # use anyhow::Result;
//! # use border::{try_from, dqn_model};
//! # use border_core::{
//! #     record::{BufferedRecorder, Record, TensorboardRecorder},
//! #     shape, util, Agent, TrainerBuilder,
//! # };
//! # use border_py_gym_env::{newtype_act_d, newtype_obs, PyGymEnv, PyGymEnvDiscreteAct};
//! # use border_tch_agent::{
//! #     dqn::{DQNBuilder, DQNExplorer, EpsilonGreedy},
//! #     replay_buffer::TchTensorBuffer,
//! #     util::OptInterval,
//! # };
//! # use std::convert::TryFrom;
//! # use tch::Tensor;
//! #
//! # const DIM_OBS: i64 = 4;
//! # 
//! # shape!(ObsShape, [DIM_OBS as usize]);
//! # shape!(ActShape, [1]);
//! # newtype_obs!(Obs, ObsFilter, ObsShape, f64, f32);
//! # newtype_act_d!(Act, ActFilter);
//! # 
//! # impl From<Obs> for Tensor {
//! #     fn from(obs: Obs) -> Tensor {
//! #         try_from(obs.0.obs).unwrap()
//! #     }
//! # }
//! # 
//! # impl From<Act> for Tensor {
//! #     fn from(act: Act) -> Tensor {
//! #         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//! #         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();
//! # 
//! #         // The first dimension of the action tensor is the number of processes,
//! #         // which is 1 for the non-vectorized environment.
//! #         t.unsqueeze(0)
//! #     }
//! # }
//! # 
//! # impl From<Tensor> for Act {
//! #     /// `t` must be a 1-dimentional tensor of `f32`.
//! #     fn from(t: Tensor) -> Self {
//! #         let data: Vec<i64> = t.into();
//! #         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//! #         Act(PyGymEnvDiscreteAct::new(data))
//! #     }
//! # }
//! #
//! # type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
//! # type ObsBuffer = TchTensorBuffer<f32, ObsShape, Obs>;
//! # type ActBuffer = TchTensorBuffer<i64, ActShape, Act>;
//! #
//! # const DIM_ACT: i64 = 2;
//! # const LR_CRITIC: f64 = 0.001;
//! # const DISCOUNT_FACTOR: f64 = 0.99;
//! # const BATCH_SIZE: usize = 64;
//! # const N_TRANSITIONS_WARMUP: usize = 100;
//! # const N_UPDATES_PER_OPT: usize = 1;
//! # const TAU: f64 = 0.005;
//! # const OPT_INTERVAL: OptInterval = OptInterval::Steps(50);
//! # const MAX_OPTS: usize = 1000;
//! # const EVAL_INTERVAL: usize = 50;
//! # const REPLAY_BUFFER_CAPACITY: usize = 10000;
//! # const N_EPISODES_PER_EVAL: usize = 5;
//! # const MODEL_DIR: &str = "./examples/model/dqn_cartpole";
//! #
//! # fn create_agent() -> Result<impl Agent<Env>> {
//! #     let device = tch::Device::cuda_if_available();
//! #     let qnet = dqn_model::create_dqn_model(DIM_OBS, DIM_ACT, LR_CRITIC, device)?;
//! #
//! #     Ok(DQNBuilder::default()
//! #         .opt_interval(OPT_INTERVAL)
//! #         .n_updates_per_opt(N_UPDATES_PER_OPT)
//! #         .min_transitions_warmup(N_TRANSITIONS_WARMUP)
//! #         .batch_size(BATCH_SIZE)
//! #         .discount_factor(DISCOUNT_FACTOR)
//! #         .tau(TAU)
//! #         .replay_burffer_capacity(REPLAY_BUFFER_CAPACITY)
//! #         .explorer(DQNExplorer::EpsilonGreedy(EpsilonGreedy::new()))
//! #         .build::<_, _, ObsBuffer, ActBuffer>(qnet, device, tch::Device::Cpu))
//! # }
//! #
//! # fn create_env() -> Env {
//! #     let obs_filter = ObsFilter::default();
//! #     let act_filter = ActFilter::default();
//! #     Env::new("CartPole-v0", obs_filter, act_filter, None).unwrap()
//! # }
//! fn main() -> Result<()> {
//!     /// Training
//!     let env = create_env();
//!     let env_eval = create_env();
//!     let agent = create_agent()?;
//!     let mut trainer = TrainerBuilder::default()
//!         .max_opts(MAX_OPTS)
//!         .eval_interval(EVAL_INTERVAL)
//!         .n_episodes_per_eval(N_EPISODES_PER_EVAL)
//!         .model_dir(MODEL_DIR)
//!         .build(env, env_eval, agent);
//!     let mut recorder = TensorboardRecorder::new(MODEL_DIR);
//!    
//!     trainer.train(&mut recorder);
//!    
//!     /// Evaluation
//!     let mut env = create_env();
//!     let mut agent = create_agent()?;
//!     let mut recorder = BufferedRecorder::new();
//!     env.set_render(true);
//!     agent.load(MODEL_DIR)?;
//!     agent.eval();
//!    
//!     util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder);
//!
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! ## Core: [border_core]
//!
//! * Multiple observations and actions, intended to support vectorized environments.
//! * Tensorboard support ([TensorboardRecorder](border_core::record::TensorboardRecorder))
//! * Recording sequences of observation and action ([Record](border_core::record::Record) and
//!   [BufferedRecorder](border_core::record::BufferedRecorder))
//!
//! ## Environments: [border_py_gym_env]
//!
//! * Atari using `examples/atari_wrapper.py` and [pybullet-gym](https://github.com/benelot/pybullet-gym)
//! * Vectorized environment ([PyVecGymEnv](border_py_gym_env::PyVecGymEnv))
//! * Macros to define newtypes of observation and action ([newtype_obs](border_py_gym_env::newtype_obs),
//!   [newtype_act_d](border_py_gym_env::newtype_act_d) and [newtype_act_c](border_py_gym_env::newtype_act_c))
//! * A filter for observation, used in Atari ([FrameStackFilter](border_py_gym_env::FrameStackFilter))
//!
//! ## Agents: [border_tch_agent]
//!
//! * [DQN](border_tch_agent::dqn::DQN), [IQN](border_tch_agent::iqn::IQN) and [SAC](border_tch_agent::sac::SAC) agents
//! * Flexible composition of neural networks using [SubModel](border_tch_agent::model::SubModel)
//! * Model save/load
//! * [serde] implemented on configurations (e.g., [DQNBuilder](border_tch_agent::dqn::DQNBuilder))
//!
//! # License
//!
//! Border is primarily distributed under the terms of both the MIT license and the Apache License (Version 2.0).
//!
pub mod error;
pub mod util;

use ndarray::ArrayD;
use tch::{TchError, Tensor};

/// Converts [ndarray::ArrayD] into tch Tensor.
/// Borrowed from tch-rs. The original code didn't work with ndarray 0.14.
pub fn try_from<T>(value: ArrayD<T>) -> Result<Tensor, TchError>
where
    T: tch::kind::Element,
{
    // TODO: Replace this with `?` once it works with `std::option::ErrorNone`
    let slice = match value.as_slice() {
        None => return Err(TchError::Convert("cannot convert to slice".to_string())),
        Some(v) => v,
    };
    let tn = Tensor::f_of_slice(slice)?;
    let shape: Vec<i64> = value.shape().iter().map(|s| *s as i64).collect();
    // Ok(tn.f_reshape(&shape)?)
    tn.f_reshape(&shape)
}

/// Defined for document.
#[allow(missing_docs)]
pub mod dqn_model {
    use anyhow::Result;
    use border_tch_agent::{
        dqn::{DQNModel, DQNModelBuilder},
        model::SubModel,
        util::OutDim,
    };
    use serde::{Deserialize, Serialize};
    use tch::{nn, nn::Module, Device, Tensor};

    #[allow(clippy::upper_case_acronyms)]
    #[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
    pub struct MLPConfig {
        in_dim: i64,
        out_dim: i64,
    }

    impl OutDim for MLPConfig {
        fn get_out_dim(&self) -> i64 {
            self.out_dim
        }

        fn set_out_dim(&mut self, v: i64) {
            self.out_dim = v;
        }
    }

    impl MLPConfig {
        fn new(in_dim: i64, out_dim: i64) -> Self {
            Self { in_dim, out_dim }
        }
    }

    #[allow(clippy::upper_case_acronyms)]
    // Two layer perceptron
    pub struct MLP {
        in_dim: i64,
        out_dim: i64,
        device: Device,
        seq: nn::Sequential,
    }

    impl MLP {
        fn create_net(var_store: &nn::VarStore, in_dim: i64, out_dim: i64) -> nn::Sequential {
            let p = &var_store.root();
            nn::seq()
                .add(nn::linear(p / "cl1", in_dim, 256, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(p / "cl2", 256, out_dim, Default::default()))
        }
    }

    impl SubModel for MLP {
        type Config = MLPConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, x: &Self::Input) -> Tensor {
            self.seq.forward(&x.to(self.device))
        }

        fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
            let in_dim = config.in_dim;
            let out_dim = config.out_dim;
            let device = var_store.device();
            let seq = Self::create_net(var_store, in_dim, out_dim);

            Self {
                in_dim,
                out_dim,
                device,
                seq,
            }
        }

        fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
            let in_dim = self.in_dim;
            let out_dim = self.out_dim;
            let device = var_store.device();
            let seq = Self::create_net(&var_store, in_dim, out_dim);

            Self {
                in_dim,
                out_dim,
                device,
                seq,
            }
        }
    }

    // DQN model
    pub fn create_dqn_model(
        in_dim: i64,
        out_dim: i64,
        learning_rate: f64,
        device: Device,
    ) -> Result<DQNModel<MLP>> {
        let q_config = MLPConfig::new(in_dim, out_dim);
        DQNModelBuilder::default()
            .opt_config(border_tch_agent::opt::OptimizerConfig::Adam { lr: learning_rate })
            .build_with_submodel_configs(q_config, device)
    }
}
