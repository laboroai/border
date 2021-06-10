use anyhow::Result;
use border::try_from;
use border_core::{
    record::{BufferedRecorder, Record, RecordValue, TensorboardRecorder},
    shape,
    util::eval_with_recorder,
    Agent, Shape, TrainerBuilder,
};
use border_py_gym_env::{
    newtype_act_c, newtype_obs, to_pyobj, PyGymEnv, PyGymEnvActFilter, PyGymEnvBuilder,
    PyGymEnvContinuousAct,
};
use border_tch_agent::{
    model::{SubModel, SubModel2},
    opt::OptimizerConfig,
    replay_buffer::TchTensorBuffer,
    sac::{Actor, ActorBuilder, Critic, CriticBuilder, EntCoefMode, SACBuilder},
    util::{CriticLoss, OptInterval, OutDim},
};
use ndarray::{Array1, IxDyn};
use pyo3::PyObject;
use serde::{Deserialize, Serialize};
use std::{convert::TryFrom, fs::File};
use tch::{nn, nn::Module, Device, Tensor};

// use anyhow::Result;
// use border::{
//     agent::{
//         tch::{
//             model::{SubModel, SubModel2},
//             opt::OptimizerConfig,
//             sac::{Actor, ActorBuilder, Critic, CriticBuilder, EntCoefMode},
//             util::OutDim,
//             ReplayBuffer, SACBuilder,
//         },
//         CriticLoss, OptInterval,
//     },
//     env::py_gym_env::{
//         act_c::{to_pyobj, PyGymEnvContinuousAct},
//         obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
//         tch::{act_c::TchPyGymEnvContinuousActBuffer, obs::TchPyGymEnvObsBuffer},
//         PyGymEnv, PyGymEnvActFilter, PyGymEnvBuilder, Shape,
//     },
//     shape,
// };
// use border_core::{
//     record::{BufferedRecorder, Record, RecordValue, TensorboardRecorder},
//     util::eval_with_recorder,
//     Agent, TrainerBuilder,
// };
// use pyo3::PyObject;
// use serde::{Deserialize, Serialize};
// use std::{convert::TryFrom, fs::File};
// use tch::{
//     nn::{self, Module},
//     Device, Tensor,
// };

const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const N_CRITICS: usize = 1;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 128;
const N_TRANSITIONS_WARMUP: usize = 1000;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.001;
const ALPHA: f64 = 1.0;
// const TARGET_ENTROPY: f64 = -(ACT_DIM as f64);
// const LR_ENT_COEF: f64 = 3e-4;
const REWARD_SCALE: f32 = 1.0;
const CRITIC_LOSS: CriticLoss = CriticLoss::SmoothL1;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const MAX_OPTS: usize = 40_000;
const EVAL_INTERVAL: usize = 2_000;
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
const N_EPISODES_PER_EVAL: usize = 5;
const MAX_STEPS_IN_EPISODE: usize = 200;
const MODEL_DIR: &str = "./examples/model/sac_pendulum";

shape!(ObsShape, [3]);
shape!(ActShape, [1]);
newtype_obs!(Obs, ObsFilter, ObsShape, f64, f32);
newtype_act_c!(Act, ActShape);

impl From<Obs> for Tensor {
    fn from(obs: Obs) -> Tensor {
        try_from(obs.0.obs).unwrap()
    }
}

impl From<Act> for Tensor {
    fn from(act: Act) -> Tensor {
        let v = act.0.act.iter().map(|e| *e as f32).collect::<Vec<_>>();
        let t: Tensor = TryFrom::<Vec<f32>>::try_from(v).unwrap();

        // The first dimension of the action tensor is the number of processes,
        // which is 1 for the non-vectorized environment.
        t.unsqueeze(0)
    }
}

impl From<Tensor> for Act {
    /// `t` must be a 1-dimentional tensor of `f32`.
    fn from(t: Tensor) -> Self {
        // In non-vectorized environment, the batch dimension is not required, thus dropped.
        let shape = t.size()[1..]
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>();
        let act: Vec<f32> = t.into();
        let act = act.iter().map(|e| 2f32 * e).collect::<Vec<_>>();

        let act = Array1::<f32>::from(act).into_shape(IxDyn(&shape)).unwrap();

        Act(PyGymEnvContinuousAct::new(act))
    }
}

// Custom activation filter
#[derive(Clone, Debug)]
struct ActFilter {}

impl PyGymEnvActFilter<Act> for ActFilter {
    fn filt(&mut self, act: Act) -> (PyObject, Record) {
        let act_filt = 2f32 * &act.0.act;
        let record = Record::from_slice(&[
            (
                "act_org",
                RecordValue::Array1(act.0.act.iter().cloned().collect()),
            ),
            (
                "act_filt",
                RecordValue::Array1(act_filt.iter().cloned().collect()),
            ), // ("act_org", RecordValue::Array1(Vec::from_iter(act.act.iter().cloned()))),
               // ("act_filt", RecordValue::Array1(Vec::from_iter(act_filt.iter().cloned())))
        ]);
        (to_pyobj::<ActShape>(act_filt), record)
    }
}

type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchTensorBuffer<f32, ObsShape, Obs>;
type ActBuffer = TchTensorBuffer<f32, ActShape, Act>;

#[allow(clippy::clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct MLPConfig {
    in_dim: i64,
    units: Vec<i64>,
    out_dim: i64,
}

impl MLPConfig {
    fn new(in_dim: i64, units: Vec<i64>, out_dim: i64) -> Self {
        Self {
            in_dim,
            units,
            out_dim,
        }
    }
}

impl OutDim for MLPConfig {
    fn get_out_dim(&self) -> i64 {
        self.out_dim
    }

    fn set_out_dim(&mut self, out_dim: i64) {
        self.out_dim = out_dim;
    }
}

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct MLP {
    in_dim: i64,
    units: Vec<i64>,
    out_dim: i64,
    device: Device,
    seq: nn::Sequential,
}

pub fn mlp(var_store: &nn::VarStore, config: &MLPConfig) -> nn::Sequential {
    let mut seq = nn::seq();
    let mut in_dim = config.in_dim;
    let p = &var_store.root();

    for (i, &n) in config.units.iter().enumerate() {
        seq = seq.add(nn::linear(
            p / format!("l{}", i),
            in_dim,
            n,
            Default::default(),
        ));
        seq = seq.add_fn(|x| x.relu());
        in_dim = n;
    }

    seq
}

impl SubModel2 for MLP {
    type Config = MLPConfig;
    type Input1 = Tensor;
    type Input2 = Tensor;
    type Output = Tensor;

    fn forward(&self, input1: &Self::Input1, input2: &Self::Input2) -> Self::Output {
        let input = Tensor::cat(&[input1, input2], -1).to(self.device);
        self.seq.forward(&input.to(self.device))
    }

    fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
        let units = &config.units;
        let in_dim = *units.last().unwrap_or(&config.in_dim);
        let out_dim = config.out_dim;
        let p = &var_store.root();
        let seq = mlp(var_store, &config).add(nn::linear(
            p / format!("l{}", units.len()),
            in_dim,
            out_dim,
            Default::default(),
        ));

        Self {
            in_dim: config.in_dim,
            units: config.units,
            out_dim: config.out_dim,
            device: var_store.device(),
            seq,
        }
    }

    fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
        Self::build(
            var_store,
            Self::Config {
                in_dim: self.in_dim,
                units: self.units.clone(),
                out_dim: self.out_dim,
            },
        )
    }
}

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct MLP2 {
    in_dim: i64,
    units: Vec<i64>,
    out_dim: i64,
    device: Device,
    head1: nn::Linear,
    head2: nn::Linear,
    seq: nn::Sequential,
}

impl SubModel for MLP2 {
    type Config = MLPConfig;
    type Input = Tensor;
    type Output = (Tensor, Tensor);

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let x = self.seq.forward(&input.to(self.device));
        let mean = x.apply(&self.head1);
        let std = x.apply(&self.head2).exp();
        (mean, std)
    }

    fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
        let units = config.units;
        let out_dim = config.out_dim;
        let device = var_store.device();
        let mut seq = nn::seq();
        let mut in_dim = config.in_dim;
        let p = &var_store.root();

        for (i, &n) in units.iter().enumerate() {
            seq = seq.add(nn::linear(
                p / format!("l{}", i),
                in_dim,
                n,
                Default::default(),
            ));
            seq = seq.add_fn(|x| x.relu());
            in_dim = n;
        }

        let head1 = nn::linear(p / "head1", in_dim, out_dim as _, Default::default());
        let head2 = nn::linear(p / "head2", in_dim, out_dim as _, Default::default());

        let in_dim = config.in_dim;

        Self {
            in_dim,
            units,
            out_dim,
            device,
            head1,
            head2,
            seq,
        }
    }

    fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
        let config = Self::Config {
            in_dim: self.in_dim,
            units: self.units.clone(),
            out_dim: self.out_dim,
        };

        Self::build(var_store, config)
    }
}

pub fn create_actor(
    in_dim: i64,
    out_dim: i64,
    lr_actor: f64,
    units: Vec<i64>,
    device: Device,
) -> Result<Actor<MLP2>> {
    ActorBuilder::default()
        .pi_config(MLPConfig::new(in_dim, units, out_dim))
        .opt_config(OptimizerConfig::Adam { lr: lr_actor })
        .build(device)
}

pub fn create_critic(
    in_dim: i64,
    out_dim: i64,
    lr_critic: f64,
    units: Vec<i64>,
    device: Device,
) -> Result<Critic<MLP>> {
    CriticBuilder::default()
        .q_config(MLPConfig::new(in_dim, units, out_dim))
        .opt_config(OptimizerConfig::Adam { lr: lr_critic })
        .build(device)
}

fn create_agent() -> Result<impl Agent<Env>> {
    let device = tch::Device::cuda_if_available();
    let actor = create_actor(
        ObsShape::shape()[0] as _,
        ActShape::shape()[0] as _,
        LR_ACTOR,
        vec![64, 64],
        device,
    )?;
    let critics = (0..N_CRITICS)
        .map(|_| {
            create_critic(
                (ObsShape::shape()[0] + ActShape::shape()[0]) as _,
                1,
                LR_CRITIC,
                vec![64, 64],
                device,
            )
            .expect("Cannot create critic")
        })
        .collect::<Vec<_>>();

    Ok(SACBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU)
        // .ent_coef_mode(EntCoefMode::Auto(TARGET_ENTROPY, LR_ENT_COEF))
        .ent_coef_mode(EntCoefMode::Fix(ALPHA))
        .reward_scale(REWARD_SCALE)
        .critic_loss(CRITIC_LOSS)
        .replay_burffer_capacity(REPLAY_BUFFER_CAPACITY)
        .build::<_, _, _, ObsBuffer, ActBuffer>(critics, actor, device))
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::default();
    let act_filter = ActFilter {};
    PyGymEnvBuilder::default()
        .build("Pendulum-v0", obs_filter, act_filter)
        .unwrap()
        .max_steps(Some(MAX_STEPS_IN_EPISODE)) // TODO: consider moving the method to the builder
}

#[derive(Debug, Serialize)]
struct PendulumRecord {
    episode: usize,
    step: usize,
    reward: f32,
    obs: Vec<f32>,
    act_org: Vec<f32>,
    act_filt: Vec<f32>,
}

impl TryFrom<&Record> for PendulumRecord {
    type Error = anyhow::Error;

    fn try_from(record: &Record) -> Result<Self> {
        Ok(Self {
            episode: record.get_scalar("episode")? as _,
            step: record.get_scalar("step")? as _,
            reward: record.get_scalar("reward")?,
            obs: record.get_array1("obs")?.to_vec(),
            act_org: record.get_array1("act_org")?.to_vec(),
            act_filt: record.get_array1("act_filt")?.to_vec()
            // obs: Vec::from_iter(record.get_array1("obs")?.iter().cloned()),
            // act_org: Vec::from_iter(record.get_array1("act_org")?.iter().cloned()),
            // act_filt: Vec::from_iter(record.get_array1("act_filt")?.iter().cloned()),
        })
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let env = create_env();
    let env_eval = create_env();
    let agent = create_agent()?;
    let mut trainer = TrainerBuilder::default()
        .max_opts(MAX_OPTS)
        .eval_interval(EVAL_INTERVAL)
        .n_episodes_per_eval(N_EPISODES_PER_EVAL)
        .model_dir(MODEL_DIR)
        .build(env, env_eval, agent);
    let mut recorder = TensorboardRecorder::new("./examples/model/sac_pendulum");

    trainer.train(&mut recorder);

    let mut env = create_env();
    let mut agent = create_agent()?;
    let mut recorder = BufferedRecorder::new();
    env.set_render(true);
    agent.load(MODEL_DIR).unwrap();
    agent.eval();

    let reward = eval_with_recorder(&mut env, &mut agent, 5, &mut recorder);
    println!("{:?}", reward);

    // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create("examples/model/sac_pendulum_eval.csv")?);
    for record in recorder.iter() {
        wtr.serialize(PendulumRecord::try_from(record)?)?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use tempdir::TempDir;

    #[test]
    fn test_sac_pendulum() -> Result<()> {
        // env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
        tch::manual_seed(42);

        let model_dir = TempDir::new("sac_pendulum")?;
        let model_dir = model_dir.path().to_str().unwrap();

        let env = create_env();
        let env_eval = create_env();
        let agent = create_agent()?;
        let mut trainer = TrainerBuilder::default()
            .max_opts(100)
            .eval_interval(100)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .model_dir(model_dir)
            .build(env, env_eval, agent);
        let mut recorder = TensorboardRecorder::new(model_dir);

        trainer.train(&mut recorder);

        let mut env = create_env();
        let mut agent = create_agent()?;
        let mut recorder = BufferedRecorder::new();
        // env.set_render(true);
        agent.load(model_dir).unwrap();
        agent.eval();

        eval_with_recorder(&mut env, &mut agent, 5, &mut recorder);

        Ok(())
    }
}
