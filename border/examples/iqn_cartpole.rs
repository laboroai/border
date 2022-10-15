use anyhow::Result;
use border_core::{
    record::{BufferedRecorder, Record, TensorboardRecorder},
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    shape, util, Agent, Policy, Shape, Trainer, TrainerConfig,
};
use border_derive::{Act, Obs, SubBatch};
use border_py_gym_env::{
    PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter,
    PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
};

use border_tch_agent::{
    iqn::{Iqn as Iqn_, IqnConfig, IqnModelConfig},
    mlp::{MLP, MLPConfig},
    TensorSubBatch,
};
use clap::{App, Arg};
// use csv::WriterBuilder;
// use serde::Serialize;
use std::{convert::TryFrom, fs::File};
use tch::Tensor;

const DIM_OBS: i64 = 4;
const DIM_FEATURE: i64 = 256;
const DIM_EMBED: i64 = 64;
const DIM_ACT: i64 = 1;
const LR_CRITIC: f64 = 0.001;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const N_TRANSITIONS_WARMUP: usize = 100;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.1;
const SOFT_UPDATE_INTERVAL: usize = 100;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 10000;
const EVAL_INTERVAL: usize = 500;
const REPLAY_BUFFER_CAPACITY: usize = 10000;
const N_EPISODES_PER_EVAL: usize = 5;
const EPS_START: f64 = 1.0;
const EPS_FINAL: f64 = 0.1;
const FINAL_STEP: usize = 5000; // MAX_OPTS;
const MODEL_DIR: &str = "border/examples/model/iqn_cartpole";

shape!(ObsShape, [DIM_OBS as usize]);
shape!(ActShape, [1]);

type PyObsDtype = f32;
type ObsDtype = f32;

#[derive(Clone, Debug, Obs)]
struct Obs(PyGymEnvObs<ObsShape, PyObsDtype, f32>);

#[derive(Clone, Debug, Act)]
struct Act(PyGymEnvDiscreteAct);

// impl From<Act> for Tensor {
//     fn from(act: Act) -> Tensor {
//         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//         let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();

//         // The first dimension of the action tensor is the number of processes,
//         // which is 1 for the non-vectorized environment.
//         t.unsqueeze(0)
//     }
// }

// impl From<Tensor> for Act {
//     /// `t` must be a 1-dimentional tensor of `f32`.
//     fn from(t: Tensor) -> Self {
//         let data: Vec<i64> = t.into();
//         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//         Act(PyGymEnvDiscreteAct::new(data))
//     }
// }

#[derive(Clone, SubBatch)]
struct ObsBatch(TensorSubBatch<ObsShape, ObsDtype>);

impl From<Obs> for ObsBatch {
    fn from(obs: Obs) -> Self {
        let tensor = obs.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

#[derive(SubBatch)]
struct ActBatch(TensorSubBatch<ActShape, i64>);

impl From<Act> for ActBatch {
    fn from(act: Act) -> Self {
        let tensor = act.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, PyObsDtype, f32, Obs>;
type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
// type ObsBuffer = TchTensorBuffer<f32, ObsShape, Obs>;
// type ActBuffer = TchTensorBuffer<i64, ActShape, Act>;
type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
type Iqn = Iqn_<Env, MLP, MLP, ReplayBuffer>;

// #[derive(Debug, Serialize)]
// struct CartpoleRecord {
//     episode: usize,
//     step: usize,
//     reward: f32,
//     obs: Vec<f64>,
// }

// impl TryFrom<&Record> for CartpoleRecord {
//     type Error = anyhow::Error;

//     fn try_from(record: &Record) -> Result<Self> {
//         Ok(Self {
//             episode: record.get_scalar("episode")? as _,
//             step: record.get_scalar("step")? as _,
//             reward: record.get_scalar("reward")?,
//             obs: record
//                 .get_array1("obs")?
//                 .iter()
//                 .map(|v| *v as f64)
//                 .collect(),
//         })
//     }
// }

fn env_config() -> PyGymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
    PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
        .name("CartPole-v0".to_string())
        .obs_filter_config(ObsFilter::default_config())
        .act_filter_config(ActFilter::default_config())
}

fn create_agent(in_dim: i64, out_dim: i64) -> Iqn {
    let device = tch::Device::cuda_if_available();
    let config = {
        let opt_config = border_tch_agent::opt::OptimizerConfig::Adam { lr: LR_CRITIC };
        let f_config = MLPConfig::new(in_dim, vec![DIM_EMBED], DIM_FEATURE);
        let m_config = MLPConfig::new(DIM_FEATURE, vec![DIM_EMBED], out_dim);
        let model_config = IqnModelConfig::default()
            .feature_dim(DIM_FEATURE)
            .embed_dim(DIM_EMBED)
            .opt_config(opt_config)
            .f_config(f_config)
            .m_config(m_config);

        IqnConfig::default()
            .n_updates_per_opt(N_UPDATES_PER_OPT)
            .min_transitions_warmup(N_TRANSITIONS_WARMUP)
            .batch_size(BATCH_SIZE)
            .discount_factor(DISCOUNT_FACTOR)
            .tau(TAU)
            .model_config(model_config)
            .device(device)
    };

    Iqn::build(config)
}

fn train(max_opts: usize, model_dir: &str) -> Result<()> {
    let mut trainer = {
        let env_config = env_config();
        let step_proc_config = SimpleStepProcessorConfig {};
        let replay_buffer_config =
            SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
        let config = TrainerConfig::default()
            .max_opts(max_opts)
            .opt_interval(OPT_INTERVAL)
            .eval_interval(EVAL_INTERVAL)
            .record_interval(EVAL_INTERVAL)
            .save_interval(EVAL_INTERVAL)
            .eval_episodes(N_EPISODES_PER_EVAL)
            .model_dir(model_dir);
        let trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
            config,
            env_config,
            None,
            step_proc_config,
            replay_buffer_config,
        );

        trainer
    };

    let mut recorder = TensorboardRecorder::new(model_dir);
    let mut agent = create_agent(DIM_OBS, DIM_ACT);

    trainer.train(&mut agent, &mut recorder)?;

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = App::new("dqn_cartpole")
        .version("0.1.0")
        .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
        .arg(
            Arg::with_name("skip training")
                .long("skip_training")
                .takes_value(false)
                .help("Skip training"),
        )
        .get_matches();

    if !matches.is_present("skip training") {
        train(MAX_OPTS, MODEL_DIR)?;
    }

    //     let env = create_env();
    //     let env_eval = create_env();
    //     let agent = create_agent();
    //     let mut trainer = TrainerBuilder::default()
    //         .max_opts(MAX_OPTS)
    //         .eval_interval(EVAL_INTERVAL)
    //         .n_episodes_per_eval(N_EPISODES_PER_EVAL)
    //         .model_dir(MODEL_DIR)
    //         .build(env, env_eval, agent);
    //     let mut recorder = TensorboardRecorder::new(MODEL_DIR);

    //     trainer.train(&mut recorder);
    // }

    // let mut env = create_env();
    // let mut agent = create_agent();
    // let mut recorder = BufferedRecorder::new();
    // env.set_render(true);
    // agent.load(MODEL_DIR)?;
    // agent.eval();

    // util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder);

    // // Vec<_> field in a struct does not support writing a header in csv crate, so disable it.
    // let mut wtr = WriterBuilder::new()
    //     .has_headers(false)
    //     .from_writer(File::create("border/examples/model/iqn_cartpole_eval.csv")?);
    // for record in recorder.iter() {
    //     wtr.serialize(CartpoleRecord::try_from(record)?)?;
    // }

    Ok(())
}
