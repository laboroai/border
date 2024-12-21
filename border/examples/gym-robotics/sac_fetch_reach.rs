use anyhow::Result;
use border_core::{
    generic_replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    record::{Record, RecordValue, Recorder},
    Agent, Configurable, DefaultEvaluator, Env as _, Evaluator as _, ReplayBufferBase,
    StepProcessor, Trainer, TrainerConfig,
};
use border_derive::BatchBase;
use border_mlflow_tracking::MlflowTrackingClient;
use border_py_gym_env::{
    util::{arrayd_to_pyobj, ArrayType},
    ArrayDictObsFilter, GymActFilter, GymEnv, GymEnvConfig, GymObsFilter,
};
use border_tch_agent::{
    mlp::{Mlp, Mlp2, MlpConfig},
    opt::OptimizerConfig,
    sac::{ActorConfig, CriticConfig, EntCoefMode, Sac, SacConfig},
    util::CriticLoss,
    util::{arrayd_to_tensor, tensor_to_arrayd},
    TensorBatch,
};
use border_tensorboard::TensorboardRecorder;
use clap::Parser;
// use csv::WriterBuilder;
use ndarray::ArrayD;
use pyo3::PyObject;
// use serde::Serialize;
use std::convert::TryFrom;
use tch::Tensor;

const DIM_OBS: i64 = 16;
const DIM_ACT: i64 = 4;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const BATCH_SIZE: usize = 128;
const N_TRANSITIONS_WARMUP: usize = 1000;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 20_000_000;
const EVAL_INTERVAL: usize = 2_000;
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
const N_EPISODES_PER_EVAL: usize = 5;
const N_CRITICS: usize = 2;
const TAU: f64 = 0.02;
const TARGET_ENTROPY: f64 = -(DIM_ACT as f64);
const LR_ENT_COEF: f64 = 3e-4;
const CRITIC_LOSS: CriticLoss = CriticLoss::SmoothL1;
const MODEL_DIR: &str = "./border/examples/gym-robotics/model/tch/sac_fetch_reach/";

fn cuda_if_available() -> tch::Device {
    tch::Device::cuda_if_available()
}

mod obs_act_types {
    use super::*;
    use border_py_gym_env::util::Array;

    #[derive(Clone, Debug)]
    pub struct Obs(Vec<(String, Array)>);

    #[derive(Clone, BatchBase)]
    pub struct ObsBatch(TensorBatch);

    impl border_core::Obs for Obs {
        fn len(&self) -> usize {
            match self.0.get(0) {
                None => 0,
                Some(v) => v.1.len(),
            }
        }
    }

    impl From<Vec<(String, Array)>> for Obs {
        fn from(obs: Vec<(String, Array)>) -> Self {
            Obs(obs)
        }
    }

    impl From<Obs> for Tensor {
        fn from(obs: Obs) -> Tensor {
            let arrays = obs.0.into_iter().map(|e| e.1).collect::<Vec<_>>();
            let array = Array::hstack(arrays);
            Tensor::try_from(&array.as_f32_array()).unwrap()
        }
    }

    impl From<Obs> for ObsBatch {
        fn from(obs: Obs) -> Self {
            let tensor = obs.into();
            Self(TensorBatch::from_tensor(tensor))
        }
    }

    #[derive(Clone, Debug)]
    pub struct Act(ArrayD<f32>);

    impl border_core::Act for Act {}

    impl From<Act> for ArrayD<f32> {
        fn from(value: Act) -> Self {
            value.0
        }
    }

    impl From<Tensor> for Act {
        fn from(t: Tensor) -> Self {
            Self(tensor_to_arrayd(t, true))
        }
    }

    // Required by Sac
    impl From<Act> for Tensor {
        fn from(value: Act) -> Self {
            arrayd_to_tensor::<_, f32>(value.0, true)
        }
    }

    #[derive(BatchBase)]
    pub struct ActBatch(TensorBatch);

    impl From<Act> for ActBatch {
        fn from(act: Act) -> Self {
            let tensor = act.into();
            Self(TensorBatch::from_tensor(tensor))
        }
    }

    // Custom activation filter
    #[derive(Clone, Debug)]
    pub struct ActFilter {}

    impl GymActFilter<Act> for ActFilter {
        type Config = ();

        fn build(_config: &Self::Config) -> Result<Self>
        where
            Self: Sized,
        {
            Ok(Self {})
        }

        fn filt(&mut self, act: Act) -> (PyObject, Record) {
            let act_filt = 2f32 * &act.0;
            let record = Record::from_slice(&[
                (
                    "act_org",
                    RecordValue::Array1(act.0.iter().cloned().collect()),
                ),
                (
                    "act_filt",
                    RecordValue::Array1(act_filt.iter().cloned().collect()),
                ),
            ]);
            (arrayd_to_pyobj(act_filt), record)
        }
    }

    pub type ObsFilter = ArrayDictObsFilter<Obs>;
    pub type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
    pub type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
    pub type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
    pub type Evaluator = DefaultEvaluator<Env, Sac<Env, Mlp, Mlp2, ReplayBuffer>>;
}

use obs_act_types::*;

mod config {
    use super::*;

    pub fn env_config() -> GymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
        GymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
            .name("FetchReach-v2".to_string())
            .obs_filter_config(ObsFilter::default_config().add_key_and_types(vec![
                ("observation", ArrayType::F32Array),
                ("desired_goal", ArrayType::F32Array),
                ("achieved_goal", ArrayType::F32Array),
            ]))
            .act_filter_config(ActFilter::default_config())
    }

    pub fn create_trainer_config(max_opts: usize, model_dir: &str) -> TrainerConfig {
        TrainerConfig::default()
            .max_opts(max_opts)
            .opt_interval(OPT_INTERVAL)
            .eval_interval(EVAL_INTERVAL)
            .record_agent_info_interval(EVAL_INTERVAL)
            .record_compute_cost_interval(EVAL_INTERVAL)
            .flush_record_interval(EVAL_INTERVAL)
            .save_interval(EVAL_INTERVAL)
            .warmup_period(N_TRANSITIONS_WARMUP)
            .model_dir(model_dir)
    }

    pub fn create_sac_config(dim_obs: i64, dim_act: i64, target_ent: f64) -> SacConfig<Mlp, Mlp2> {
        let device = cuda_if_available();
        let actor_config = ActorConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_ACTOR })
            .out_dim(dim_act)
            .pi_config(MlpConfig::new(dim_obs, vec![64, 64], dim_act, false));
        let critic_config = CriticConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_CRITIC })
            .q_config(MlpConfig::new(dim_obs + dim_act, vec![64, 64], 1, false));

        SacConfig::default()
            .batch_size(BATCH_SIZE)
            .actor_config(actor_config)
            .critic_config(critic_config)
            .tau(TAU)
            .critic_loss(CRITIC_LOSS)
            .n_critics(N_CRITICS)
            .ent_coef_mode(EntCoefMode::Auto(target_ent, LR_ENT_COEF))
            .device(device)
    }
}

mod utils {
    use super::*;

    pub fn create_recorder(
        model_dir: &str,
        mlflow: bool,
        config: &TrainerConfig,
    ) -> Result<Box<dyn Recorder>> {
        match mlflow {
            true => {
                let client =
                    MlflowTrackingClient::new("http://localhost:8080").set_experiment("Fetch")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&config)?;
                recorder_run.set_tag("env", "reach")?;
                recorder_run.set_tag("algo", "sac")?;
                recorder_run.set_tag("backend", "tch")?;
                Ok(Box::new(recorder_run))
            }
            false => Ok(Box::new(TensorboardRecorder::new(model_dir))),
        }
    }
}

/// Train/eval SAC agent in fetch environment
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Train SAC agent, not evaluate
    #[arg(short, long, default_value_t = false)]
    train: bool,

    /// Evaluate SAC agent, not train
    #[arg(short, long, default_value_t = false)]
    eval: bool,

    /// Log metrics with MLflow
    #[arg(short, long, default_value_t = false)]
    mlflow: bool,
}

fn train(max_opts: usize, model_dir: &str, mlflow: bool) -> Result<()> {
    let trainer_config = config::create_trainer_config(max_opts, model_dir);
    let env_config = config::env_config();
    let step_proc_config = SimpleStepProcessorConfig {};
    let sac_config = config::create_sac_config(DIM_OBS, DIM_ACT, TARGET_ENTROPY);
    let replay_buffer_config = SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);

    let env = Env::build(&env_config, 0)?;
    let step_proc = StepProc::build(&step_proc_config);
    let mut recorder = utils::create_recorder(model_dir, mlflow, &trainer_config)?;
    let mut trainer = Trainer::build(trainer_config);
    let mut agent = Sac::build(sac_config);
    let mut buffer = ReplayBuffer::build(&replay_buffer_config);
    let mut evaluator = Evaluator::new(&config::env_config(), 0, N_EPISODES_PER_EVAL)?;

    trainer.train(
        env,
        step_proc,
        &mut agent,
        &mut buffer,
        &mut recorder,
        &mut evaluator,
    )?;

    Ok(())
}

fn eval(n_episodes: usize, render: bool, model_dir: &str) -> Result<()> {
    let env_config = {
        let mut env_config = config::env_config();
        if render {
            env_config = env_config
                .render_mode(Some("human".to_string()))
                .set_wait_in_millis(10);
        };
        env_config
    };
    let mut agent = {
        let mut agent = Sac::build(config::create_sac_config(DIM_OBS, DIM_ACT, TARGET_ENTROPY));
        agent.load_params(model_dir)?;
        agent.eval();
        Box::new(agent)
    };
    // let mut recorder = BufferedRecorder::new();

    let _ = Evaluator::new(&env_config, 0, n_episodes)?.evaluate(&mut agent);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let args = Args::parse();

    if args.train {
        train(MAX_OPTS, MODEL_DIR, args.mlflow)?;
    } else if args.eval {
        eval(5, true, format!("{}/best", MODEL_DIR).as_str())?;
    } else {
        train(MAX_OPTS, MODEL_DIR, args.mlflow)?;
        eval(5, true, format!("{}/best", MODEL_DIR).as_str())?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use tempdir::TempDir;

    #[test]
    fn test_sac_fetch_reach() -> Result<()> {
        tch::manual_seed(42);

        let model_dir = TempDir::new("sac_fetch_reach")?;
        let model_dir = model_dir.path().to_str().unwrap();
        train(100, model_dir, false)?;
        eval(1, false, (model_dir.to_string() + "/best").as_str())?;

        Ok(())
    }
}
