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
use serde::Serialize;
// use serde::Serialize;
use std::convert::TryFrom;
use tch::Tensor;

const DIM_OBS: i64 = 16;
const DIM_ACT: i64 = 4;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const BATCH_SIZE: usize = 256;
const N_TRANSITIONS_WARMUP: usize = 1000;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 20_000_000;
const EVAL_INTERVAL: usize = 20_000;
const REPLAY_BUFFER_CAPACITY: usize = 131_072;
const N_EPISODES_PER_EVAL: usize = 5;
const N_CRITICS: usize = 2;
const TAU: f64 = 0.05;
const TARGET_ENTROPY: f64 = -(DIM_ACT as f64);
const LR_ENT_COEF: f64 = 3e-4;
const CRITIC_LOSS: CriticLoss = CriticLoss::SmoothL1;
const ENV_NAME: &str = "FetchReach-v3";
const MODEL_DIR: &str = "./border/examples/gym-robotics/model/tch/sac_fetch_reach/";
const MLFLOW_EXPERIMENT_NAME: &str = "Fetch";
const MLFLOW_RUN_NAME: &str = "sac_fetch_reach";
const MLFLOW_TAGS: &[(&str, &str)] = &[("env", "reach"), ("algo", "sac"), ("backend", "tch")];

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
    pub type EnvConfig = GymEnvConfig<Obs, Act, ObsFilter, ActFilter>;
    pub type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
    pub type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
    pub type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
    pub type Evaluator = DefaultEvaluator<Env>;
}

use obs_act_types::*;

mod config {
    use super::*;

    #[derive(Serialize)]
    pub struct SacFetchReachConfig {
        pub env_config: EnvConfig,
        pub agent_config: SacConfig<Mlp, Mlp2>,
        pub trainer_config: TrainerConfig,
    }

    impl SacFetchReachConfig {
        pub fn new(
            in_dim: i64,
            out_dim: i64,
            max_opts: usize,
            model_dir: &str,
            eval_interval: usize,
        ) -> Self {
            let env_config = create_env_config(false);
            let agent_config = create_agent_config(in_dim, out_dim, TARGET_ENTROPY);
            let trainer_config = {
                TrainerConfig::default()
                    .max_opts(max_opts)
                    .opt_interval(OPT_INTERVAL)
                    .eval_interval(eval_interval)
                    .record_agent_info_interval(eval_interval)
                    .record_compute_cost_interval(eval_interval)
                    .flush_record_interval(eval_interval)
                    .save_interval(eval_interval)
                    .warmup_period(N_TRANSITIONS_WARMUP)
                    .model_dir(model_dir)
            };
            Self {
                env_config,
                agent_config,
                trainer_config,
            }
        }
    }

    pub fn create_env_config(render: bool) -> GymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
        let mut env_config = GymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
            .name(ENV_NAME.to_string())
            .obs_filter_config(ObsFilter::default_config().add_key_and_types(vec![
                ("observation", ArrayType::F32Array),
                ("desired_goal", ArrayType::F32Array),
                ("achieved_goal", ArrayType::F32Array),
            ]))
            .act_filter_config(ActFilter::default_config());

        if render {
            env_config = env_config
                .render_mode(Some("human".to_string()))
                .set_wait_in_millis(10);
        };
        env_config
    }

    pub fn create_agent_config(
        dim_obs: i64,
        dim_act: i64,
        target_ent: f64,
    ) -> SacConfig<Mlp, Mlp2> {
        let device = cuda_if_available();
        let actor_config = ActorConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_ACTOR })
            .out_dim(dim_act)
            .pi_config(MlpConfig::new(dim_obs, vec![256, 256, 256], dim_act, false));
        let critic_config = CriticConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_CRITIC })
            .q_config(MlpConfig::new(
                dim_obs + dim_act,
                vec![256, 256, 256],
                1,
                false,
            ));

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

use config::{create_agent_config, create_env_config, SacFetchReachConfig};

/// `model_dir` - Directory where TFRecord and model parameters are saved with
///               [`TensorboardRecorder`].
/// `config` - Configuration parameters for a run of MLflow. These are used for
///            recording purpose only when a new run is created.
fn create_recorder<T: Serialize>(
    args: &Args,
    model_dir: &str,
    config: Option<&T>,
) -> Result<Box<dyn Recorder<Env, ReplayBuffer>>> {
    match args.mlflow {
        true => {
            let client = MlflowTrackingClient::new("http://localhost:8080")
                .set_experiment(MLFLOW_EXPERIMENT_NAME)?;
            let recorder_run = client.create_recorder(MLFLOW_RUN_NAME)?;
            if let Some(config) = config {
                recorder_run.log_params(config)?;
                recorder_run.set_tags(MLFLOW_TAGS)?;
            }
            Ok(Box::new(recorder_run))
        }
        false => Ok(Box::new(TensorboardRecorder::new(
            model_dir, model_dir, false,
        ))),
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

fn train(args: &Args, max_opts: usize, model_dir: &str, eval_interval: usize) -> Result<()> {
    let config = SacFetchReachConfig::new(DIM_OBS, DIM_ACT, max_opts, model_dir, eval_interval);
    let step_proc_config = SimpleStepProcessorConfig {};
    let replay_buffer_config = SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
    let mut recorder = create_recorder(&args, model_dir, Some(&config))?;
    let mut trainer = Trainer::build(config.trainer_config.clone());

    let env = Env::build(&config.env_config, 0)?;
    let step_proc = StepProc::build(&step_proc_config);
    let mut agent = Box::new(Sac::build(config.agent_config)) as _;
    let mut buffer = ReplayBuffer::build(&replay_buffer_config);
    let mut evaluator = Evaluator::new(&config.env_config, 0, N_EPISODES_PER_EVAL)?;

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

fn eval(args: &Args, model_dir: &str, render: bool) -> Result<()> {
    let env_config = create_env_config(render);
    let mut agent: Box<dyn Agent<_, ReplayBuffer>> = {
        let agent_config = create_agent_config(DIM_OBS, DIM_ACT, TARGET_ENTROPY);
        let mut agent = Box::new(Sac::build(agent_config)) as _;
        let recorder = create_recorder::<()>(&args, model_dir, None)?;
        recorder.load_model("best".as_ref(), &mut agent)?;
        agent.eval();
        agent
    };
    let _ = Evaluator::new(&env_config, 0, 5)?.evaluate(&mut agent);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let args = Args::parse();

    if args.train {
        train(&args, MAX_OPTS, MODEL_DIR, EVAL_INTERVAL)?;
    } else if args.eval {
        eval(&args, MODEL_DIR, true)?;
    } else {
        train(&args, MAX_OPTS, MODEL_DIR, EVAL_INTERVAL)?;
        eval(&args, MODEL_DIR, true)?;
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

        let tmp_dir = TempDir::new("sac_fetch_reach")?;
        let model_dir = match tmp_dir.as_ref().to_str() {
            Some(s) => s,
            None => panic!("Failed to get string of temporary directory"),
        };
        let args = Args {
            train: false,
            eval: false,
            mlflow: false,
        };

        let model_dir = TempDir::new("sac_fetch_reach")?;
        let model_dir = model_dir.path().to_str().unwrap();
        train(100, model_dir, false, 100)?;
        eval(&args, model_dir, false)?;
        Ok(())
    }
}
