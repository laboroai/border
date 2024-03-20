use anyhow::Result;
use border::util::get_model_from_url;
use border_core::{
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    Agent, DefaultEvaluator, Evaluator as _, Policy, Trainer, TrainerConfig,
    record::Recorder,
};
use border_derive::SubBatch;
use border_mlflow_tracking::MlflowTrackingClient;
use border_py_gym_env::{
    util::{arrayd_to_tensor, tensor_to_arrayd},
    ArrayObsFilter, ContinuousActFilter, GymActFilter, GymEnv, GymEnvConfig, GymObsFilter,
};
use border_tch_agent::{
    mlp::{Mlp, Mlp2, MlpConfig},
    opt::OptimizerConfig,
    sac::{ActorConfig, CriticConfig, EntCoefMode, Sac, SacConfig},
    util::CriticLoss,
    TensorSubBatch,
};
use border_tensorboard::TensorboardRecorder;
use clap::{App, Arg, ArgMatches};
use log::info;
use ndarray::{ArrayD, IxDyn};
use std::convert::TryFrom;
use tch::Tensor;

const DIM_OBS: i64 = 27;
const DIM_ACT: i64 = 8;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const BATCH_SIZE: usize = 256;
const N_TRANSITIONS_WARMUP: usize = 10_000;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 3_000_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 300_000;
const N_EPISODES_PER_EVAL: usize = 5;
const N_CRITICS: usize = 2;
const TAU: f64 = 0.02;
const TARGET_ENTROPY: f64 = -(DIM_ACT as f64);
const LR_ENT_COEF: f64 = 3e-4;
const CRITIC_LOSS: CriticLoss = CriticLoss::SmoothL1;
const MODEL_DIR: &str = "./border/examples/model/sac_ant";

type PyObsDtype = f32;

mod obs {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct Obs(ArrayD<f32>);

    #[derive(Clone, SubBatch)]
    pub struct ObsBatch(TensorSubBatch);

    impl border_core::Obs for Obs {
        fn dummy(_n: usize) -> Self {
            Self(ArrayD::zeros(IxDyn(&[0])))
        }

        fn len(&self) -> usize {
            self.0.shape()[0]
        }
    }

    impl From<ArrayD<f32>> for Obs {
        fn from(obs: ArrayD<f32>) -> Self {
            Obs(obs)
        }
    }

    impl From<Obs> for Tensor {
        fn from(obs: Obs) -> Tensor {
            Tensor::try_from(&obs.0).unwrap()
        }
    }

    impl From<Obs> for ObsBatch {
        fn from(obs: Obs) -> Self {
            let tensor = obs.into();
            Self(TensorSubBatch::from_tensor(tensor))
        }
    }
}

mod act {
    use super::*;

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

    #[derive(SubBatch)]
    pub struct ActBatch(TensorSubBatch);

    impl From<Act> for ActBatch {
        fn from(act: Act) -> Self {
            let tensor = act.into();
            Self(TensorSubBatch::from_tensor(tensor))
        }
    }
}

use act::{Act, ActBatch};
use obs::{Obs, ObsBatch};

type ObsFilter = ArrayObsFilter<PyObsDtype, f32, Obs>;
type ActFilter = ContinuousActFilter<Act>;
type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
type Evaluator = DefaultEvaluator<Env, Sac<Env, Mlp, Mlp2, ReplayBuffer>>;

fn create_agent(in_dim: i64, out_dim: i64) -> Sac<Env, Mlp, Mlp2, ReplayBuffer> {
    let device = tch::Device::cuda_if_available();
    let actor_config = ActorConfig::default()
        .opt_config(OptimizerConfig::Adam { lr: LR_ACTOR })
        .out_dim(out_dim)
        .pi_config(MlpConfig::new(in_dim, vec![400, 300], out_dim, false));
    let critic_config = CriticConfig::default()
        .opt_config(OptimizerConfig::Adam { lr: LR_CRITIC })
        .q_config(MlpConfig::new(in_dim + out_dim, vec![400, 300], 1, false));
    let sac_config = SacConfig::default()
        .batch_size(BATCH_SIZE)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .actor_config(actor_config)
        .critic_config(critic_config)
        .tau(TAU)
        .critic_loss(CRITIC_LOSS)
        .n_critics(N_CRITICS)
        .ent_coef_mode(EntCoefMode::Auto(TARGET_ENTROPY, LR_ENT_COEF))
        .device(device);
    Sac::build(sac_config)
}

fn env_config() -> GymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
    GymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
        .name("Ant-v4".to_string())
        .obs_filter_config(ObsFilter::default_config())
        .act_filter_config(ActFilter::default_config())
}

fn create_recorder(
    model_dir: &str,
    mlflow: bool,
    config: &TrainerConfig,
) -> Result<Box<dyn Recorder>> {
    match mlflow {
        true => {
            let client =
                MlflowTrackingClient::new("http://localhost:8080").set_experiment_id("Default")?;
            let recorder_run = client.create_recorder("")?;
            recorder_run.log_params(&config)?;
            Ok(Box::new(recorder_run))
        }
        false => Ok(Box::new(TensorboardRecorder::new(model_dir))),
    }
}

fn train(max_opts: usize, model_dir: &str, mlflow: bool) -> Result<()> {
    let (mut trainer, config) = {
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
            .model_dir(model_dir);
        let trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
            config.clone(),
            env_config,
            step_proc_config,
            replay_buffer_config,
        );

        (trainer, config)
    };
    let mut agent = create_agent(DIM_OBS, DIM_ACT);
    let mut recorder = create_recorder(model_dir, mlflow, &config)?;
    let mut evaluator = Evaluator::new(&env_config(), 0, N_EPISODES_PER_EVAL)?;

    trainer.train(&mut agent, &mut recorder, &mut evaluator)?;

    Ok(())
}

fn eval(model_dir: &str, render: bool, wait: u64) -> Result<()> {
    let env_config = {
        let mut env_config = env_config();
        if render {
            env_config = env_config
                .render_mode(Some("human".to_string()))
                .set_wait_in_millis(wait);
        };
        env_config
    };
    let mut agent = {
        let mut agent = create_agent(DIM_OBS, DIM_ACT);
        agent.load(model_dir)?;
        agent.eval();
        agent
    };
    // let mut recorder = BufferedRecorder::new();

    let _ = Evaluator::new(&env_config, 0, N_EPISODES_PER_EVAL)?.evaluate(&mut agent);

    Ok(())
}

fn create_matches<'a>() -> ArgMatches<'a> {
    App::new("sac_ant")
        .version("0.1.0")
        .author("Taku Yoshioka <yoshioka@laboro.ai>")
        .arg(
            Arg::with_name("play")
                .long("play")
                .takes_value(true)
                .help("Play with the trained model of the given path"),
        )
        .arg(
            Arg::with_name("play-gdrive")
                .long("play-gdrive")
                .takes_value(false)
                .help("Play with the trained model downloaded from google drive"),
        )
        .arg(
            Arg::with_name("wait")
                .long("wait")
                .takes_value(true)
                .default_value("25")
                .help("Waiting time in milliseconds between frames when playing"),
        )
        .arg(
            Arg::with_name("mlflow")
                .long("mlflow")
                .takes_value(false)
                .help("User mlflow tracking"),
        )
        .get_matches()
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let matches = create_matches();
    let mlflow = matches.is_present("mlflow");

    if !(matches.is_present("play") || matches.is_present("play-gdrive")) {
        train(MAX_OPTS, MODEL_DIR, mlflow)?;
    } else {
        let model_dir = if matches.is_present("play") {
            let model_dir = matches
                .value_of("play")
                .expect("Failed to parse model directory");
            format!("{}{}", model_dir, "/best").to_owned()
        } else {
            let file_base = "sac_ant_20210324_ec2_smoothl1";
            let url =
                "https://drive.google.com/uc?export=download&id=1XvFi2nJD5OhpTvs-Et3YREuoqy8c3Vkq";
            let model_dir = get_model_from_url(url, file_base)?;
            info!("Download the model in {:?}", model_dir.as_ref().to_str());
            model_dir.as_ref().to_str().unwrap().to_string()
        };

        let wait = matches.value_of("wait").unwrap().parse().unwrap();
        eval(model_dir.as_str(), true, wait)?;
    }

    Ok(())
}
