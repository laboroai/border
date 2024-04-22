use anyhow::Result;
use border_candle_agent::{
    mlp::{Mlp, Mlp2, MlpConfig},
    opt::OptimizerConfig,
    sac::{ActorConfig, CriticConfig, Sac, SacConfig},
    TensorSubBatch,
};
use border_core::{
    record::{AggregateRecorder, Record},
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    Agent, DefaultEvaluator, Evaluator as _, Policy, Trainer, TrainerConfig,
};
use border_derive::SubBatch;
use border_py_gym_env::{
    util::{arrayd_to_tensor, tensor_to_arrayd},
    ArrayObsFilter, ContinuousActFilter, GymActFilter, GymEnv, GymEnvConfig, GymObsFilter,
};
use border_tensorboard::TensorboardRecorder;
use clap::{App, Arg, ArgMatches};
//use csv::WriterBuilder;
use border_mlflow_tracking::MlflowTrackingClient;
use candle_core::Tensor;
use ndarray::{ArrayD, IxDyn};
use serde::Serialize;
use std::convert::TryFrom;

const DIM_OBS: i64 = 8;
const DIM_ACT: i64 = 2;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const BATCH_SIZE: usize = 128;
const WARMUP_PERIOD: usize = 1000;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 200_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
const N_EPISODES_PER_EVAL: usize = 5;
const MODEL_DIR: &str = "./border/examples/gym/model/candle/sac_lunarlander_cont";

fn cuda_if_available() -> candle_core::Device {
    candle_core::Device::cuda_if_available(0).unwrap()
}

mod obs_act_types {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct Obs(ArrayD<f32>);

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
            arrayd_to_tensor::<_, f32>(obs.0, false).unwrap()
        }
    }

    #[derive(Clone, SubBatch)]
    pub struct ObsBatch(TensorSubBatch);

    impl From<Obs> for ObsBatch {
        fn from(obs: Obs) -> Self {
            let tensor = obs.into();
            Self(TensorSubBatch::from_tensor(tensor))
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
            Self(tensor_to_arrayd(t, true).unwrap())
        }
    }

    // Required by Sac
    impl From<Act> for Tensor {
        fn from(value: Act) -> Self {
            arrayd_to_tensor::<_, f32>(value.0, true).unwrap()
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

    type PyObsDtype = f32;
    pub type ObsFilter = ArrayObsFilter<PyObsDtype, f32, Obs>;
    pub type ActFilter = ContinuousActFilter<Act>;
    pub type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
    pub type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
    pub type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
    pub type Evaluator = DefaultEvaluator<Env, Sac<Env, Mlp, Mlp2, ReplayBuffer>>;
}

use obs_act_types::*;

mod config {
    use serde::Serialize;

    use super::*;

    #[derive(Serialize)]
    pub struct SacLunarLanderConfig {
        pub trainer_config: TrainerConfig,
        pub replay_buffer_config: SimpleReplayBufferConfig,
        pub agent_config: SacConfig<Mlp, Mlp2>,
    }

    pub fn env_config() -> GymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
        GymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
            .name("LunarLanderContinuous-v2".to_string())
            .obs_filter_config(ObsFilter::default_config())
            .act_filter_config(ActFilter::default_config())
    }

    pub fn trainer_config(max_opts: usize, eval_interval: usize) -> TrainerConfig {
        TrainerConfig::default()
            .max_opts(max_opts)
            .opt_interval(OPT_INTERVAL)
            .eval_interval(eval_interval)
            .record_agent_info_interval(EVAL_INTERVAL)
            .record_compute_cost_interval(EVAL_INTERVAL)
            .flush_record_interval(EVAL_INTERVAL)
            .save_interval(EVAL_INTERVAL)
            .warmup_period(WARMUP_PERIOD)
            .model_dir(MODEL_DIR)
    }

    pub fn agent_config(in_dim: i64, out_dim: i64) -> SacConfig<Mlp, Mlp2> {
        let device = cuda_if_available();
        let actor_config = ActorConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_ACTOR })
            .out_dim(out_dim)
            .pi_config(MlpConfig::new(in_dim, vec![64, 64], out_dim, false));
        let critic_config = CriticConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_CRITIC })
            .q_config(MlpConfig::new(in_dim + out_dim, vec![64, 64], 1, false));

        SacConfig::default()
            .batch_size(BATCH_SIZE)
            .actor_config(actor_config)
            .critic_config(critic_config)
            .device(device)
    }
}

#[derive(Debug, Serialize)]
struct LunarlanderRecord {
    episode: usize,
    step: usize,
    reward: f32,
    obs: Vec<f32>,
    act: Vec<f32>,
}

impl TryFrom<&Record> for LunarlanderRecord {
    type Error = anyhow::Error;

    fn try_from(record: &Record) -> Result<Self> {
        Ok(Self {
            episode: record.get_scalar("episode")? as _,
            step: record.get_scalar("step")? as _,
            reward: record.get_scalar("reward")?,
            obs: record.get_array1("obs")?.to_vec(),
            act: record.get_array1("act")?.to_vec(),
        })
    }
}

mod utils {
    use super::*;

    pub fn create_recorder(
        matches: &ArgMatches,
        config: &config::SacLunarLanderConfig,
    ) -> Result<Box<dyn AggregateRecorder>> {
        match matches.is_present("mlflow") {
            true => {
                let client =
                    MlflowTrackingClient::new("http://localhost:8080").set_experiment_id("Gym")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&config)?;
                recorder_run.set_tag("env", "lunarlander")?;
                recorder_run.set_tag("algo", "sac")?;
                recorder_run.set_tag("backend", "candle")?;
                Ok(Box::new(recorder_run))
            }
            false => Ok(Box::new(TensorboardRecorder::new(MODEL_DIR))),
        }
    }

    pub fn create_matches<'a>() -> ArgMatches<'a> {
        App::new("sac_lunarlander_cont")
            .version("0.1.0")
            .author("Taku Yoshioka <yoshioka@laboro.ai>")
            .arg(
                Arg::with_name("train")
                    .long("train")
                    .takes_value(false)
                    .help("Do training only"),
            )
            .arg(
                Arg::with_name("eval")
                    .long("eval")
                    .takes_value(false)
                    .help("Do evaluation only"),
            )
            .arg(
                Arg::with_name("mlflow")
                    .long("mlflow")
                    .takes_value(false)
                    .help("User mlflow tracking"),
            )
            .get_matches()
    }
}

fn train(matches: ArgMatches, max_opts: usize) -> Result<()> {
    let (mut trainer, config) = {
        // Configs
        let env_config = config::env_config();
        let step_proc_config = SimpleStepProcessorConfig {};
        let replay_buffer_config =
            SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
        let trainer_config = config::trainer_config(max_opts, EVAL_INTERVAL);
        let agent_config = config::agent_config(DIM_OBS, DIM_ACT);

        let trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
            trainer_config.clone(),
            env_config,
            step_proc_config,
            replay_buffer_config.clone(),
        );

        // For logging
        let config = config::SacLunarLanderConfig {
            agent_config,
            replay_buffer_config,
            trainer_config,
        };

        (trainer, config)
    };
    let mut recorder = utils::create_recorder(&matches, &config)?;
    let mut agent = Sac::build(config.agent_config);
    let mut evaluator = Evaluator::new(&config::env_config(), 0, N_EPISODES_PER_EVAL)?;

    trainer.train(&mut agent, &mut recorder, &mut evaluator)?;

    Ok(())
}

fn eval(render: bool) -> Result<()> {
    let model_dir = MODEL_DIR.to_owned() + "/best";
    let env_config = {
        let mut env_config = config::env_config();
        if render {
            env_config = env_config
                .render_mode(Some("human".to_string()))
                .set_wait_in_millis(10);
        }
        env_config
    };
    let mut agent = {
        let mut agent = Sac::build(config::agent_config(DIM_OBS, DIM_ACT));
        agent.load(model_dir)?;
        agent.eval();
        agent
    };

    let _ = Evaluator::new(&env_config, 0, 5)?.evaluate(&mut agent);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let matches = utils::create_matches();

    if matches.is_present("eval") {
        eval(true)?;
    } else if matches.is_present("train") {
        train(matches, MAX_OPTS)?;
    } else {
        train(matches, MAX_OPTS)?;
        eval(true)?;
    }

    Ok(())
}
