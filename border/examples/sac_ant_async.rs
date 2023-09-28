use anyhow::Result;
use border_async_trainer::{
    actor_stats_fmt, ActorManager as ActorManager_, ActorManagerConfig,
    AsyncTrainer as AsyncTrainer_, AsyncTrainerConfig,
};
use border_core::{
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    DefaultEvaluator,
};
use border_derive::SubBatch;
use border_py_gym_env::{
    util::{arrayd_to_tensor, tensor_to_arrayd},
    ArrayObsFilter, ContinuousActFilter, GymActFilter, GymEnv, GymEnvConfig,
    GymObsFilter,
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
use crossbeam_channel::unbounded;
use ndarray::{ArrayD, IxDyn};
use std::{
    convert::TryFrom,
    default::Default,
    sync::{Arc, Mutex},
};
use tch::Tensor;

type PyObsDtype = f32;

const DIM_OBS: i64 = 27;
const DIM_ACT: i64 = 8;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const BATCH_SIZE: usize = 256;
const N_TRANSITIONS_WARMUP: usize = 10_000;
const MAX_OPTS: usize = 3_000_000;
const EVAL_INTERVAL: usize = 10_000;
const RECORD_INTERVAL: usize = 10_000;
const SAVE_INTERVAL: usize = 500_000;
const REPLAY_BUFFER_CAPACITY: usize = 300_000;
const SYNC_INTERVAL: usize = 100;
const EVAL_EPISODES: usize = 5;
const N_CRITICS: usize = 2;
const TAU: f64 = 0.02;
const TARGET_ENTROPY: f64 = -(DIM_ACT as f64);
const LR_ENT_COEF: f64 = 3e-4;
const CRITIC_LOSS: CriticLoss = CriticLoss::SmoothL1;
const MODEL_DIR: &str = "./border/examples/model/sac_ant_async";

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
type EnvConfig = GymEnvConfig<Obs, Act, ObsFilter, ActFilter>;
type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
type Agent = Sac<Env, Mlp, Mlp2, ReplayBuffer>;
type AgentConfig = SacConfig<Mlp, Mlp2>;
type ActorManager = ActorManager_<Agent, Env, ReplayBuffer, StepProc>;
type AsyncTrainer = AsyncTrainer_<Agent, Env, ReplayBuffer>;
type Evaluator = DefaultEvaluator<Env, Agent>;

mod config {
    use super::*;

    pub fn env_config() -> GymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
        GymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
            .name("Ant-v4".to_string())
            .obs_filter_config(ObsFilter::default_config())
            .act_filter_config(ActFilter::default_config())
    }

    pub fn agent_config(in_dim: i64, out_dim: i64) -> AgentConfig {
        let actor_config = ActorConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_ACTOR })
            .out_dim(out_dim)
            .pi_config(MlpConfig::new(in_dim, vec![400, 300], out_dim, true));
        let critic_config = CriticConfig::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_CRITIC })
            .q_config(MlpConfig::new(in_dim + out_dim, vec![400, 300], 1, true));

        SacConfig::default()
            .batch_size(BATCH_SIZE)
            .min_transitions_warmup(N_TRANSITIONS_WARMUP)
            .actor_config(actor_config)
            .critic_config(critic_config)
            .tau(TAU)
            .critic_loss(CRITIC_LOSS)
            .n_critics(N_CRITICS)
            .ent_coef_mode(EntCoefMode::Auto(TARGET_ENTROPY, LR_ENT_COEF))
    }

    pub fn async_trainer_config() -> AsyncTrainerConfig {
        AsyncTrainerConfig {
            model_dir: Some(MODEL_DIR.to_string()),
            record_interval: RECORD_INTERVAL,
            eval_interval: EVAL_INTERVAL,
            max_train_steps: MAX_OPTS,
            save_interval: SAVE_INTERVAL,
            sync_interval: SYNC_INTERVAL,
        }
    }
}

use config::{agent_config, async_trainer_config, env_config};

fn parse_args<'a>() -> ArgMatches<'a> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = App::new("dqn_atari_async")
        .version("0.1.0")
        .author("Taku Yoshioka <yoshioka@laboro.ai>")
        .arg(
            Arg::with_name("show-config")
                .long("show-config")
                .takes_value(false)
                .help("Showing configuration loaded from files"),
        )
        .arg(
            Arg::with_name("n-actors")
                .long("n-actors")
                .takes_value(true)
                .default_value("6")
                .help("The number of actors"),
        )
        .get_matches();

    matches
}

fn show_config(
    env_config: &EnvConfig,
    agent_config: &AgentConfig,
    actor_man_config: &ActorManagerConfig,
    trainer_config: &AsyncTrainerConfig,
) {
    println!("Device: {:?}", tch::Device::cuda_if_available());
    println!("{}", serde_yaml::to_string(&env_config).unwrap());
    println!("{}", serde_yaml::to_string(&agent_config).unwrap());
    println!("{}", serde_yaml::to_string(&actor_man_config).unwrap());
    println!("{}", serde_yaml::to_string(&trainer_config).unwrap());
}

fn train(matches: ArgMatches) -> Result<()> {
    // exploration parameters
    let n_actors = matches
        .value_of("n-actors")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let env_config_train = env_config();
    let env_config_eval = env_config();
    let agent_configs = (0..n_actors)
        .map(|ix| {
            agent_config(DIM_OBS, DIM_ACT)
                .seed(ix as i64)
                .device(tch::Device::Cpu)
        })
        .collect::<Vec<_>>();
    let agent_config = agent_config(DIM_OBS, DIM_ACT).device(tch::Device::cuda_if_available());
    let replay_buffer_config = SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
    let step_proc_config = SimpleStepProcessorConfig::default();
    let actor_man_config = ActorManagerConfig::default();
    let async_trainer_config = async_trainer_config();

    if matches.is_present("show-config") {
        show_config(
            &env_config_train,
            &agent_config,
            &actor_man_config,
            &async_trainer_config,
        );
    } else {
        let mut recorder = TensorboardRecorder::new(MODEL_DIR);
        let mut evaluator = Evaluator::new(&env_config_eval, 0, EVAL_EPISODES)?;

        // Shared flag to stop actor threads
        let stop = Arc::new(Mutex::new(false));

        // Creates channels
        let (item_s, item_r) = unbounded(); // items pushed to replay buffer
        let (model_s, model_r) = unbounded(); // model_info

        // guard for initialization of envs in multiple threads
        let guard_init_env = Arc::new(Mutex::new(true));

        // Actor manager and async trainer
        let mut actors = ActorManager::build(
            &actor_man_config,
            &agent_configs,
            &env_config_train,
            &step_proc_config,
            item_s,
            model_r,
            stop.clone(),
        );
        let mut trainer = AsyncTrainer::build(
            &async_trainer_config,
            &agent_config,
            &env_config_eval,
            &replay_buffer_config,
            item_r,
            model_s,
            stop.clone(),
        );

        // Set the number of threads
        tch::set_num_threads(1);

        // Starts sampling and training
        actors.run(guard_init_env.clone());
        let stats = trainer.train(&mut recorder, &mut evaluator, guard_init_env);
        println!("Stats of async trainer");
        println!("{}", stats.fmt());

        let stats = actors.stop_and_join();
        println!("Stats of generated samples in actors");
        println!("{}", actor_stats_fmt(&stats));
    }

    Ok(())
}

fn main() -> Result<()> {
    let matches = parse_args();

    train(matches)?;

    Ok(())
}
