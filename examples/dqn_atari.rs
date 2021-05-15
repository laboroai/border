use anyhow::Result;
use clap::{App, Arg};
use std::{path::Path, time::Duration};

use border::{agent::tch::{DQNBuilder, dqn::DQNModelBuilder}, core::{record::TensorboardRecorder, util, Agent, TrainerBuilder}, env::py_gym_env::{
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
        framestack::FrameStackFilter,
        obs::PyGymEnvObs,
        tch::{act_d::TchPyGymEnvDiscreteActBuffer, obs::TchPyGymEnvObsBuffer},
        AtariWrapper, PyGymEnv, PyGymEnvBuilder, Shape,
    }, util::url::get_model_from_url};

const N_STACK: i64 = 4;
const DIM_OBS: [usize; 4] = [4, 1, 84, 84];

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &DIM_OBS
    }
}

type ObsFilter = FrameStackFilter<ObsShape, u8, u8>;
type ActFilter = PyGymEnvDiscreteActRawFilter;
type Obs = PyGymEnvObs<ObsShape, u8, u8>;
type Act = PyGymEnvDiscreteAct;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, u8, u8>;
type ActBuffer = TchPyGymEnvDiscreteActBuffer;

mod dqn_model {
    use border::agent::tch::{
        model::SubModel,
        util::OutDim,
    };
    use serde::{Deserialize, Serialize};
    use tch::{nn, nn::Module, Device, Tensor};

    #[allow(clippy::upper_case_acronyms)]
    #[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
    pub struct CNNConfig {
        n_stack: i64,
        out_dim: i64,
    }

    impl OutDim for CNNConfig {
        fn get_out_dim(&self) -> i64 {
            self.out_dim
        }

        fn set_out_dim(&mut self, v: i64) {
            self.out_dim = v;
        }
    }

    impl CNNConfig {
        pub fn new(n_stack: i64, out_dim: i64) -> Self {
            Self {
                n_stack,
                out_dim,
            }
        }
    }

    #[allow(clippy::upper_case_acronyms)]
    // Convolutional neural network
    pub struct CNN {
        n_stack: i64,
        out_dim: i64,
        device: Device,
        seq: nn::Sequential,
    }

    impl CNN {
        fn stride(s: i64) -> nn::ConvConfig {
            nn::ConvConfig {
                stride: s,
                ..Default::default()
            }
        }    
    
        fn create_net(
            var_store: &nn::VarStore,
            n_stack: i64,
            out_dim: i64
        ) -> nn::Sequential {
            let p = &var_store.root();
            nn::seq()
            .add_fn(|xs| xs.squeeze1(2).internal_cast_float(true) / 255)
            .add(nn::conv2d(p / "c1", n_stack, 32, 8, Self::stride(4)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c2", 32, 64, 4, Self::stride(2)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c3", 64, 64, 3, Self::stride(1)))
            .add_fn(|xs| xs.relu().flat_view())
            .add(nn::linear(p / "l1", 3136, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "l2", 512, out_dim as _, Default::default()))
        }
    }

    impl SubModel for CNN {
        type Config = CNNConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, x: &Self::Input) -> Tensor {
            self.seq.forward(&x.to(self.device))
        }

        fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
            let n_stack = config.n_stack;
            let out_dim = config.out_dim;
            let device = var_store.device();
            let seq = Self::create_net(var_store, n_stack, out_dim);

            Self {
                n_stack,
                out_dim,
                device,
                seq,
            }
        }

        fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
            let n_stack = self.n_stack;
            let out_dim = self.out_dim;
            let device = var_store.device();
            let seq = Self::create_net(&var_store, n_stack, out_dim);

            Self {
                n_stack,
                out_dim,
                device,
                seq,
            }
        }
    }
}

use dqn_model::CNN;

fn create_agent(
    dim_act: i64,
    env_name: impl Into<String>,
) -> Result<(impl Agent<Env>, DQNBuilder)> {
    let device = tch::Device::cuda_if_available();
    let env_name = env_name.into();
    let model_cfg = format!("./examples/model/dqn_{}/model.yaml", &env_name);
    let model_cfg = DQNModelBuilder::<CNN>::load(Path::new(&model_cfg))?;
    let qnet = model_cfg.out_dim(dim_act).build(device)?;
    let agent_cfg = format!("./examples/model/dqn_{}/agent.yaml", &env_name);
    let agent_cfg = DQNBuilder::load(Path::new(&agent_cfg))?;
    let agent = agent_cfg
        .clone()
        .build::<_, _, ObsBuffer, ActBuffer>(qnet, device);

    Ok((agent, agent_cfg))
}

fn create_env(name: &str, mode: AtariWrapper) -> Env {
    let obs_filter = ObsFilter::new(N_STACK as i64);
    let act_filter = ActFilter::default();
    PyGymEnvBuilder::default()
        .atari_wrapper(Some(mode))
        .build(name, obs_filter, act_filter)
        .unwrap()
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = App::new("dqn_atari")
        .version("0.1.0")
        .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
        .arg(
            Arg::with_name("name")
                .long("name")
                .takes_value(true)
                .required(true)
                .index(1)
                .help("The name of the atari environment (e.g., PongNoFrameskip-v4)"),
        )
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
            Arg::with_name("show-config")
                .long("show-config")
                .takes_value(false)
                .help("Showing configuration loaded from files"),
        )
        .get_matches();

    let name = matches.value_of("name").unwrap();
    let mut env_eval = create_env(name, AtariWrapper::Eval);
    let dim_act = env_eval.get_num_actions_atari();
    let agent = create_agent(dim_act as _, name)?;
    let agent_cfg = agent.1;
    let mut agent = agent.0;

    if !(matches.is_present("play") || matches.is_present("play-gdrive")) {
        let env_train = create_env(name, AtariWrapper::Train);
        let saving_model_dir = format!("./examples/model/dqn_{}", name);
        let trainer_cfg = Path::new(&saving_model_dir).join("trainer.yaml");
        let trainer_cfg = TrainerBuilder::load(&trainer_cfg)?;
        let mut trainer = trainer_cfg.clone().build(env_train, env_eval, agent);
        let mut recorder = TensorboardRecorder::new(format!("./examples/model/dqn_{}", name));

        if matches.is_present("show-config") {
            println!("Device: {:?}", tch::Device::cuda_if_available());
            println!("{:?}", trainer_cfg);
            println!("{:?}", agent_cfg);
            return Ok(());
        }

        trainer.train(&mut recorder);
    } else {
        if matches.is_present("play") {
            let model_dir = matches
                .value_of("play")
                .expect("Failed to parse model directory");
            agent.load(model_dir).unwrap(); // TODO: define appropriate error
        } else {
            // TODO: change file_base and url depending on the game
            let file_base = "dqn_PongNoFrameskip-v4_20210428_ec2";
            let url =
                "https://drive.google.com/uc?export=download&id=1TF5aN9fH5wd4APFHj9RP1JxuVNoi6lqJ";
            let model_dir = get_model_from_url(url, file_base)?;
            agent.load(model_dir).unwrap(); // TODO: define appropriate error
        };

        let time = matches.value_of("wait").unwrap().parse::<u64>()?;
        env_eval.set_render(true);
        env_eval.set_wait_in_render(Duration::from_millis(time));
        agent.eval();
        util::eval(&mut env_eval, &mut agent, 5);
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use std::{path::Path, default::Default};
    use anyhow::Result;
    use border::{agent::{tch::opt::OptimizerConfig, OptInterval, tch::dqn::{DQNBuilder, DQNModelBuilder, explorer::EpsilonGreedy}}, core::TrainerBuilder};
    use super::{N_STACK, dqn_model::{CNN, CNNConfig}};

    // DQN agent parameters
    const DISCOUNT_FACTOR: f64 = 0.99;
    const BATCH_SIZE: usize = 32;
    const N_TRANSITIONS_WARMUP: usize = 2500;
    const N_UPDATES_PER_OPT: usize = 1;
    const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
    const SOFT_UPDATE_INTERVAL: usize = 10_000;
    const TAU: f64 = 1.0;
    const EPS_FINAL_STEP: usize = 1_000_000;
    const REPLAY_BUFFER_CAPACITY: usize = 50_000;

    // DQN model parameters
    const LR_QNET: f64 = 1e-4;

    // Training parameters
    const MAX_OPTS: usize = 3_000_000;
    const EVAL_INTERVAL: usize = 10_000;
    const N_EPISODES_PER_EVAL: usize = 1;

    #[test]
    fn save_configs() -> Result<()> {
        let env_name = "PongNoFrameskip-v4";
        let saving_model_dir = format!("./examples/model/dqn_{}", env_name);
        let model_cfg = Path::new(&saving_model_dir).join("model.yaml");
        let agent_cfg = Path::new(&saving_model_dir).join("agent.yaml");
        let trainer_cfg = Path::new(&saving_model_dir).join("trainer.yaml");
        println!("{:?}", agent_cfg);

        let out_dim = 0; // set in training/evaluation code
        let builder = DQNModelBuilder::<CNN>::default()
            .opt_config(OptimizerConfig::Adam { lr: LR_QNET })
            .q_config(CNNConfig::new(N_STACK, out_dim));
        let _ = builder.save(model_cfg);

        let builder = DQNBuilder::default()
            .opt_interval(OPT_INTERVAL)
            .n_updates_per_opt(N_UPDATES_PER_OPT)
            .min_transitions_warmup(N_TRANSITIONS_WARMUP)
            .batch_size(BATCH_SIZE)
            .discount_factor(DISCOUNT_FACTOR)
            .soft_update_interval(SOFT_UPDATE_INTERVAL)
            .tau(TAU)
            .replay_burffer_capacity(REPLAY_BUFFER_CAPACITY)
            .explorer(EpsilonGreedy::with_final_step(EPS_FINAL_STEP));
        let _ = builder.save(agent_cfg);

        let builder = TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .model_dir(saving_model_dir);
        let _ = builder.save(trainer_cfg);
    Ok(())
    }
}
