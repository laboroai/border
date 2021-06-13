mod dqn_atari_model;
use anyhow::Result;
use border::{try_from, util::url::get_model_from_url};
use border_core::{
    record::TensorboardRecorder,
    shape, util, Agent, TrainerBuilder,
};
use border_py_gym_env::{
    newtype_act_d, newtype_obs, AtariWrapper, FrameStackFilter, PyGymEnv, PyGymEnvBuilder,
    PyGymEnvDiscreteAct, PyGymEnvObs,
};
use border_tch_agent::{
    dqn::{DQNBuilder, DQNModelBuilder},
    replay_buffer::TchTensorBuffer,
};
use clap::{App, Arg};
use dqn_atari_model::CNN;
use ndarray::ArrayD;
use std::{convert::TryFrom, path::Path, time::Duration};
use tch::Tensor;

const N_STACK: i64 = 4;
shape!(ObsShape, [N_STACK as usize, 1, 84, 84]);
shape!(ActShape, [1]);
newtype_obs!(Obs, ObsShape, u8, u8);
newtype_act_d!(Act, ActFilter);

impl From<Obs> for Tensor {
    fn from(obs: Obs) -> Tensor {
        try_from(obs.0.obs).unwrap()
    }
}

impl From<Act> for Tensor {
    fn from(act: Act) -> Tensor {
        let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
        let t: Tensor = TryFrom::<Vec<i64>>::try_from(v).unwrap();

        // The first dimension of the action tensor is the number of processes,
        // which is 1 for the non-vectorized environment.
        t.unsqueeze(0)
    }
}

/// Converts Tensor to Act, called when applying actions estimated by DQN.
/// DQN outputs Tensor, while PyGymEnv accepts Act as actions to the environment.
impl From<Tensor> for Act {
    /// `t` must be a 1-dimentional tensor of `f32`.
    fn from(t: Tensor) -> Self {
        let data: Vec<i64> = t.into();
        let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
        Act(PyGymEnvDiscreteAct::new(data))
    }
}

/// This implementation is required by FrameStackFilter.
impl From<ArrayD<u8>> for Obs {
    fn from(obs: ArrayD<u8>) -> Self {
        Obs(PyGymEnvObs::from(obs))
    }
}

type ObsFilter = FrameStackFilter<Obs, u8, u8>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchTensorBuffer<u8, ObsShape, Obs>;
type ActBuffer = TchTensorBuffer<i64, ActShape, Act>;

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
    use super::{
        dqn_atari_model::{CNNConfig, CNN},
        N_STACK,
    };
    use anyhow::Result;
    use border_core::TrainerBuilder;
    use border_tch_agent::{
        dqn::{explorer::EpsilonGreedy, DQNBuilder, DQNModelBuilder},
        opt::OptimizerConfig,
        util::OptInterval,
    };
    use std::{default::Default, path::Path};

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
