use anyhow::Result;
use clap::{App, Arg};
use std::{path::Path, time::Duration};
use tch::nn;

use border::{
    agent::tch::{model::Model1_1, DQNBuilder},
    core::{record::TensorboardRecorder, util, Agent, TrainerBuilder},
    env::py_gym_env::{
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
        framestack::FrameStackFilter,
        obs::PyGymEnvObs,
        tch::{act_d::TchPyGymEnvDiscreteActBuffer, obs::TchPyGymEnvObsBuffer},
        AtariWrapper, PyGymEnv, PyGymEnvBuilder, Shape,
    },
    util::url::get_model_from_url,
};

// const N_PROCS: usize = 1;
const N_STACK: usize = 4;
const DIM_OBS: [usize; 4] = [4, 1, 84, 84];
const LR_QNET: f64 = 1e-4;
// const DISCOUNT_FACTOR: f64 = 0.99;
// const BATCH_SIZE: usize = 32;
// const N_TRANSITIONS_WARMUP: usize = 2500;
// const N_UPDATES_PER_OPT: usize = 1;
// const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
// const SOFT_UPDATE_INTERVAL: usize = 10_000;
// const TAU: f64 = 1.0;
// const MAX_OPTS: usize = 3_000_000;
// const EVAL_INTERVAL: usize = 10_000;
// const REPLAY_BUFFER_CAPACITY: usize = 50_000;
// const N_EPISODES_PER_EVAL: usize = 1;
// const EPS_FINAL_STEP: usize = 1_000_000;

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
// type ReplayBuffer = ReplayBuffer_<Env, ObsBuffer, ActBuffer>;

fn stride(s: i64) -> nn::ConvConfig {
    nn::ConvConfig {
        stride: s,
        ..Default::default()
    }
}

fn create_critic(dim_act: usize, device: tch::Device) -> Model1_1 {
    let network_fn = |p: &nn::Path, _in_shape: &[usize], out_dim| {
        nn::seq()
            .add_fn(|xs| xs.squeeze1(2).internal_cast_float(true) / 255)
            .add(nn::conv2d(p / "c1", N_STACK as i64, 32, 8, stride(4)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c2", 32, 64, 4, stride(2)))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(p / "c3", 64, 64, 3, stride(1)))
            .add_fn(|xs| xs.relu().flat_view())
            .add(nn::linear(p / "l1", 3136, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "l2", 512, out_dim as _, Default::default()))
    };
    Model1_1::new(&DIM_OBS, dim_act, LR_QNET, network_fn, device)
}

fn create_agent(
    dim_act: usize,
    env_name: impl Into<String>,
) -> Result<(impl Agent<Env>, DQNBuilder)> {
    let device = tch::Device::cuda_if_available();
    let qnet = create_critic(dim_act, device);
    let agent_cfg = format!("./examples/model/dqn_{}/agent.yaml", env_name.into());

    let agent_cfg = DQNBuilder::load(Path::new(&agent_cfg))?;
    // .opt_interval(OPT_INTERVAL)
    // .n_updates_per_opt(N_UPDATES_PER_OPT)
    // .min_transitions_warmup(N_TRANSITIONS_WARMUP)
    // .batch_size(BATCH_SIZE)
    // .discount_factor(DISCOUNT_FACTOR)
    // .soft_update_interval(SOFT_UPDATE_INTERVAL)
    // .tau(TAU)
    // .explorer(EpsilonGreedy::with_final_step(EPS_FINAL_STEP))
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
        // let mut trainer = TrainerBuilder::default()
        //     .max_opts(MAX_OPTS)
        //     .eval_interval(EVAL_INTERVAL)
        //     .n_episodes_per_eval(N_EPISODES_PER_EVAL)
        //     .model_dir(saving_model_dir)
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

// #[cfg(test)]
// mod test {
//     use std::path::Path;
//     use anyhow::Result;
//     use super::{DQNBuilder, OptInterval, EpsilonGreedy, TrainerBuilder};

//     // DQN agent parameters
//     const DISCOUNT_FACTOR: f64 = 0.99;
//     const BATCH_SIZE: usize = 32;
//     const N_TRANSITIONS_WARMUP: usize = 2500;
//     const N_UPDATES_PER_OPT: usize = 1;
//     const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
//     const SOFT_UPDATE_INTERVAL: usize = 10_000;
//     const TAU: f64 = 1.0;

//     // Training parameters
//     const MAX_OPTS: usize = 3_000_000;
//     const EVAL_INTERVAL: usize = 10_000;
//     const REPLAY_BUFFER_CAPACITY: usize = 50_000;
//     const N_EPISODES_PER_EVAL: usize = 1;
//     const EPS_FINAL_STEP: usize = 1_000_000;

//     #[test]
//     fn save_configs() -> Result<()> {
//         let env_name = "PongNoFrameskip-v4";
//         let saving_model_dir = format!("./examples/model/dqn_{}", env_name);
//         let agent_cfg = Path::new(&saving_model_dir).join("agent.yaml");
//         let trainer_cfg = Path::new(&saving_model_dir).join("trainer.yaml");
//         println!("{:?}", agent_cfg);

//         let builder = DQNBuilder::default()
//             .opt_interval(OPT_INTERVAL)
//             .n_updates_per_opt(N_UPDATES_PER_OPT)
//             .min_transitions_warmup(N_TRANSITIONS_WARMUP)
//             .batch_size(BATCH_SIZE)
//             .discount_factor(DISCOUNT_FACTOR)
//             .soft_update_interval(SOFT_UPDATE_INTERVAL)
//             .tau(TAU)
//             .explorer(EpsilonGreedy::with_final_step(EPS_FINAL_STEP));
//         builder.save(agent_cfg);

//         let builder = TrainerBuilder::default()
//             .max_opts(MAX_OPTS)
//             .eval_interval(EVAL_INTERVAL)
//             .n_episodes_per_eval(N_EPISODES_PER_EVAL)
//             .model_dir(saving_model_dir);
//         builder.save(trainer_cfg);
//     Ok(())
//     }
// }
