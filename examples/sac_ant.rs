use anyhow::Result;
use border::{
    agent::{
        tch::{
            model::{SubModel, SubModel2},
            opt::OptimizerConfig,
            sac::{Actor, ActorBuilder, Critic, CriticBuilder, EntCoefMode},
            util::OutDim,
            ReplayBuffer, SACBuilder,
        },
        CriticLoss, OptInterval,
    },
    env::py_gym_env::{
        act_c::{PyGymEnvContinuousAct, PyGymEnvContinuousActRawFilter},
        obs::{PyGymEnvObs, PyGymEnvObsRawFilter},
        tch::{act_c::TchPyGymEnvContinuousActBuffer, obs::TchPyGymEnvObsBuffer},
        PyGymEnv, PyGymEnvBuilder, Shape,
    },
    util::url::get_model_from_url,
};
use border_core::{record::TensorboardRecorder, util, Agent, TrainerBuilder};
use clap::{App, Arg};
use log::info;
use serde::{Deserialize, Serialize};
use std::{default::Default, time::Duration};
use tch::{
    nn::{self, Module},
    Device, Tensor,
};

const OBS_DIM: usize = 28;
const ACT_DIM: usize = 8;
const LR_ACTOR: f64 = 3e-4;
const LR_CRITIC: f64 = 3e-4;
const N_CRITICS: usize = 2;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 256;
const N_TRANSITIONS_WARMUP: usize = 10_000;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.02;
// const ALPHA: f64 = 1.0;
const TARGET_ENTROPY: f64 = -(ACT_DIM as f64);
const LR_ENT_COEF: f64 = 3e-4;
const REWARD_SCALE: f32 = 1.0;
const CRITIC_LOSS: CriticLoss = CriticLoss::SmoothL1;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const MAX_OPTS: usize = 3_000_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 300_000;
const N_EPISODES_PER_EVAL: usize = 5;
// const MAX_STEPS_IN_EPISODE: usize = 200;

#[derive(Debug, Clone)]
struct ObsShape {}

impl Shape for ObsShape {
    fn shape() -> &'static [usize] {
        &[OBS_DIM]
    }
}

#[derive(Debug, Clone)]
struct ActShape {}

impl Shape for ActShape {
    fn shape() -> &'static [usize] {
        &[ACT_DIM]
    }

    fn squeeze_first_dim() -> bool {
        true
    }
}

#[allow(clippy::clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
struct MLPConfig {
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
struct MLP {
    in_dim: i64,
    units: Vec<i64>,
    out_dim: i64,
    device: Device,
    seq: nn::Sequential,
}

fn mlp(var_store: &nn::VarStore, config: &MLPConfig) -> nn::Sequential {
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
struct MLP2 {
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

fn create_actor(device: Device) -> Result<Actor<MLP2>> {
    let in_dim = 0;
    let out_dim = 0;
    let lr_actor = LR_ACTOR;
    ActorBuilder::default()
        .pi_config(MLPConfig::new(in_dim, vec![400, 300], out_dim))
        .opt_config(OptimizerConfig::Adam { lr: lr_actor })
        .build(device)
}

fn create_critic(device: Device) -> Result<Critic<MLP>> {
    let in_dim = 0;
    let out_dim = 1;
    let lr_critic = LR_CRITIC;
    CriticBuilder::default()
        .q_config(MLPConfig::new(in_dim, vec![400, 300], out_dim))
        .opt_config(OptimizerConfig::Adam { lr: lr_critic })
        .build(device)
}

// fn create_actor(device: tch::Device) -> Model1_2 {
//     let network_fn = |p: &nn::Path, in_dim, hidden_dim| {
//         nn::seq()
//             .add(nn::linear(p / "al1", in_dim as _, 400, Default::default()))
//             .add_fn(|xs| xs.relu())
//             .add(nn::linear(
//                 p / "al2",
//                 400,
//                 hidden_dim as _,
//                 Default::default(),
//             ))
//             .add_fn(|xs| xs.relu())
//     };
//     Model1_2::new(DIM_OBS, 300, DIM_ACT, LR_ACTOR, network_fn, device)
// }

// fn create_critic(device: tch::Device) -> Model2_1 {
//     let network_fn = |p: &nn::Path, in_dim, out_dim| {
//         nn::seq()
//             .add(nn::linear(p / "cl1", in_dim as _, 400, Default::default()))
//             .add_fn(|xs| xs.relu())
//             .add(nn::linear(p / "cl2", 400, 300, Default::default()))
//             .add_fn(|xs| xs.relu())
//             .add(nn::linear(p / "cl3", 300, out_dim as _, Default::default()))
//     };
//     Model2_1::new(DIM_OBS + DIM_ACT, 1, LR_CRITIC, network_fn, device)
// }

type ObsFilter = PyGymEnvObsRawFilter<ObsShape, f32, f32>;
type ActFilter = PyGymEnvContinuousActRawFilter;
type Obs = PyGymEnvObs<ObsShape, f32, f32>;
type Act = PyGymEnvContinuousAct<ActShape>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, f32, f32>;
type ActBuffer = TchPyGymEnvContinuousActBuffer<ActShape>;

fn create_agent() -> Result<impl Agent<Env>> {
    let device = tch::Device::cuda_if_available();
    let actor = create_actor(device)?;
    let critics = (0..N_CRITICS)
        .map(|_| create_critic(device).expect("Cannot create critic"))
        .collect::<Vec<_>>();
    let replay_buffer = ReplayBuffer::<Env, ObsBuffer, ActBuffer>::new(REPLAY_BUFFER_CAPACITY, 1);

    Ok(SACBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .tau(TAU)
        .ent_coef_mode(EntCoefMode::Auto(TARGET_ENTROPY, LR_ENT_COEF))
        .reward_scale(REWARD_SCALE)
        .critic_loss(CRITIC_LOSS)
        .build(critics, actor, replay_buffer, device))
}

fn create_env() -> Env {
    let obs_filter = ObsFilter::default();
    let act_filter = ActFilter::default();
    PyGymEnvBuilder::default()
        .pybullet(true)
        .atari_wrapper(None)
        .build("AntPyBulletEnv-v0", obs_filter, act_filter)
        .unwrap()
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);
    fastrand::seed(42);

    let matches = App::new("dqn_cartpole")
        .version("0.1.0")
        .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
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
        .get_matches();

    if !(matches.is_present("play") || matches.is_present("play-gdrive")) {
        let env = create_env();
        let env_eval = create_env();
        let agent = create_agent()?;
        let mut trainer = TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .build(env, env_eval, agent);
        let mut recorder = TensorboardRecorder::new("./examples/model/sac_ant");

        trainer.train(&mut recorder);
        trainer
            .get_agent()
            .save("./examples/model/sac_ant")
            .unwrap();
    } else {
        let mut env = create_env();
        let mut agent = create_agent()?;

        if matches.is_present("play") {
            let model_dir = matches
                .value_of("play")
                .expect("Failed to parse model directory");
            agent.load(model_dir).unwrap(); // TODO: define appropriate error
        } else {
            let file_base = "sac_ant_20210324_ec2_smoothl1";
            let url =
                "https://drive.google.com/uc?export=download&id=1XvFi2nJD5OhpTvs-Et3YREuoqy8c3Vkq";
            let model_dir = get_model_from_url(url, file_base)?;
            info!("Download the model in {:?}", model_dir.as_ref().to_str());
            agent.load(model_dir).unwrap(); // TODO: define appropriate error
        };

        let time = matches.value_of("wait").unwrap().parse::<u64>()?;
        env.set_render(true);
        env.set_wait_in_render(Duration::from_millis(time));
        agent.eval();
        util::eval(&mut env, &mut agent, 5);
    }

    Ok(())
}
