use anyhow::{Context, Result};
// use border::{
//     agent::{
//         tch::{
//             dqn::explorer::EpsilonGreedy, model::Model1_1, DQNBuilder,
//             ReplayBuffer as ReplayBuffer_,
//         },
//         OptInterval,
//     },
//     env::py_gym_env::{
//         act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
//         framestack::FrameStackFilter,
//         obs::PyGymEnvObs,
//         tch::{act_d::TchPyGymEnvDiscreteActBuffer, obs::TchPyGymEnvObsBuffer},
//         AtariWrapper, PyVecGymEnv, PyVecGymEnvBuilder, Shape,
//     },
// };
// use border_core::{record::TensorboardRecorder, Agent, TrainerBuilder};
// use tch::nn;

// const N_PROCS: usize = 4;
// const N_STACK: usize = 4;
// const DIM_OBS: [usize; 4] = [4, 1, 84, 84];
// const LR_QNET: f64 = 1e-4;
// const DISCOUNT_FACTOR: f64 = 0.99;
// const BATCH_SIZE: usize = 32;
// const N_TRANSITIONS_WARMUP: usize = 2500;
// const N_UPDATES_PER_OPT: usize = 1;
// const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
// const SOFT_UPDATE_INTERVAL: usize = 10_000;
// const TAU: f64 = 1.0;
// const MAX_OPTS: usize = 5;
// const EVAL_INTERVAL: usize = 10_000;
// const REPLAY_BUFFER_CAPACITY: usize = 1_000;
// const N_EPISODES_PER_EVAL: usize = 1;
// const EPS_FINAL_STEP: usize = 1_000_000;

// #[derive(Debug, Clone)]
// struct ObsShape {}

// impl Shape for ObsShape {
//     fn shape() -> &'static [usize] {
//         &DIM_OBS
//     }
// }

// type ObsFilter = FrameStackFilter<ObsShape, u8, u8>;
// type ActFilter = PyGymEnvDiscreteActRawFilter;
// type Obs = PyGymEnvObs<ObsShape, u8, u8>;
// type Act = PyGymEnvDiscreteAct;
// type Env = PyVecGymEnv<Obs, Act, ObsFilter, ActFilter>;
// type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, u8, u8>;
// type ActBuffer = TchPyGymEnvDiscreteActBuffer;
// type ReplayBuffer = ReplayBuffer_<Env, ObsBuffer, ActBuffer>;

// fn stride(s: i64) -> nn::ConvConfig {
//     nn::ConvConfig {
//         stride: s,
//         ..Default::default()
//     }
// }

// fn create_critic(dim_act: usize, device: tch::Device) -> Model1_1 {
//     let network_fn = |p: &nn::Path, _in_shape: &[usize], out_dim| {
//         nn::seq()
//             .add_fn(|xs| xs.squeeze1(2).internal_cast_float(true) / 255)
//             .add(nn::conv2d(p / "c1", N_STACK as i64, 32, 8, stride(4)))
//             .add_fn(|xs| xs.relu())
//             .add(nn::conv2d(p / "c2", 32, 64, 4, stride(2)))
//             .add_fn(|xs| xs.relu())
//             .add(nn::conv2d(p / "c3", 64, 64, 3, stride(1)))
//             .add_fn(|xs| xs.relu().flat_view())
//             .add(nn::linear(p / "l1", 3136, 512, Default::default()))
//             .add_fn(|xs| xs.relu())
//             .add(nn::linear(p / "l2", 512, out_dim as _, Default::default()))
//     };
//     Model1_1::new(&DIM_OBS, dim_act, LR_QNET, network_fn, device)
// }

// fn create_agent(dim_act: usize) -> impl Agent<Env> {
//     let device = tch::Device::cuda_if_available();
//     let qnet = create_critic(dim_act, device);
//     let replay_buffer = ReplayBuffer::new(REPLAY_BUFFER_CAPACITY, N_PROCS);

//     DQNBuilder::default()
//         .opt_interval(OPT_INTERVAL)
//         .n_updates_per_opt(N_UPDATES_PER_OPT)
//         .min_transitions_warmup(N_TRANSITIONS_WARMUP)
//         .batch_size(BATCH_SIZE)
//         .discount_factor(DISCOUNT_FACTOR)
//         .soft_update_interval(SOFT_UPDATE_INTERVAL)
//         .tau(TAU)
//         .explorer(EpsilonGreedy::with_final_step(EPS_FINAL_STEP))
//         .build_with_replay_buffer(qnet, replay_buffer, device)
// }

// fn create_env(name: &str, mode: AtariWrapper) -> Result<Env> {
//     let obs_filter = ObsFilter::vectorized(N_PROCS as _, N_STACK as _);
//     let act_filter = ActFilter::vectorized();
//     PyVecGymEnvBuilder::default()
//         .atari_wrapper(Some(mode))
//         .n_procs(N_PROCS)
//         .build(name, obs_filter, act_filter)
//         .context("Failed to create vecenv")
// }

fn main() -> Result<()> {
    // env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // tch::manual_seed(42);

    // let name = "PongNoFrameskip-v4";
    // let env_eval = create_env(name, AtariWrapper::Eval)?;
    // let dim_act = env_eval.get_num_actions_atari();
    // let agent = create_agent(dim_act as _);

    // let env_train = create_env(name, AtariWrapper::Train)?;
    // let saving_model_dir = format!("./examples/model/dqn_{}_vec", name);

    // let mut trainer = TrainerBuilder::default()
    //     .max_opts(MAX_OPTS)
    //     .eval_interval(EVAL_INTERVAL)
    //     .n_episodes_per_eval(N_EPISODES_PER_EVAL)
    //     .model_dir(saving_model_dir)
    //     .build(env_train, env_eval, agent);
    // let mut recorder = TensorboardRecorder::new(format!("./examples/model/dqn_{}_vec", name));
    // trainer.train(&mut recorder);

    Ok(())
}
