use std::time::Duration;
use clap::{Arg, App};

use border::{
    core::{
        Agent, TrainerBuilder, util,
        record::TensorboardRecorder,
    },
    env::py_gym_env::{
        Shape, PyGymEnv, PyGymEnvBuilder,
        obs::PyGymEnvObs,
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
        framestack::FrameStackFilter,
        tch::{
            obs::TchPyGymEnvObsBuffer,
            act_d::TchPyGymEnvDiscreteActBuffer
        }
    },
    agent::{
        OptInterval,
        tch::{
            ReplayBuffer as ReplayBuffer_, IQNBuilder, EpsilonGreedy,
        }
    }
};

const N_PROCS: usize = 1;
const N_STACK: usize = 4;
const DIM_OBS: [usize; 4] = [4, 1, 84, 84];
const DIM_FEATURE: i64 = 16;
const DIM_EMBED: i64 = 64;
const LR_CRITIC: f64 = 0.001;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 32;
const N_TRANSITIONS_WARMUP: usize = 2500;
const N_UPDATES_PER_OPT: usize = 1;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const SOFT_UPDATE_INTERVAL: usize = 10_000;
const TAU: f64 = 1.0;
const MAX_OPTS: usize = 3_000_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 50_000;
const N_EPISODES_PER_EVAL: usize = 1;
const EPS_GREEDY_FINAL_STEP: usize = 1_000_000;

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
type ReplayBuffer = ReplayBuffer_<Env, ObsBuffer, ActBuffer>;

mod iqn_model {
    use tch::{Tensor, Device, nn, nn::Module};
    use border::agent::tch::{
        util::{FeatureExtractor, FeatureExtractorBuilder},
        IQNModel, IQNModelBuilder
    };

    #[allow(clippy::upper_case_acronyms)]
    #[derive(Debug)]
    // ConvNet as feature extractor
    pub struct ConvNet {
        n_stack: i64,
        feature_dim: i64,
        device: Device,
        seq: nn::Sequential
    }

    fn stride(s: i64) -> nn::ConvConfig {
        nn::ConvConfig {
            stride: s,
            ..Default::default()
        }
    }

    impl ConvNet {
        fn conv_net(n_stack: i64, feature_dim: i64, p: &nn::Path) -> nn::Sequential {
            nn::seq()
                .add_fn(|xs| xs.squeeze1(2).internal_cast_float(true))
                .add(nn::conv2d(p / "c1", n_stack, 32, 8, stride(4)))
                .add_fn(|xs| xs.relu())
                .add(nn::conv2d(p / "c2", 32, 64, 4, stride(2)))
                .add_fn(|xs| xs.relu())
                .add(nn::conv2d(p / "c3", 64, 64, 3, stride(1)))
                .add_fn(|xs| xs.relu().flat_view())
                .add(nn::linear(p / "l1", 3136, 512, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(p / "l2", 512, feature_dim, Default::default()))
        }
    }

    impl FeatureExtractor for ConvNet {
        type Input = Tensor;

        fn feature(&self, x: &Self::Input) -> Tensor {
            self.seq.forward(&x.to(self.device))
        }

        fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
            let p = &var_store.root();
            Self {
                n_stack: self.n_stack,
                feature_dim: self.feature_dim,
                device: var_store.device(),
                seq: Self::conv_net(self.n_stack, self.feature_dim, p)
            }
        }
    }

    #[allow(clippy::upper_case_acronyms)]
    // Builder of feature extractor
    struct ConvNetBuilder {
        n_stack: i64,
        feature_dim: i64,
    }

    impl ConvNetBuilder {
        pub fn new(n_stack: i64, feature_dim: i64) -> Self {
            Self {
                n_stack,
                feature_dim,
            }
        }
    }

    #[allow(clippy::upper_case_acronyms)]
    impl FeatureExtractorBuilder for ConvNetBuilder {
        type F = ConvNet;

        fn build(self, p: &nn::Path) -> Self::F {
            Self::F {
                n_stack: self.n_stack,
                feature_dim: self.feature_dim,
                device: p.device(),
                seq: ConvNet::conv_net(self.n_stack, self.feature_dim, p)
            }
        }
    }

    // IQN model
    pub fn create_iqn_model(n_stack: i64, feature_dim: i64, embed_dim: i64,  out_dim: i64,
        learning_rate: f64, device: Device) -> IQNModel<ConvNet> {
        let builder = ConvNetBuilder::new(n_stack, feature_dim);
        IQNModelBuilder::default().build(
            builder, feature_dim, embed_dim, out_dim, learning_rate, device
        )
    }
}

fn create_agent(dim_act: i64) -> impl Agent<Env> {
    let device = tch::Device::cuda_if_available();
    let iqn_model = iqn_model::create_iqn_model(N_STACK as _, DIM_FEATURE, DIM_EMBED, dim_act, LR_CRITIC, device);
    let replay_buffer = ReplayBuffer::new(REPLAY_BUFFER_CAPACITY, N_PROCS);
    IQNBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .soft_update_interval(SOFT_UPDATE_INTERVAL)
        .tau(TAU)
        .explorer(EpsilonGreedy::with_final_step(EPS_GREEDY_FINAL_STEP))
        .build(iqn_model, replay_buffer, device)
}

fn create_env(name: &str) -> Env {
    let obs_filter = ObsFilter::new(N_STACK as i64);
    let act_filter = ActFilter::default();
    PyGymEnvBuilder::default()
        .atari_wrapper(true)
        .build(name, obs_filter, act_filter).unwrap()
}

fn main() {
    env_logger::init();
    tch::manual_seed(42);

    let matches = App::new("iqn_atari")
        .version("0.1.0")
        .author("Taku Yoshioka <taku.yoshioka.4096@gmail.com>")
        .arg(Arg::with_name("name")
            .long("name")
            .takes_value(true)
            .required(true)
            .index(1)
            .help("The name of the atari environment (e.g., PongNoFrameskip-v4)"))
        .arg(Arg::with_name("play")
            .long("play")
            .takes_value(true)
            .help("Play with the trained model of the given path"))
        .arg(Arg::with_name("wait")
            .long("wait")
            .takes_value(true)
            .default_value("25")
            .help("Waiting time in milliseconds between frames when playing"))        
        .get_matches();

    let name = matches.value_of("name").unwrap();
    let mut env = create_env(name);
    let dim_act = env.get_num_actions_atari();
    let mut agent = create_agent(dim_act as _);
    
    if !matches.is_present("play") {
        let env_eval = create_env(name);
        let mut trainer = TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .build(env, env_eval, agent);
        let mut recorder = TensorboardRecorder::new(
            format!("./examples/model/iqn_{}", name)
        );
        let model_dir = format!("./examples/model/iqn_{}", name);
        trainer.train(&mut recorder);
        trainer.get_agent().save(model_dir).unwrap(); // TODO: define appropriate error
    }
    else {
        let time = matches.value_of("wait").unwrap().parse::<u64>().unwrap();
        let model_dir = matches.value_of("play").unwrap();
        env.set_render(true);
        env.set_wait_in_render(Duration::from_millis(time));
        agent.load(model_dir).unwrap(); // TODO: define appropriate error
        agent.eval();

        util::eval(&mut env, &mut agent, 5);
    }
}