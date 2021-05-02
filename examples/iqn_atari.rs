use anyhow::Result;
use clap::{App, Arg};
use std::time::Duration;

use border::{
    agent::{
        tch::{
            iqn::{model::IQNSample, EpsilonGreedy, IQNBuilder},
            ReplayBuffer as ReplayBuffer_,
        },
        OptInterval,
    },
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

const N_PROCS: usize = 1;
const N_STACK: usize = 4;
const DIM_OBS: [usize; 4] = [4, 1, 84, 84];
const DIM_FEATURE: i64 = 3136;
const DIM_EMBED: i64 = 64;
const DIM_HIDDEN: i64 = 512;
const SAMPLE_PERCENTS_PRED: IQNSample = IQNSample::Uniform8;
const SAMPLE_PERCENTS_TGT: IQNSample = IQNSample::Uniform8;
const SAMPLE_PERCENTS_ACT: IQNSample = IQNSample::Uniform32;
const LR_QNET: f64 = 1e-4;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 32;
const N_TRANSITIONS_WARMUP: usize = 2500;
const N_UPDATES_PER_OPT: usize = 1;
const OPT_INTERVAL: OptInterval = OptInterval::Steps(1);
const SOFT_UPDATE_INTERVAL: usize = 10_000;
const TAU: f64 = 1.0;
const MAX_OPTS: usize = 5_000_000;
const EVAL_INTERVAL: usize = 10_000;
const REPLAY_BUFFER_CAPACITY: usize = 50_000;
const N_EPISODES_PER_EVAL: usize = 1;
const EPS_START: f64 = 1.0;
const EPS_FINAL: f64 = 0.02;
const EPS_FINAL_STEP: usize = 1_000_000;

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
    use border::agent::tch::{
        iqn::{IQNModel, IQNModelBuilder},
        model::SubModel,
    };
    use tch::{nn, nn::Module, nn::VarStore, Device, Tensor};

    #[allow(clippy::upper_case_acronyms)]
    // ConvNet as feature extractor
    pub struct ConvNetConfig {
        n_stack: i64,
        feature_dim: i64,
    }

    impl ConvNetConfig {
        fn new(n_stack: i64, feature_dim: i64) -> Self {
            Self {
                n_stack,
                feature_dim,
            }
        }
    }

    #[allow(clippy::upper_case_acronyms)]
    pub struct ConvNet {
        n_stack: i64,
        feature_dim: i64,
        device: Device,
        seq: nn::Sequential,
    }

    fn stride(s: i64) -> nn::ConvConfig {
        nn::ConvConfig {
            stride: s,
            ..Default::default()
        }
    }

    impl ConvNet {
        fn _build(var_store: &VarStore, n_stack: i64, feature_dim: i64) -> Self {
            let p = &var_store.root();
            let seq = nn::seq()
                .add_fn(|xs| xs.squeeze1(2).internal_cast_float(true) / 255)
                .add(nn::conv2d(p / "c1", n_stack, 32, 8, stride(4)))
                .add_fn(|xs| xs.relu())
                .add(nn::conv2d(p / "c2", 32, 64, 4, stride(2)))
                .add_fn(|xs| xs.relu())
                .add(nn::conv2d(p / "c3", 64, 64, 3, stride(1)))
                .add_fn(|xs| xs.relu().flat_view());

            Self {
                n_stack,
                feature_dim,
                device: var_store.device(),
                seq,
            }
        }
    }

    impl SubModel for ConvNet {
        type Config = ConvNetConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, x: &Self::Input) -> Tensor {
            self.seq.forward(&x.to(self.device))
        }

        fn build(var_store: &VarStore, config: Self::Config) -> Self {
            Self::_build(var_store, config.n_stack, config.feature_dim)
        }

        fn clone_with_var_store(&self, var_store: &nn::VarStore) -> Self {
            Self::_build(var_store, self.n_stack, self.feature_dim)
        }
    }

    // MLP as output layer of IQNModel
    #[allow(clippy::upper_case_acronyms)]
    pub struct MLPConfig {
        in_dim: i64,
        hidden_dim: i64,
        out_dim: i64,
    }

    impl MLPConfig {
        fn new(in_dim: i64, hidden_dim: i64, out_dim: i64) -> Self {
            Self {
                in_dim,
                hidden_dim,
                out_dim,
            }
        }
    }

    #[allow(clippy::upper_case_acronyms)]
    pub struct MLP {
        in_dim: i64,
        hidden_dim: i64,
        out_dim: i64,
        seq: nn::Sequential,
    }

    impl MLP {
        fn _build(var_store: &VarStore, in_dim: i64, hidden_dim: i64, out_dim: i64) -> Self {
            let p = &var_store.root();
            let seq = nn::seq()
                .add(nn::linear(
                    p / "cl1",
                    in_dim,
                    hidden_dim,
                    Default::default(),
                ))
                .add_fn(|x| x.relu())
                .add(nn::linear(
                    p / "cl2",
                    hidden_dim,
                    out_dim,
                    Default::default(),
                ));

            Self {
                in_dim,
                hidden_dim,
                out_dim,
                seq,
            }
        }
    }

    impl SubModel for MLP {
        type Config = MLPConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: &Self::Input) -> Self::Output {
            self.seq.forward(input)
        }

        fn build(var_store: &nn::VarStore, config: Self::Config) -> Self {
            Self::_build(var_store, config.in_dim, config.hidden_dim, config.out_dim)
        }

        fn clone_with_var_store(&self, var_store: &VarStore) -> Self {
            Self::_build(var_store, self.in_dim, self.hidden_dim, self.out_dim)
        }
    }

    // IQN model
    pub fn create_iqn_model(
        n_stack: i64,
        feature_dim: i64,
        embed_dim: i64,
        hidden_dim: i64,
        out_dim: i64,
        learning_rate: f64,
        device: Device,
    ) -> IQNModel<ConvNet, MLP> {
        let fe_config = ConvNetConfig::new(n_stack, feature_dim);
        let m_config = MLPConfig::new(feature_dim, hidden_dim, out_dim);
        IQNModelBuilder::default()
            .feature_dim(feature_dim)
            .embed_dim(embed_dim)
            .out_dim(out_dim)
            .learning_rate(learning_rate)
            .build(fe_config, m_config, device)
    }
}

fn create_agent(dim_act: i64) -> impl Agent<Env> {
    let device = tch::Device::cuda_if_available();
    let iqn_model = iqn_model::create_iqn_model(
        N_STACK as i64,
        DIM_FEATURE,
        DIM_EMBED,
        DIM_HIDDEN,
        dim_act,
        LR_QNET,
        device,
    );
    let replay_buffer = ReplayBuffer::new(REPLAY_BUFFER_CAPACITY, N_PROCS);
    IQNBuilder::default()
        .opt_interval(OPT_INTERVAL)
        .n_updates_per_opt(N_UPDATES_PER_OPT)
        .min_transitions_warmup(N_TRANSITIONS_WARMUP)
        .batch_size(BATCH_SIZE)
        .discount_factor(DISCOUNT_FACTOR)
        .soft_update_interval(SOFT_UPDATE_INTERVAL)
        .tau(TAU)
        .explorer(EpsilonGreedy::with_params(
            EPS_START,
            EPS_FINAL,
            EPS_FINAL_STEP,
        ))
        .sample_percent_pred(SAMPLE_PERCENTS_PRED)
        .sample_percent_tgt(SAMPLE_PERCENTS_TGT)
        .sample_percent_act(SAMPLE_PERCENTS_ACT)
        .build(iqn_model, replay_buffer, device)
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

    let matches = App::new("iqn_atari")
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
        .get_matches();

    let name = matches.value_of("name").unwrap();
    let mut env_eval = create_env(name, AtariWrapper::Eval);
    let dim_act = env_eval.get_num_actions_atari();
    let mut agent = create_agent(dim_act as _);

    if !(matches.is_present("play") || matches.is_present("play-gdrive")) {
        let env_train = create_env(name, AtariWrapper::Train);
        let saving_model_dir = format!("./examples/model/dqn_{}", name);
        let mut trainer = TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .model_dir(saving_model_dir)
            .build(env_train, env_eval, agent);
        let mut recorder = TensorboardRecorder::new(format!("./examples/model/iqn_{}", name));
        trainer.train(&mut recorder);
    } else {
        if matches.is_present("play") {
            let model_dir = matches
                .value_of("play")
                .expect("Failed to parse model directory");
            agent.load(model_dir).unwrap(); // TODO: define appropriate error
        } else {
            // TODO: change file_base and url depending on the game
            let file_base = "iqn_PongNoFrameskip-v4_20210430_ec2";
            let url =
                "https://drive.google.com/uc?export=download&id=1Urq_gTRlhTRzELUZlz8V5W3J1twwUD5E";
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
