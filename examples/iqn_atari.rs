mod dqn_atari_model;
use anyhow::Result;
use border::{try_from, util::url::get_model_from_url};
use border_core::{record::TensorboardRecorder, shape, util, Agent, TrainerBuilder};
use border_py_gym_env::{
    newtype_act_d, newtype_obs, AtariWrapper, FrameStackFilter, PyGymEnv, PyGymEnvBuilder,
    PyGymEnvDiscreteAct, PyGymEnvObs,
};
use border_tch_agent::{
    iqn::{IQNBuilder, IQNModelBuilder},
    replay_buffer::TchTensorBuffer,
};
use clap::{App, Arg};
use ndarray::ArrayD;
use std::{convert::TryFrom, path::Path, time::Duration};
use tch::Tensor;

const N_STACK: usize = 4;
shape!(ObsShape, [4, 1, 84, 84]);

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

mod iqn_model {
    use border_tch_agent::{model::SubModel, util::OutDim};
    use serde::{Deserialize, Serialize};
    use tch::{nn, nn::Module, nn::VarStore, Device, Tensor};

    #[allow(clippy::upper_case_acronyms)]
    #[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
    // ConvNet as feature extractor
    pub struct ConvNetConfig {
        n_stack: i64,
        feature_dim: i64,
    }

    impl ConvNetConfig {
        #[allow(dead_code)] // used in test
        pub fn new(n_stack: i64, feature_dim: i64) -> Self {
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
                .add_fn(|xs| xs.squeeze_dim(2).internal_cast_float(true) / 255)
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
    #[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
    pub struct MLPConfig {
        in_dim: i64,
        hidden_dim: i64,
        out_dim: i64,
    }

    impl MLPConfig {
        #[allow(dead_code)] // used in test
        pub fn new(in_dim: i64, hidden_dim: i64, out_dim: i64) -> Self {
            Self {
                in_dim,
                hidden_dim,
                out_dim,
            }
        }
    }

    impl OutDim for MLPConfig {
        fn get_out_dim(&self) -> i64 {
            self.out_dim
        }

        fn set_out_dim(&mut self, v: i64) {
            self.out_dim = v;
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
}

use iqn_model::{ConvNet, MLP};

fn create_agent(
    dim_act: i64,
    env_name: impl Into<String>,
) -> Result<(impl Agent<Env>, IQNBuilder)> {
    let device = tch::Device::cuda_if_available();
    let env_name = env_name.into();
    let model_cfg = format!("./examples/model/iqn_{}/model.yaml", &env_name);
    let model_cfg = IQNModelBuilder::<ConvNet, MLP>::load(Path::new(&model_cfg))?;
    let iqn = model_cfg.out_dim(dim_act).build(device)?;
    let agent_cfg = format!("./examples/model/iqn_{}/agent.yaml", &env_name);
    let agent_cfg = IQNBuilder::load(Path::new(&agent_cfg))?;
    let agent = agent_cfg
        .clone()
        .build::<_, _, _, ObsBuffer, ActBuffer>(iqn, device, tch::Device::Cpu);

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

fn get_info(name: &str) -> (&str, &str) {
    match name {
        "PongNoFrameskip-v4" => (
            "iqn_PongNoFrameskip-v4_20210430_ec2",
            "https://drive.google.com/uc?export=download&id=1Urq_gTRlhTRzELUZlz8V5W3J1twwUD5E",
        ),
        "SeaquestNoFrameskip-v4" => (
            "iqn_SeaquestNoFrameskip-v4_20210530_adam_eps_tuned",
            "https://drive.google.com/uc?export=download&id=1zDPd9ls0SewpmwlCd0Ui5OrcPt8Krld5",
        ),
        _ => {
            panic!()
        }
    }
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
        let saving_model_dir = format!("./examples/model/iqn_{}", name);
        let trainer_cfg = Path::new(&saving_model_dir).join("trainer.yaml");
        let trainer_cfg = TrainerBuilder::load(&trainer_cfg)?;
        let mut trainer = trainer_cfg.clone().build(env_train, env_eval, agent);
        let mut recorder = TensorboardRecorder::new(format!("./examples/model/iqn_{}", name));

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
            agent.load(model_dir)?;
        } else {
            let (file_base, url) = get_info(name);
            let model_dir = get_model_from_url(url, file_base)?;
            agent.load(model_dir)?;
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
    use super::iqn_model::{ConvNet, ConvNetConfig, MLPConfig, MLP};
    use anyhow::Result;
    use border_core::TrainerBuilder;
    use border_tch_agent::{
        iqn::{model::IQNSample, EpsilonGreedy, IQNBuilder, IQNModelBuilder},
        opt::OptimizerConfig,
        util::OptInterval,
    };
    use std::path::Path;

    // IQN model parameters
    const LR_QNET: f64 = 1e-4;
    const N_STACK: i64 = 4;
    const FEATURE_DIM: i64 = 3136;
    const EMBED_DIM: i64 = 64;
    const HIDDEN_DIM: i64 = 512;

    fn iqn_model_builder() -> IQNModelBuilder<ConvNet, MLP> {
        let out_dim = 0;
        let f_config = ConvNetConfig::new(N_STACK, FEATURE_DIM);
        let m_config = MLPConfig::new(FEATURE_DIM, HIDDEN_DIM, out_dim);
        IQNModelBuilder::default()
            .feature_dim(FEATURE_DIM)
            .embed_dim(EMBED_DIM)
            // .learning_rate(LR_QNET)
            .opt_config(OptimizerConfig::AdamEps {
                lr: 5e-5,
                eps: 0.01 / 32.0,
            })
            .f_config(f_config)
            .m_config(m_config)
    }

    // IQN agent parameters
    const DISCOUNT_FACTOR: f64 = 0.99;
    const BATCH_SIZE: usize = 32;
    const N_TRANSITIONS_WARMUP: usize = 2500;
    const N_UPDATES_PER_OPT: usize = 1;
    const OPT_INTERVAL: OptInterval = OptInterval::Steps(4);
    const SOFT_UPDATE_INTERVAL: usize = 2_500;
    const TAU: f64 = 1.0;
    const REPLAY_BUFFER_CAPACITY: usize = 1_000_000;
    const SAMPLE_PERCENTS_PRED: IQNSample = IQNSample::Uniform64;
    const SAMPLE_PERCENTS_TGT: IQNSample = IQNSample::Uniform64;
    const SAMPLE_PERCENTS_ACT: IQNSample = IQNSample::Uniform32;
    const EPS_START: f64 = 1.0;
    const EPS_FINAL: f64 = 0.02;
    const EPS_FINAL_STEP: usize = 1_000_000;

    fn iqn_agent_builder() -> IQNBuilder {
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
            .replay_buffer_capacity(REPLAY_BUFFER_CAPACITY)
    }

    // Training parameters
    const MAX_OPTS: usize = 50_000_000;
    const EVAL_INTERVAL: usize = 10_000;
    const N_EPISODES_PER_EVAL: usize = 1;

    fn trainer_builder(saving_model_dir: &String) -> TrainerBuilder {
        TrainerBuilder::default()
            .max_opts(MAX_OPTS)
            .eval_interval(EVAL_INTERVAL)
            .n_episodes_per_eval(N_EPISODES_PER_EVAL)
            .model_dir(saving_model_dir)
    }

    #[test]
    fn save_configs() -> Result<()> {
        // let env_name = "PongNoFrameskip-v4";
        let env_name = "SeaquestNoFrameskip-v4";
        let saving_model_dir = format!("./examples/model/iqn_{}", env_name);
        let model_cfg = Path::new(&saving_model_dir).join("model.yaml");
        let agent_cfg = Path::new(&saving_model_dir).join("agent.yaml");
        let trainer_cfg = Path::new(&saving_model_dir).join("trainer.yaml");
        println!("{:?}", agent_cfg);

        iqn_model_builder().save(model_cfg)?;
        iqn_agent_builder().save(agent_cfg)?;
        trainer_builder(&saving_model_dir).save(trainer_cfg)?;

        Ok(())
    }
}
