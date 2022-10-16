mod util_iqn_atari;
use anyhow::Result;
use border::util::get_model_from_url;
use border_core::{
    record::{BufferedRecorder, TensorboardRecorder},
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    shape, util, Agent, Env as _, Policy, Trainer, TrainerConfig,
};
use border_derive::{Act, SubBatch};
use border_py_gym_env::{
    FrameStackFilter, PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvDiscreteAct,
    PyGymEnvDiscreteActRawFilter, PyGymEnvObs,
};
use border_tch_agent::{
    cnn::CNN,
    mlp::Mlp,
    iqn::{Iqn as Iqn_, IqnConfig as IqnConfig_},
    TensorSubBatch,
};
use clap::{App, Arg, ArgMatches};
use util_iqn_atari::{model_dir as model_dir_, Params};

const N_STACK: i64 = 4;

type PyObsDtype = u8;
type ObsDtype = u8;

shape!(ObsShape, [N_STACK as usize, 1, 84, 84]);
shape!(ActShape, [1]);

// #[derive(Clone, Debug, Obs)]
// struct Obs(PyGymEnvObs<ObsShape, PyObsDtype, ObsDtype>);
type Obs = PyGymEnvObs<ObsShape, PyObsDtype, ObsDtype>;

#[derive(Clone, SubBatch)]
struct ObsBatch(TensorSubBatch<ObsShape, ObsDtype>);

impl From<Obs> for ObsBatch {
    fn from(obs: Obs) -> Self {
        let tensor = obs.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

// Wrap `PyGymEnvDiscreteAct` to make a new type.
// Act also implements Into<Tensor>.
// TODO: Consider to implement Into<Tensor> on PyGymEnvDiscreteAct when feature=tch.
#[derive(Clone, Debug, Act)]
struct Act(PyGymEnvDiscreteAct);

#[derive(SubBatch)]
struct ActBatch(TensorSubBatch<ActShape, i64>);

impl From<Act> for ActBatch {
    fn from(act: Act) -> Self {
        let tensor = act.into();
        Self(TensorSubBatch::from_tensor(tensor))
    }
}

type ObsFilter = FrameStackFilter<ObsShape, PyObsDtype, ObsDtype, Obs>;
type ActFilter = PyGymEnvDiscreteActRawFilter<Act>;
type Env = PyGymEnv<Obs, Act, ObsFilter, ActFilter>;
type EnvConfig = PyGymEnvConfig<Obs, Act, ObsFilter, ActFilter>;
type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
type Iqn = Iqn_<Env, CNN, Mlp, ReplayBuffer>;
type IqnConfig = IqnConfig_<CNN, Mlp>;

fn init<'a>() -> ArgMatches<'a> {
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

    matches
}

fn model_dir(matches: &ArgMatches) -> Result<String> {
    let name = matches
        .value_of("name")
        .expect("The name of the environment was not given")
        .to_string();
    let params = Params::default();

    // if matches.is_present("ddqn") {
    //     params = params.ddqn();
    // }

    // if matches.is_present("per") {
    //     params = params.per();
    // }

    // if matches.is_present("debug") {
    //     params = params.debug();
    // }

    model_dir_(name, &params)
}

fn model_dir_for_play(matches: &ArgMatches) -> String {
    let name = matches.value_of("name").unwrap();

    if matches.is_present("play") {
        matches.value_of("play").unwrap().to_string()
    } else if matches.is_present("play-gdrive") {
        if name == "SeaquestNoFrameskip-v4" {
            let file_base = "iqn_SeaquestNoFrameskip-v4_20210530_adam_eps_tuned";
            let url =
                "https://drive.google.com/uc?export=download&id=1zDPd9ls0SewpmwlCd0Ui5OrcPt8Krld5";
            let model_dir = get_model_from_url(url, file_base).unwrap();
            model_dir.as_ref().to_str().unwrap().to_string()
        } else {
            panic!("Failed to download the model for {:?}", name);
        }
    } else {
        panic!("Failed to download the model for {:?}", name);
    }
}

fn env_config(name: &str) -> EnvConfig {
    PyGymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
        .name(name.to_string())
        .obs_filter_config(ObsFilter::default_config())
        .act_filter_config(ActFilter::default_config())
        .atari_wrapper(Some(border_py_gym_env::AtariWrapper::Eval))
}

fn n_actions(env_config: &EnvConfig) -> Result<usize> {
    Ok(Env::build(env_config, 0)?.get_num_actions_atari() as usize)
}

fn load_trainer_config<'a>(model_dir: impl Into<&'a str>) -> Result<TrainerConfig> {
    let config_path = format!("{}/trainer.yaml", model_dir.into());
    TrainerConfig::load(config_path)
}

fn load_replay_buffer_config<'a>(
    model_dir: impl Into<&'a str>,
) -> Result<SimpleReplayBufferConfig> {
    let config_path = format!("{}/replay_buffer.yaml", model_dir.into());
    SimpleReplayBufferConfig::load(config_path)
}

fn load_iqn_config<'a>(model_dir: impl Into<&'a str>) -> Result<IqnConfig> {
    let config_path = format!("{}/agent.yaml", model_dir.into());
    IqnConfig::load(config_path)
}

fn train(matches: ArgMatches) -> Result<()> {
    let name = matches.value_of("name").unwrap();
    let model_dir = model_dir(&matches)?;
    let env_config_train =
        env_config(name).atari_wrapper(Some(border_py_gym_env::AtariWrapper::Train));
    let env_config_eval =
        env_config(name).atari_wrapper(Some(border_py_gym_env::AtariWrapper::Eval));
    let n_actions = n_actions(&env_config_train)?;

    // Configurations
    let agent_config = load_iqn_config(model_dir.as_str())?
        .out_dim(n_actions as _)
        .device(tch::Device::cuda_if_available());
    let trainer_config = load_trainer_config(model_dir.as_str())?;
    let replay_buffer_config = load_replay_buffer_config(model_dir.as_str())?;
    let step_proc_config = SimpleStepProcessorConfig {};

    // if matches.is_present("show-config") {
    //     show_config(&env_config_train, &agent_config, &trainer_config);
    // } else {
    let mut trainer = Trainer::<Env, StepProc, ReplayBuffer>::build(
        trainer_config,
        env_config_train,
        Some(env_config_eval),
        step_proc_config,
        replay_buffer_config,
    );
    let mut recorder = TensorboardRecorder::new(model_dir);
    let mut agent = Iqn::build(agent_config);
    trainer.train(&mut agent, &mut recorder)?;
    // }

    Ok(())
}

fn play(matches: ArgMatches) -> Result<()> {
    let name = matches.value_of("name").unwrap();
    let model_dir = model_dir_for_play(&matches);
    let env_config = env_config(name);
    let n_actions = n_actions(&env_config)?;
    let agent_config = load_iqn_config(model_dir.as_str())?
        .out_dim(n_actions as _)
        .device(tch::Device::cuda_if_available());
    let mut agent = Iqn::build(agent_config);
    let mut env = Env::build(&env_config, 0)?;
    let mut recorder = BufferedRecorder::new();

    env.set_render(true);
    agent.load(model_dir + "/best")?;
    agent.eval();

    let _ = util::eval_with_recorder(&mut env, &mut agent, 5, &mut recorder)?;

    Ok(())
}

fn main() -> Result<()> {
    let matches = init();

    if matches.is_present("play") || matches.is_present("play-gdrive") {
        play(matches)?;
    } else {
        train(matches)?;
    }

    Ok(())
}

// fn create_agent(
//     dim_act: i64,
//     env_name: impl Into<String>,
// ) -> Result<(impl Agent<Env>, IQNBuilder)> {
//     let device = tch::Device::cuda_if_available();
//     let env_name = env_name.into();
//     let model_cfg = format!("./border/examples/model/iqn_{}/model.yaml", &env_name);
//     let model_cfg = IQNModelBuilder::<ConvNet, MLP>::load(Path::new(&model_cfg))?;
//     let iqn = model_cfg.out_dim(dim_act).build(device)?;
//     let agent_cfg = format!("./border/examples/model/iqn_{}/agent.yaml", &env_name);
//     let agent_cfg = IQNBuilder::load(Path::new(&agent_cfg))?;
//     let agent = agent_cfg
//         .clone()
//         .build::<_, _, _, ObsBuffer, ActBuffer>(iqn, device);

//     Ok((agent, agent_cfg))
// }

// fn create_env(name: &str, mode: AtariWrapper) -> Env {
//     let obs_filter = ObsFilter::new(N_STACK as i64);
//     let act_filter = ActFilter::default();
//     PyGymEnvBuilder::default()
//         .atari_wrapper(Some(mode))
//         .build(name, obs_filter, act_filter)
//         .unwrap()
// }

// fn get_info(name: &str) -> (&str, &str) {
//     match name {
//         "PongNoFrameskip-v4" => (
//             "iqn_PongNoFrameskip-v4_20210430_ec2",
//             "https://drive.google.com/uc?export=download&id=1Urq_gTRlhTRzELUZlz8V5W3J1twwUD5E",
//         ),
//         "SeaquestNoFrameskip-v4" => (
//             "iqn_SeaquestNoFrameskip-v4_20210530_adam_eps_tuned",
//             "https://drive.google.com/uc?export=download&id=1zDPd9ls0SewpmwlCd0Ui5OrcPt8Krld5",
//         ),
//         _ => {
//             panic!()
//         }
//     }
// }

// fn main() -> Result<()> {
//     env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
//     tch::manual_seed(42);

//     let name = matches.value_of("name").unwrap();
//     let mut env_eval = create_env(name, AtariWrapper::Eval);
//     let dim_act = env_eval.get_num_actions_atari();
//     let agent = create_agent(dim_act as _, name)?;
//     let agent_cfg = agent.1;
//     let mut agent = agent.0;

//     if !(matches.is_present("play") || matches.is_present("play-gdrive")) {
//         let env_train = create_env(name, AtariWrapper::Train);
//         let saving_model_dir = format!("./border/examples/model/iqn_{}", name);
//         let trainer_cfg = Path::new(&saving_model_dir).join("trainer.yaml");
//         let trainer_cfg = TrainerBuilder::load(&trainer_cfg)?;
//         let mut trainer = trainer_cfg.clone().build(env_train, env_eval, agent);
//         let mut recorder =
//             TensorboardRecorder::new(format!("./border/examples/model/iqn_{}", name));

//         if matches.is_present("show-config") {
//             println!("Device: {:?}", tch::Device::cuda_if_available());
//             println!("{:?}", trainer_cfg);
//             println!("{:?}", agent_cfg);
//             return Ok(());
//         }

//         trainer.train(&mut recorder);
//     } else {
//         if matches.is_present("play") {
//             let model_dir = matches
//                 .value_of("play")
//                 .expect("Failed to parse model directory");
//             agent.load(model_dir)?;
//         } else {
//             let (file_base, url) = get_info(name);
//             let model_dir = get_model_from_url(url, file_base)?;
//             agent.load(model_dir)?;
//         };

//         let time = matches.value_of("wait").unwrap().parse::<u64>()?;
//         env_eval.set_render(true);
//         env_eval.set_wait_in_render(Duration::from_millis(time));
//         agent.eval();
//         util::eval(&mut env_eval, &mut agent, 5);
//     }

//     Ok(())
// }

// #[cfg(test)]
// mod test {
//     use super::iqn_model::{ConvNet, ConvNetConfig, MLPConfig, MLP};
//     use anyhow::Result;
//     use border_core::TrainerBuilder;
//     use border_tch_agent::{
//         iqn::{model::IQNSample, EpsilonGreedy, IQNBuilder, IQNModelBuilder},
//         opt::OptimizerConfig,
//         util::OptInterval,
//     };
//     use std::path::Path;

//     // IQN model parameters
//     const LR_QNET: f64 = 1e-4;
//     const N_STACK: i64 = 4;
//     const FEATURE_DIM: i64 = 3136;
//     const EMBED_DIM: i64 = 64;
//     const HIDDEN_DIM: i64 = 512;

//     fn iqn_model_builder() -> IQNModelBuilder<ConvNet, MLP> {
//         let out_dim = 0;
//         let f_config = ConvNetConfig::new(N_STACK, FEATURE_DIM);
//         let m_config = MLPConfig::new(FEATURE_DIM, HIDDEN_DIM, out_dim);
//         IQNModelBuilder::default()
//             .feature_dim(FEATURE_DIM)
//             .embed_dim(EMBED_DIM)
//             // .learning_rate(LR_QNET)
//             .opt_config(OptimizerConfig::AdamEps {
//                 lr: 5e-5,
//                 eps: 0.01 / 32.0,
//             })
//             .f_config(f_config)
//             .m_config(m_config)
//     }

//     // IQN agent parameters
//     const DISCOUNT_FACTOR: f64 = 0.99;
//     const BATCH_SIZE: usize = 32;
//     const N_TRANSITIONS_WARMUP: usize = 2500;
//     const N_UPDATES_PER_OPT: usize = 1;
//     const OPT_INTERVAL: OptInterval = OptInterval::Steps(4);
//     const SOFT_UPDATE_INTERVAL: usize = 2_500;
//     const TAU: f64 = 1.0;
//     const REPLAY_BUFFER_CAPACITY: usize = 1_000_000;
//     const SAMPLE_PERCENTS_PRED: IQNSample = IQNSample::Uniform64;
//     const SAMPLE_PERCENTS_TGT: IQNSample = IQNSample::Uniform64;
//     const SAMPLE_PERCENTS_ACT: IQNSample = IQNSample::Uniform32;
//     const EPS_START: f64 = 1.0;
//     const EPS_FINAL: f64 = 0.02;
//     const EPS_FINAL_STEP: usize = 1_000_000;

//     fn iqn_agent_builder() -> IQNBuilder {
//         IQNBuilder::default()
//             .opt_interval(OPT_INTERVAL)
//             .n_updates_per_opt(N_UPDATES_PER_OPT)
//             .min_transitions_warmup(N_TRANSITIONS_WARMUP)
//             .batch_size(BATCH_SIZE)
//             .discount_factor(DISCOUNT_FACTOR)
//             .soft_update_interval(SOFT_UPDATE_INTERVAL)
//             .tau(TAU)
//             .explorer(EpsilonGreedy::with_params(
//                 EPS_START,
//                 EPS_FINAL,
//                 EPS_FINAL_STEP,
//             ))
//             .sample_percent_pred(SAMPLE_PERCENTS_PRED)
//             .sample_percent_tgt(SAMPLE_PERCENTS_TGT)
//             .sample_percent_act(SAMPLE_PERCENTS_ACT)
//             .replay_buffer_capacity(REPLAY_BUFFER_CAPACITY)
//     }

//     // Training parameters
//     const MAX_OPTS: usize = 50_000_000;
//     const EVAL_INTERVAL: usize = 10_000;
//     const N_EPISODES_PER_EVAL: usize = 1;

//     fn trainer_builder(saving_model_dir: &String) -> TrainerBuilder {
//         TrainerBuilder::default()
//             .max_opts(MAX_OPTS)
//             .eval_interval(EVAL_INTERVAL)
//             .n_episodes_per_eval(N_EPISODES_PER_EVAL)
//             .model_dir(saving_model_dir)
//     }

//     #[test]
//     fn save_configs() -> Result<()> {
//         // let env_name = "PongNoFrameskip-v4";
//         let env_name = "SeaquestNoFrameskip-v4";
//         let saving_model_dir = format!("./border/examples/model/iqn_{}", env_name);
//         let model_cfg = Path::new(&saving_model_dir).join("model.yaml");
//         let agent_cfg = Path::new(&saving_model_dir).join("agent.yaml");
//         let trainer_cfg = Path::new(&saving_model_dir).join("trainer.yaml");
//         println!("{:?}", agent_cfg);

//         iqn_model_builder().save(model_cfg)?;
//         iqn_agent_builder().save(agent_cfg)?;
//         trainer_builder(&saving_model_dir).save(trainer_cfg)?;

//         Ok(())
//     }
// }
