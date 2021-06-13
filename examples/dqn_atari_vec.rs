mod dqn_atari_model;
use anyhow::{Context, Result};
use border::try_from;
use border_core::{record::TensorboardRecorder, shape, Agent, Shape, TrainerBuilder};
use border_py_gym_env::{
    newtype_act_d, newtype_obs, AtariWrapper, FrameStackFilter, PyGymEnvDiscreteAct, PyGymEnvObs,
    PyVecGymEnv, PyVecGymEnvBuilder,
};
use border_tch_agent::{
    dqn::{DQNBuilder, DQNModelBuilder},
    replay_buffer::TchTensorBuffer,
};
use clap::{App, Arg};
use dqn_atari_model::CNN;
use log::info;
use ndarray::ArrayD;
use std::{convert::TryFrom, path::Path};
use tch::Tensor;

const N_PROCS: i64 = 4;
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
        let t = t.unsqueeze(-1);

        debug_assert_eq!(&t.size()[..], &[N_PROCS, 1]);
        t
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
type Env = PyVecGymEnv<Obs, Act, ObsFilter, ActFilter>;
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

fn create_env(name: &str, mode: AtariWrapper) -> Result<Env> {
    let obs_filter = ObsFilter::vectorized(N_PROCS as _, N_STACK as _);
    let act_filter = ActFilter::default(); //vectorized();
    PyVecGymEnvBuilder::default()
        .atari_wrapper(Some(mode))
        .n_procs(N_PROCS as _)
        .build(name, obs_filter, act_filter)
        .context("Failed to create vecenv")
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = App::new("dqn_atari_vec")
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
            Arg::with_name("show-config")
                .long("show-config")
                .takes_value(false)
                .help("Showing configuration loaded from files"),
        )
        .get_matches();

    let name = matches.value_of("name").unwrap();
    info!("name = {:?}", name);

    let env_eval = create_env(name, AtariWrapper::Eval)?;
    let dim_act = env_eval.get_num_actions_atari();
    let (agent, agent_cfg) = create_agent(dim_act as _, name)?;
    let env_train = create_env(name, AtariWrapper::Train)?;

    let cfg_dir = format!("./examples/model/dqn_{}", name);
    let saving_model_dir = format!("./examples/model/dqn_{}_vec", name);
    let trainer_cfg_file = Path::new(&cfg_dir).join("trainer.yaml");
    let trainer_cfg = TrainerBuilder::load(trainer_cfg_file)?.model_dir(&saving_model_dir);

    if matches.is_present("show-config") {
        println!("Device: {:?}", tch::Device::cuda_if_available());
        println!("(trainer_cfg)");
        println!("{}", serde_yaml::to_string(&trainer_cfg)?);
        println!("(agent_cfg)");
        println!("{}", serde_yaml::to_string(&agent_cfg)?);
        env_train.close();
        env_eval.close();
    } else {
        let mut trainer = trainer_cfg.build(env_train, env_eval, agent);
        let mut recorder = TensorboardRecorder::new(&saving_model_dir);
        trainer.train(&mut recorder);
        trainer.get_env().close();
        trainer.get_env_eval().close();
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use tempdir::TempDir;

    #[test]
    fn test_dqn_atari_vec() -> Result<()> {
        tch::manual_seed(42);

        let name = "PongNoFrameskip-v4";
        let saving_model_dir_test = TempDir::new("dqn_PongNoFrameskip-v4_vec")?;
        let saving_model_dir_test = saving_model_dir_test.path().to_str().unwrap();
        let saving_model_dir = format!("./examples/model/dqn_{}_vec", name);

        let env_eval = create_env(name, AtariWrapper::Eval)?;
        let dim_act = env_eval.get_num_actions_atari();
        let agent = create_agent(dim_act as _, name)?;
        let env_train = create_env(name, AtariWrapper::Train)?;

        let trainer_cfg = Path::new(&saving_model_dir).join("trainer.yaml");
        let trainer_cfg = TrainerBuilder::load(&trainer_cfg)?
            .max_opts(5)
            .model_dir(saving_model_dir_test);
        let mut trainer = trainer_cfg.build(env_train, env_eval, agent);
        let mut recorder = TensorboardRecorder::new(saving_model_dir_test);

        trainer.train(&mut recorder);

        Ok(())
    }
}
