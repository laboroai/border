use anyhow::{Context, Result};
use border::{
    agent::tch::{dqn::model::DQNModelBuilder, DQNBuilder, ReplayBuffer as ReplayBuffer_},
    env::py_gym_env::{
        act_d::{PyGymEnvDiscreteAct, PyGymEnvDiscreteActRawFilter},
        framestack::FrameStackFilter,
        obs::PyGymEnvObs,
        tch::{act_d::TchPyGymEnvDiscreteActBuffer, obs::TchPyGymEnvObsBuffer},
        AtariWrapper, PyVecGymEnv, PyVecGymEnvBuilder, Shape,
    },
    shape,
};
use border_core::{record::TensorboardRecorder, Agent, TrainerBuilder};
use clap::{App, Arg};
use std::path::Path;

mod dqn_atari_model;
use dqn_atari_model::CNN;

const N_PROCS: usize = 4;
const N_STACK: usize = 4;
shape!(ObsShape, [4, 1, 84, 84]);

type ObsFilter = FrameStackFilter<ObsShape, u8, u8>;
type ActFilter = PyGymEnvDiscreteActRawFilter;
type Obs = PyGymEnvObs<ObsShape, u8, u8>;
type Act = PyGymEnvDiscreteAct;
type Env = PyVecGymEnv<Obs, Act, ObsFilter, ActFilter>;
type ObsBuffer = TchPyGymEnvObsBuffer<ObsShape, u8, u8>;
type ActBuffer = TchPyGymEnvDiscreteActBuffer;
type ReplayBuffer = ReplayBuffer_<Env, ObsBuffer, ActBuffer>;

fn create_agent(dim_act: i64, env_name: impl Into<String>) -> Result<impl Agent<Env>> {
    let device = tch::Device::cuda_if_available();
    let env_name = env_name.into();
    let model_cfg = format!("./examples/model/dqn_{}_vec/model.yaml", &env_name);
    let model_cfg = DQNModelBuilder::<CNN>::load(Path::new(&model_cfg))?;
    let qnet = model_cfg.out_dim(dim_act).build(device)?;
    let agent_cfg = format!("./examples/model/dqn_{}_vec/agent.yaml", &env_name);
    let agent_cfg = DQNBuilder::load(Path::new(&agent_cfg))?;
    let replay_buffer = ReplayBuffer::new(agent_cfg.get_replay_burffer_capacity(), N_PROCS);
    let agent = agent_cfg.build_with_replay_buffer(qnet, replay_buffer, device);

    Ok(agent)
}

fn create_env(name: &str, mode: AtariWrapper) -> Result<Env> {
    let obs_filter = ObsFilter::vectorized(N_PROCS as _, N_STACK as _);
    let act_filter = ActFilter::vectorized();
    PyVecGymEnvBuilder::default()
        .atari_wrapper(Some(mode))
        .n_procs(N_PROCS)
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
        .get_matches();

    let name = matches.value_of("name").unwrap();
    let env_eval = create_env(name, AtariWrapper::Eval)?;
    let dim_act = env_eval.get_num_actions_atari();
    let agent = create_agent(dim_act as _, name)?;

    let env_train = create_env(name, AtariWrapper::Train)?;

    let saving_model_dir = format!("./examples/model/dqn_{}_vec", name);
    let trainer_cfg = Path::new(&saving_model_dir).join("trainer.yaml");
    let trainer_cfg = TrainerBuilder::load(&trainer_cfg)?;
    let mut trainer = trainer_cfg.build(env_train, env_eval, agent);
    let mut recorder = TensorboardRecorder::new(format!("./examples/model/dqn_{}_vec", name));

    trainer.train(&mut recorder);

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
