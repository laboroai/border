use anyhow::Result;
use border_core::{
    record::AggregateRecorder,
    replay_buffer::{
        SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig, SubBatch,
    },
    Agent, DefaultEvaluator, Evaluator as _, Policy, Trainer, TrainerConfig,
};
use border_mlflow_tracking::MlflowTrackingClient;
use border_py_gym_env::{
    util::vec_to_tensor, ArrayObsFilter, DiscreteActFilter, GymActFilter, GymEnv, GymEnvConfig,
    GymObsFilter,
};
use border_tch_agent::{
    iqn::{EpsilonGreedy, Iqn as Iqn_, IqnConfig, IqnModelConfig},
    mlp::{Mlp, MlpConfig},
    TensorSubBatch,
};
use border_tensorboard::TensorboardRecorder;
use clap::{App, Arg, ArgMatches};
use ndarray::{ArrayD, IxDyn};
use serde::Serialize;
use std::convert::TryFrom;
use tch::Tensor;

const DIM_OBS: i64 = 4;
const DIM_ACT: i64 = 2;
const LR_CRITIC: f64 = 0.001;
const DIM_FEATURE: i64 = 256;
const DIM_EMBED: i64 = 64;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const WARMUP_PERIOD: usize = 100;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.01;
const SOFT_UPDATE_INTERVAL: usize = 1;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 30000;
const EVAL_INTERVAL: usize = 1000;
const REPLAY_BUFFER_CAPACITY: usize = 10000;
const N_EPISODES_PER_EVAL: usize = 5;
const EPS_START: f64 = 1.0;
const EPS_FINAL: f64 = 0.02;
const FINAL_STEP: usize = MAX_OPTS / 3;
const MODEL_DIR: &str = "border/examples/gym/model/tch/iqn_cartpole";

mod obs_act_types {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct Obs(ArrayD<f32>);

    impl border_core::Obs for Obs {
        fn dummy(_n: usize) -> Self {
            Self(ArrayD::zeros(IxDyn(&[0])))
        }

        fn len(&self) -> usize {
            self.0.shape()[0]
        }
    }

    impl From<ArrayD<f32>> for Obs {
        fn from(obs: ArrayD<f32>) -> Self {
            Obs(obs)
        }
    }

    impl From<Obs> for Tensor {
        fn from(obs: Obs) -> Tensor {
            Tensor::try_from(&obs.0).unwrap()
        }
    }

    pub struct ObsBatch(TensorSubBatch);

    impl SubBatch for ObsBatch {
        fn new(capacity: usize) -> Self {
            Self(TensorSubBatch::new(capacity))
        }

        fn push(&mut self, i: usize, data: Self) {
            self.0.push(i, data.0)
        }

        fn sample(&self, ixs: &Vec<usize>) -> Self {
            let buf = self.0.sample(ixs);
            Self(buf)
        }
    }

    impl From<Obs> for ObsBatch {
        fn from(obs: Obs) -> Self {
            let tensor = obs.into();
            Self(TensorSubBatch::from_tensor(tensor))
        }
    }

    impl From<ObsBatch> for Tensor {
        fn from(b: ObsBatch) -> Self {
            b.0.into()
        }
    }

    #[derive(Clone, Debug)]
    pub struct Act(Vec<i32>);

    impl border_core::Act for Act {}

    impl From<Act> for Vec<i32> {
        fn from(value: Act) -> Self {
            value.0
        }
    }

    impl From<Tensor> for Act {
        // `t` must be a 1-dimentional tensor of `f32`
        fn from(t: Tensor) -> Self {
            let data: Vec<i64> = t.into();
            let data = data.iter().map(|&e| e as i32).collect();
            Act(data)
        }
    }

    pub struct ActBatch(TensorSubBatch);

    impl SubBatch for ActBatch {
        fn new(capacity: usize) -> Self {
            Self(TensorSubBatch::new(capacity))
        }

        fn push(&mut self, i: usize, data: Self) {
            self.0.push(i, data.0)
        }

        fn sample(&self, ixs: &Vec<usize>) -> Self {
            let buf = self.0.sample(ixs);
            Self(buf)
        }
    }

    impl From<Act> for ActBatch {
        fn from(act: Act) -> Self {
            let t = vec_to_tensor::<_, i64>(act.0, true);
            Self(TensorSubBatch::from_tensor(t))
        }
    }

    // Required by Dqn
    impl From<ActBatch> for Tensor {
        fn from(act: ActBatch) -> Self {
            act.0.into()
        }
    }

    type PyObsDtype = f32;
    pub type ObsFilter = ArrayObsFilter<PyObsDtype, f32, Obs>;
    pub type ActFilter = DiscreteActFilter<Act>;
    pub type EnvConfig = GymEnvConfig<Obs, Act, ObsFilter, ActFilter>;
    pub type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
    pub type StepProc = SimpleStepProcessor<Env, ObsBatch, ActBatch>;
    pub type ReplayBuffer = SimpleReplayBuffer<ObsBatch, ActBatch>;
    pub type Iqn = Iqn_<Env, Mlp, Mlp, ReplayBuffer>;
    pub type Evaluator = DefaultEvaluator<Env, Iqn>;
}

use obs_act_types::*;

mod config {
    use super::*;

    #[derive(Serialize)]
    pub struct IqnCartpoleConfig {
        pub env_config: EnvConfig,
        pub agent_config: IqnConfig<Mlp, Mlp>,
        pub trainer_config: TrainerConfig,
    }

    impl IqnCartpoleConfig {
        pub fn new(
            in_dim: i64,
            out_dim: i64,
            max_opts: usize,
            model_dir: &str,
            eval_interval: usize,
        ) -> Self {
            let env_config = env_config();
            let agent_config = agent_config(in_dim, out_dim);
            let trainer_config = TrainerConfig::default()
                .max_opts(max_opts)
                .opt_interval(OPT_INTERVAL)
                .eval_interval(eval_interval)
                .record_agent_info_interval(EVAL_INTERVAL)
                .record_compute_cost_interval(EVAL_INTERVAL)
                .flush_record_interval(EVAL_INTERVAL)
                .save_interval(EVAL_INTERVAL)
                .warmup_period(WARMUP_PERIOD)
                .model_dir(model_dir);
            Self {
                env_config,
                agent_config,
                trainer_config,
            }
        }
    }

    pub fn env_config() -> EnvConfig {
        GymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
            .name("CartPole-v0".to_string())
            .obs_filter_config(ObsFilter::default_config())
            .act_filter_config(ActFilter::default_config())
    }

    pub fn agent_config(in_dim: i64, out_dim: i64) -> IqnConfig<Mlp, Mlp> {
        let device = tch::Device::cuda_if_available();
        let opt_config = border_tch_agent::opt::OptimizerConfig::Adam { lr: LR_CRITIC };
        let f_config = MlpConfig::new(in_dim, vec![256], DIM_FEATURE, true);
        let m_config = MlpConfig::new(DIM_FEATURE, vec![256], out_dim, false);
        let model_config = IqnModelConfig::default()
            .feature_dim(DIM_FEATURE)
            .embed_dim(DIM_EMBED)
            .opt_config(opt_config)
            .f_config(f_config)
            .m_config(m_config);

        IqnConfig::default()
            .n_updates_per_opt(N_UPDATES_PER_OPT)
            .batch_size(BATCH_SIZE)
            .discount_factor(DISCOUNT_FACTOR)
            .tau(TAU)
            .model_config(model_config)
            .explorer(EpsilonGreedy::with_params(EPS_START, EPS_FINAL, FINAL_STEP))
            .soft_update_interval(SOFT_UPDATE_INTERVAL)
            .device(device)
    }
}

mod utils {
    use super::*;

    pub fn create_recorder(
        matches: &ArgMatches,
        model_dir: &str,
        config: &config::IqnCartpoleConfig,
    ) -> Result<Box<dyn AggregateRecorder>> {
        match matches.is_present("mlflow") {
            true => {
                let client =
                    MlflowTrackingClient::new("http://localhost:8080").set_experiment_id("Gym")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&config)?;
                recorder_run.set_tag("env", "cartpole")?;
                recorder_run.set_tag("algo", "iqn")?;
                recorder_run.set_tag("backend", "tch")?;
                Ok(Box::new(recorder_run))
            }
            false => Ok(Box::new(TensorboardRecorder::new(model_dir))),
        }
    }

    pub fn create_matches<'a>() -> ArgMatches<'a> {
        App::new("iqn_cartpole_tch")
            .version("0.1.0")
            .author("Taku Yoshioka <yoshioka@laboro.ai>")
            .arg(
                Arg::with_name("train")
                    .long("train")
                    .takes_value(false)
                    .help("Do training only"),
            )
            .arg(
                Arg::with_name("eval")
                    .long("eval")
                    .takes_value(false)
                    .help("Do evaluation only"),
            )
            .arg(
                Arg::with_name("mlflow")
                    .long("mlflow")
                    .takes_value(false)
                    .help("User mlflow tracking"),
            )
            .get_matches()
    }
}

fn train(
    matches: &ArgMatches,
    max_opts: usize,
    model_dir: &str,
    eval_interval: usize,
) -> Result<()> {
    let config =
        config::IqnCartpoleConfig::new(DIM_OBS, DIM_ACT, max_opts, model_dir, eval_interval);
    let mut recorder = utils::create_recorder(&matches, model_dir, &config)?;
    let mut trainer = {
        let step_proc_config = SimpleStepProcessorConfig {};
        let replay_buffer_config =
            SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);

        Trainer::<Env, StepProc, ReplayBuffer>::build(
            config.trainer_config.clone(),
            config.env_config.clone(),
            step_proc_config,
            replay_buffer_config,
        )
    };

    let mut agent = Iqn::build(config.agent_config);
    let mut evaluator = Evaluator::new(&config.env_config, 0, N_EPISODES_PER_EVAL)?;

    trainer.train(&mut agent, &mut recorder, &mut evaluator)?;

    Ok(())
}

fn eval(model_dir: &str, render: bool) -> Result<()> {
    let env_config = {
        let mut env_config = config::env_config();
        if render {
            env_config = env_config
                .render_mode(Some("human".to_string()))
                .set_wait_in_millis(10);
        }
        env_config
    };
    let mut agent = {
        let mut agent = Iqn::build(config::agent_config(DIM_OBS, DIM_ACT));
        agent.load(model_dir)?;
        agent.eval();
        agent
    };
    // let mut recorder = BufferedRecorder::new();

    let _ = Evaluator::new(&env_config, 0, 5)?.evaluate(&mut agent);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tch::manual_seed(42);

    let matches = utils::create_matches();

    if matches.is_present("train") {
        train(&matches, MAX_OPTS, MODEL_DIR, EVAL_INTERVAL)?;
    } else if matches.is_present("eval") {
        eval(&(MODEL_DIR.to_owned() + "/best"), true)?;
    } else {
        train(&matches, MAX_OPTS, MODEL_DIR, EVAL_INTERVAL)?;
        eval(&(MODEL_DIR.to_owned() + "/best"), true)?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::create_matches;

    use super::{eval, train};
    use anyhow::Result;
    use tempdir::TempDir;

    #[test]
    fn test_dqn_cartpole() -> Result<()> {
        let tmp_dir = TempDir::new("iqn_cartpole")?;
        let model_dir = match tmp_dir.as_ref().to_str() {
            Some(s) => s,
            None => panic!("Failed to get string of temporary directory"),
        };
        train(&create_matches(), 100, model_dir, 100)?;
        eval(&(model_dir.to_owned() + "/best"), false)?;
        Ok(())
    }
}
