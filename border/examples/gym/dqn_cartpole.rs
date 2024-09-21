use anyhow::Result;
use border_candle_agent::{
    dqn::{Dqn, DqnConfig, DqnModelConfig},
    mlp::{Mlp, MlpConfig},
    opt::OptimizerConfig,
    util::{arrayd_to_tensor, vec_to_tensor, CriticLoss},
    TensorBatch,
};
use border_core::{
    generic_replay_buffer::{
        BatchBase, SimpleReplayBuffer, SimpleReplayBufferConfig, SimpleStepProcessor,
        SimpleStepProcessorConfig,
    },
    record::AggregateRecorder,
    Agent, Configurable, DefaultEvaluator, Env as _, Evaluator as _, ReplayBufferBase,
    StepProcessor, Trainer, TrainerConfig,
};
use border_mlflow_tracking::MlflowTrackingClient;
use border_py_gym_env::{
    ArrayObsFilter, DiscreteActFilter, GymActFilter, GymEnv, GymEnvConfig, GymObsFilter,
};
use border_tensorboard::TensorboardRecorder;
use candle_core::{Device, Tensor};
use clap::Parser;
use ndarray::{ArrayD, IxDyn};
use serde::Serialize;

const DIM_OBS: i64 = 4;
const DIM_ACT: i64 = 2;
const LR_CRITIC: f64 = 0.001;
const DISCOUNT_FACTOR: f64 = 0.99;
const BATCH_SIZE: usize = 64;
const WARMUP_PERIOD: usize = 100;
const N_UPDATES_PER_OPT: usize = 1;
const TAU: f64 = 0.01;
const OPT_INTERVAL: usize = 1;
const MAX_OPTS: usize = 30000;
const EVAL_INTERVAL: usize = 1000;
const REPLAY_BUFFER_CAPACITY: usize = 10000;
const N_EPISODES_PER_EVAL: usize = 5;
const CRITIC_LOSS: CriticLoss = CriticLoss::Mse;
const MODEL_DIR: &str = "./border/examples/gym/model/candle/dqn_cartpole";

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
            arrayd_to_tensor::<_, f32>(obs.0, false).unwrap()
        }
    }

    pub struct ObsBatch(TensorBatch);

    impl BatchBase for ObsBatch {
        fn new(capacity: usize) -> Self {
            Self(TensorBatch::new(capacity))
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
            Self(TensorBatch::from_tensor(tensor))
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
        // `t` must be a 1-dimentional tensor of `i64`
        fn from(t: Tensor) -> Self {
            let data = t.to_vec1::<i64>().expect("Failed to convert Tensor to Act");
            let data = data.iter().map(|&e| e as i32).collect();
            Self(data)
        }
    }

    pub struct ActBatch(TensorBatch);

    impl BatchBase for ActBatch {
        fn new(capacity: usize) -> Self {
            Self(TensorBatch::new(capacity))
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
            let t =
                vec_to_tensor::<_, i64>(act.0, true).expect("Failed to convert Act to ActBatch");
            Self(TensorBatch::from_tensor(t))
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
    pub type Evaluator = DefaultEvaluator<Env, Dqn<Env, Mlp, ReplayBuffer>>;
}

use obs_act_types::*;

mod config {
    use super::*;

    #[derive(Serialize)]
    pub struct DqnCartpoleConfig {
        pub env_config: EnvConfig,
        pub agent_config: DqnConfig<Mlp>,
        pub trainer_config: TrainerConfig,
    }

    impl DqnCartpoleConfig {
        pub fn new(
            in_dim: i64,
            out_dim: i64,
            max_opts: usize,
            model_dir: &str,
            eval_interval: usize,
        ) -> Self {
            let env_config = create_env_config();
            let agent_config = create_agent_config(in_dim, out_dim);
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

    pub fn create_env_config() -> EnvConfig {
        EnvConfig::default()
            .name("CartPole-v0".to_string())
            .obs_filter_config(ObsFilter::default_config())
            .act_filter_config(ActFilter::default_config())
    }

    pub fn create_agent_config(in_dim: i64, out_dim: i64) -> DqnConfig<Mlp> {
        let device = Device::cuda_if_available(0).unwrap();
        let opt_config = OptimizerConfig::default().learning_rate(LR_CRITIC);
        let mlp_config = MlpConfig::new(in_dim, vec![256, 256], out_dim, false);
        let model_config = DqnModelConfig::default()
            .q_config(mlp_config)
            .out_dim(out_dim)
            .opt_config(opt_config);
        DqnConfig::default()
            .n_updates_per_opt(N_UPDATES_PER_OPT)
            .batch_size(BATCH_SIZE)
            .discount_factor(DISCOUNT_FACTOR)
            .tau(TAU)
            .model_config(model_config)
            .device(device)
            .critic_loss(CRITIC_LOSS)
    }
}

use config::{create_agent_config, create_env_config, DqnCartpoleConfig};

mod utils {
    use super::*;

    pub fn create_recorder(
        args: &Args,
        model_dir: &str,
        config: &DqnCartpoleConfig,
    ) -> Result<Box<dyn AggregateRecorder>> {
        match args.mlflow {
            true => {
                let client =
                    MlflowTrackingClient::new("http://localhost:8080").set_experiment_id("Gym")?;
                let recorder_run = client.create_recorder("")?;
                recorder_run.log_params(&config)?;
                recorder_run.set_tag("env", "cartpole")?;
                recorder_run.set_tag("algo", "dqn")?;
                recorder_run.set_tag("backend", "candle")?;
                Ok(Box::new(recorder_run))
            }
            false => Ok(Box::new(TensorboardRecorder::new(model_dir))),
        }
    }
}

/// Train/eval DQN agent in cartpole environment
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Train DQN agent, not evaluate
    #[arg(short, long, default_value_t = false)]
    train: bool,

    /// Evaluate DQN agent, not train
    #[arg(short, long, default_value_t = false)]
    eval: bool,

    /// Log metrics with MLflow
    #[arg(short, long, default_value_t = false)]
    mlflow: bool,
}

fn train(args: &Args, max_opts: usize, model_dir: &str, eval_interval: usize) -> Result<()> {
    let config = DqnCartpoleConfig::new(DIM_OBS, DIM_ACT, max_opts, model_dir, eval_interval);
    let step_proc_config = SimpleStepProcessorConfig {};
    let replay_buffer_config = SimpleReplayBufferConfig::default().capacity(REPLAY_BUFFER_CAPACITY);
    let mut recorder = utils::create_recorder(&args, model_dir, &config)?;
    let mut trainer = Trainer::build(config.trainer_config.clone());

    let env = Env::build(&config.env_config, 0)?;
    let step_proc = StepProc::build(&step_proc_config);
    let mut agent = Dqn::build(config.agent_config);
    let mut buffer = ReplayBuffer::build(&replay_buffer_config);
    let mut evaluator = {
        let env = Env::build(&config.env_config, 0)?;
        Evaluator::new(env, N_EPISODES_PER_EVAL)?
    };

    trainer.train(
        env,
        step_proc,
        &mut agent,
        &mut buffer,
        &mut recorder,
        &mut evaluator,
    )?;

    Ok(())
}

fn eval(model_dir: &str, render: bool) -> Result<()> {
    let env_config = {
        let mut env_config = create_env_config();
        if render {
            env_config = env_config
                .render_mode(Some("human".to_string()))
                .set_wait_in_millis(10);
        }
        env_config
    };
    let mut agent = {
        let mut agent = Dqn::build(create_agent_config(DIM_OBS, DIM_ACT));
        agent.load_params(model_dir)?;
        agent.eval();
        agent
    };

    let _ = {
        let env = Env::build(&env_config, 0)?;
        Evaluator::new(env, 5)?
    }.evaluate(&mut agent);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // TODO: set seed

    let args = Args::parse();

    if args.train {
        train(&args, MAX_OPTS, MODEL_DIR, EVAL_INTERVAL)?;
    } else if args.eval {
        eval(&(MODEL_DIR.to_owned() + "/best"), true)?;
    } else {
        train(&args, MAX_OPTS, MODEL_DIR, EVAL_INTERVAL)?;
        eval(&(MODEL_DIR.to_owned() + "/best"), true)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{eval, train, Args};
    use anyhow::Result;
    use tempdir::TempDir;

    #[test]
    fn test_dqn_cartpole() -> Result<()> {
        let tmp_dir = TempDir::new("dqn_cartpole")?;
        let model_dir = match tmp_dir.as_ref().to_str() {
            Some(s) => s,
            None => panic!("Failed to get string of temporary directory"),
        };
        let args = Args {
            train: false,
            eval: false,
            mlflow: false,
        };
        train(&args, 100, model_dir, 100)?;
        eval(&(model_dir.to_owned() + "/best"), false)?;
        Ok(())
    }
}
