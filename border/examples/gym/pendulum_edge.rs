use anyhow::Result;
use border_core::{DefaultEvaluator, Evaluator as _};
use border_policy_no_backend::{Mat, Mlp};
use border_py_gym_env::{
    ArrayObsFilter, ContinuousActFilter, GymActFilter, GymEnv, GymEnvConfig, GymObsFilter,
};
use clap::Parser;
use ndarray::ArrayD;
use std::fs;

type PyObsDtype = f32;

mod obs_act_types {
    use super::*;

    #[derive(Clone, Debug)]
    /// Observation type.
    pub struct Obs(Mat);

    impl border_core::Obs for Obs {
        fn len(&self) -> usize {
            self.0.shape()[0] as _
        }
    }

    impl From<ArrayD<f32>> for Obs {
        fn from(obs: ArrayD<f32>) -> Self {
            let obs = obs.t().to_owned();
            let shape = obs.shape().iter().map(|e| *e as i32).collect();
            let data = obs.into_raw_vec();
            Self(Mat::new(data, shape))
        }
    }

    impl From<Obs> for Mat {
        fn from(obs: Obs) -> Mat {
            obs.0
        }
    }

    #[derive(Clone, Debug)]
    pub struct Act(Mat);

    impl border_core::Act for Act {}

    impl From<Act> for ArrayD<f32> {
        fn from(value: Act) -> Self {
            // let shape: Vec<_> = value.0.shape.iter().map(|e| *e as usize).collect();
            let shape = vec![(value.0.shape[0] * value.0.shape[1]) as usize];
            // let data = value.0.data;
            let data: Vec<f32> = value.0.data.iter().map(|e| 2f32 * *e).collect();
            let t = ArrayD::from_shape_vec(shape, data).unwrap();
            t
        }
    }

    impl Into<Act> for Mat {
        fn into(self) -> Act {
            Act(self)
        }
    }
}

mod policy {
    use std::{io::Read, path::Path};

    use super::*;
    use border_core::Policy;

    pub struct MlpPolicy {
        mlp: Mlp,
    }

    impl Policy<Env> for MlpPolicy {
        fn sample(&mut self, obs: &Obs) -> Act {
            self.mlp.forward(&obs.clone().into()).into()
        }
    }

    impl MlpPolicy {
        pub fn from_serialized_path(path: impl AsRef<Path>) -> Result<Self> {
            let mut file = fs::OpenOptions::new().read(true).open(&path)?;
            let mut buf = Vec::<u8>::new();
            let _ = file.read_to_end(&mut buf).unwrap();
            let mlp: Mlp = bincode::deserialize(&buf[..])?;
            Ok(Self { mlp })
        }
    }
}

use obs_act_types::*;
use policy::*;

type ObsFilter = ArrayObsFilter<PyObsDtype, f32, Obs>;
type ActFilter = ContinuousActFilter<Act>;
type Env = GymEnv<Obs, Act, ObsFilter, ActFilter>;
type Evaluator = DefaultEvaluator<Env, MlpPolicy>;

fn env_config() -> GymEnvConfig<Obs, Act, ObsFilter, ActFilter> {
    GymEnvConfig::<Obs, Act, ObsFilter, ActFilter>::default()
        .name("Pendulum-v1".to_string())
        .obs_filter_config(ObsFilter::default_config())
        .act_filter_config(ActFilter::default_config())
}

fn eval(n_episodes: usize, render: bool) -> Result<()> {
    let env_config = {
        let mut env_config = env_config();
        if render {
            env_config = env_config
                .render_mode(Some("human".to_string()))
                .set_wait_in_millis(10);
        };
        env_config
    };
    let mut policy = MlpPolicy::from_serialized_path(
        "./border/examples/gym/model/edge/sac_pendulum/best/mlp.bincode",
    )?;

    let _ = {
        let env = env_config.build()?;
        Evaluator::new(env, n_episodes)?
    }
    .evaluate(&mut policy);

    Ok(())
}

/// Train/eval SAC agent in pendulum environment
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Train SAC agent, not evaluate
    #[arg(short, long, default_value_t = false)]
    train: bool,

    /// Evaluate SAC agent, not train
    #[arg(short, long, default_value_t = false)]
    eval: bool,

    /// Log metrics with MLflow
    #[arg(short, long, default_value_t = false)]
    mlflow: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let _ = eval(5, true)?;

    // let args = Args::parse();

    // if args.train {
    //     train(
    //         MAX_OPTS,
    //         "./border/examples/gym/model/tch/sac_pendulum",
    //         EVAL_INTERVAL,
    //         args.mlflow,
    //     )?;
    // } else if args.eval {
    //     eval(5, true, "./border/examples/gym/model/tch/sac_pendulum/best")?;
    // } else {
    //     train(
    //         MAX_OPTS,
    //         "./border/examples/gym/model/tch/sac_pendulum",
    //         EVAL_INTERVAL,
    //         args.mlflow,
    //     )?;
    //     eval(5, true, "./border/examples/gym/model/tch/sac_pendulum/best")?;
    // }

    Ok(())
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use tempdir::TempDir;

//     #[test]
//     fn test_sac_pendulum() -> Result<()> {
//         tch::manual_seed(42);

//         let model_dir = TempDir::new("sac_pendulum_tch")?;
//         let model_dir = model_dir.path().to_str().unwrap();
//         train(100, model_dir, 100, false)?;
//         eval(1, false, (model_dir.to_string() + "/best").as_str())?;

//         Ok(())
//     }
// }
