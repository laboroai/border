use anyhow::Result;
use border_core::{DefaultEvaluator, Evaluator as _};
use border_policy_no_backend::{Mat, Mlp};
use border_py_gym_env::{util::pyobj_to_arrayd, GymEnv, GymEnvConfig, GymEnvConverter};
use ndarray::ArrayD;
use numpy::PyArrayDyn;
use pyo3::{IntoPy, PyObject};
use std::fs;

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
    /// Action type.
    pub struct Act(Mat);

    impl border_core::Act for Act {}

    impl From<Act> for ArrayD<f32> {
        fn from(value: Act) -> Self {
            // let shape: Vec<_> = value.0.shape.iter().map(|e| *e as usize).collect();
            let shape = vec![(value.0.shape[0] * value.0.shape[1]) as usize];
            // let data = value.0.data;
            // let data: Vec<f32> = value.0.data.iter().map(|e| 2f32 * *e).collect();
            let data = value.0.data;
            let t = ArrayD::from_shape_vec(shape, data).unwrap();
            t
        }
    }

    impl Into<Act> for Mat {
        fn into(self) -> Act {
            Act(self)
        }
    }

    #[derive(Clone, Debug)]
    /// Converter for bridging env and obs/act types.
    pub struct MatConverter {}

    impl GymEnvConverter for MatConverter {
        type Obs = Obs;
        type Act = Act;
        type Config = ();

        fn new(_config: &Self::Config) -> Result<Self> {
            let converter = Self {};
            Ok(converter)
        }

        /// Convert observation.
        ///
        /// Data type should be f32.
        fn filt_obs(&mut self, obs: PyObject) -> Result<Self::Obs> {
            // ndarray
            let obs = pyo3::Python::with_gil(|py| {
                if obs.as_ref(py).get_type().name().unwrap() == "NoneType" {
                    panic!();
                } else {
                    pyobj_to_arrayd::<f32, f32>(obs)
                }
            });

            Ok(Obs::from(obs))
        }

        /// Convert [`Act`] to [`PyObject`].
        fn filt_act(&mut self, act: Self::Act) -> Result<PyObject> {
            let arrayd = act.into();
            let pyobj = pyo3::Python::with_gil(|py| {
                let act = PyArrayDyn::<f32>::from_array(py, &arrayd);
                act.into_py(py)
            });
            Ok(pyobj)
        }
    }

    pub type Env = GymEnv<MatConverter>;
    pub type Evaluator = DefaultEvaluator<Env>;
}

use obs_act_types::*;

mod agent {
    use std::{io::Read, path::Path};

    use super::*;
    use border_core::{Agent, NullReplayBuffer, Policy};

    pub struct MlpAgent {
        mlp: Mlp,
    }

    impl Policy<Env> for MlpAgent {
        fn sample(&mut self, obs: &Obs) -> Act {
            self.mlp.forward(&obs.clone().into()).into()
        }
    }

    impl MlpAgent {
        pub fn from_serialized_path(path: impl AsRef<Path>) -> Result<Self> {
            let mut file = fs::OpenOptions::new().read(true).open(&path)?;
            let mut buf = Vec::<u8>::new();
            let _ = file.read_to_end(&mut buf).unwrap();
            let mlp: Mlp = bincode::deserialize(&buf[..])?;
            Ok(Self { mlp })
        }
    }

    impl Agent<Env, NullReplayBuffer> for MlpAgent {}
}

use agent::*;

fn create_env_config(render: bool) -> Result<GymEnvConfig<MatConverter>> {
    let mut env_config = GymEnvConfig::default()
        .name("Pendulum-v1".to_string())
        .converter_config(());
    if render {
        env_config = env_config
            .render_mode(Some("human".to_string()))
            .set_wait_in_millis(10);
    };
    Ok(env_config)
}

fn eval(path: &str, n_episodes: usize, render: bool) -> Result<()> {
    let env_config = create_env_config(render)?;
    let mut policy = Box::new(MlpAgent::from_serialized_path(path)?) as _;
    let _ = Evaluator::new(&env_config, 0, n_episodes)?.evaluate(&mut policy);

    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let path = "../convert_policy/model/mlp.bincode";
    let _ = eval(path, 5, true)?;
    Ok(())
}

// #[test]
// fn test_pendulum_edge() -> Result<()> {
//     env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
//     let path = "/root/border/examples/convert_policy/model/mlp.bincode";
//     let _ = eval(path, 1, false)?;
//     Ok(())
// }
