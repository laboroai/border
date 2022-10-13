//! This program is used to quickly check pybullet works properly with pyo3.
use border_py_gym_env::{
    // PyGymEnv, PyGymEnvActFilter, PyGymEnvConfig, PyGymEnvContinuousAct,
    //PyGymEnvContinuousActRawFilter, //PyGymEnvObs, PyGymEnvObsFilter, PyGymEnvObsRawFilter,
};

use anyhow::Result;
use pyo3::{Python, types::IntoPyDict};

fn main() -> Result<()> {
    Python::with_gil(|py| {
        let gym = py.import("gym")?;
        py.import("pybulletgym")?;
        let env = gym.call_method("make", ("AntPyBulletEnv-v0",), None)?;
        let kwargs = vec![("mode", "human")].into_py_dict(py);
        env.call_method("render", (), Some(kwargs))?;

        env.call_method0("reset")?;
        loop {}

        Ok(())
    })
}
