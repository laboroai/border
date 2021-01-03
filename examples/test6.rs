#[allow(unused_imports)]
use std::error::Error;
#[allow(unused_imports)]
use rayon::prelude::*;
#[allow(unused_imports)]
use lrr::core::{Env, Step};
#[allow(unused_imports)]
use lrr::py_gym_env::{PyGymEnv, PyNDArrayObs, PyGymDiscreteAct};    

#[allow(unused_variables)]
fn main() -> Result<(), Box<dyn Error>> {
    std::env::set_var("RUST_LOG", "trace");
    // let env_name = "CartPole-v0";
    let env_name = "SpaceInvadersNoFrameskip-v4";

    let env = PyGymEnv::<PyGymDiscreteAct>::new(env_name)?;
    // let env1 = PyGymEnv::<PyGymDiscreteAct>::new(env_name)?;
    // let env2 = PyGymEnv::<PyGymDiscreteAct>::new(env_name)?;
    // let env3 = PyGymEnv::<PyGymDiscreteAct>::new(env_name)?;
    // let env4 = PyGymEnv::<PyGymDiscreteAct>::new(env_name)?;
    // let env5 = PyGymEnv::<PyGymDiscreteAct>::new(env_name)?;
    // let env6 = PyGymEnv::<PyGymDiscreteAct>::new(env_name)?;
    // let env7 = PyGymEnv::<PyGymDiscreteAct>::new(env_name)?;
    // let env8 = PyGymEnv::<PyGymDiscreteAct>::new(env_name)?;
    // let envs = vec![env1, env2, env3, env4, env5, env6, env7, env8];

    // for i in 0..400 {
    //     // let obss: Vec<Step<PyGymEnv<PyGymDiscreteAct>>> = envs.par_iter()
    //     let obss: Vec<()> = envs.par_iter()
    //     .map(|env| {
    //         let obs = env.reset();
    //         let _ = env.step(&PyGymDiscreteAct::new(0));
    //     })
    //     .collect();

    //     // if i == 9999 {
    //     //     println!("{:?}", obss[0]);
    //     //     println!("{:?}", obss[1]);
    //     //     println!("{:?}", obss[2]);
    //     //     println!("{:?}", obss[3]);
    //     // }
    // }

    for i in 0..3200 {
        let obs = env.reset();
        let _ = env.step(&PyGymDiscreteAct::new(0));
    }

    Ok(())
}
