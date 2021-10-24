use anyhow::Result;
use border_atari_env::{BorderAtariEnv, BorderAtariEnvConfig};
use border_core::Env;

fn main() -> Result<()> {
    let env_config = BorderAtariEnvConfig::default()
        .name("pong");
    let mut env = BorderAtariEnv::build(&env_config, 42)?;
    env.open()?;
    // let mut i = 0;

    loop {
        let n_actions = env.get_num_actions_atari();
        let ix = fastrand::u8(..n_actions as u8);
        let (step, _) = env.step(&ix.into());
        if step.is_done[0] == 1 {
            env.reset(None).unwrap();
        }
        // println!("{}", i);
        // i += 1;
    }
}
