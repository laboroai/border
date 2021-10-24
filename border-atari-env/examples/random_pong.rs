use anyhow::Result;
use border_atari_env::{BorderAtariEnv, BorderAtariAct};
use border_core::Env;

fn main() -> Result<()> {
    let mut env = BorderAtariEnv::default();
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
