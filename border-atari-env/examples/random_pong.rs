use anyhow::Result;
use border_atari_env::BorderAtariEnv;

fn main() -> Result<()> {
    let mut env = BorderAtariEnv::default();
    // env.open()?;
    let mut i = 0;

    loop {
        env.step();
        println!("{}", i);
        i += 1;
    }
}
