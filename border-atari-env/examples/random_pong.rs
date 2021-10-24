use anyhow::Result;
use border_atari_env::{BorderAtariObs, BorderAtariAct, BorderAtariEnv, BorderAtariEnvConfig};
use border_core::{Policy, Env as _, record::BufferedRecorder, util};

type Obs = BorderAtariObs;
type Act = BorderAtariAct;
type Env = BorderAtariEnv;

struct RandomPolicy {
    n_acts: usize,
}

impl RandomPolicy {
    fn new(n_acts: usize) -> Self {
        Self { n_acts }
    }
}

impl Policy<Env> for RandomPolicy {
    fn sample(&mut self, _: &Obs) -> Act {
        fastrand::u8(..self.n_acts as u8).into()
    }
}

fn env_config(name: String) -> BorderAtariEnvConfig {
    BorderAtariEnvConfig::default()
        .name(name)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    fastrand::seed(42);

    let env_config = env_config("pong".to_string());
    let mut env = Env::build(&env_config, 42)?;
    let mut recorder = BufferedRecorder::new();
    let n_acts = env.get_num_actions_atari();
    let mut policy = RandomPolicy::new(n_acts as _);

    env.open()?;
   let _ = util::eval_with_recorder(&mut env, &mut policy, 5, &mut recorder)?;

   Ok(())
}
