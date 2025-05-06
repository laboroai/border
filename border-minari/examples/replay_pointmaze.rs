use anyhow::Result;
use border_core::Env;
use border_minari::{d4rl::pointmaze::ndarray::PointMazeConverter, MinariDataset};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let dataset = MinariDataset::load_dataset("D4RL/pointmaze/umaze-v2", true)?;

    // The number of transitions over all episodes
    let num_transitions = dataset.get_num_transitions(None)?;
    println!("{:?}", num_transitions);

    // Converter for observation and action
    let mut converter = PointMazeConverter {};

    // Create replay buffer for the sixth episode
    let replay_buffer = dataset.create_replay_buffer(&mut converter, Some(vec![1]))?;

    // Recover the environment from the dataset
    let mut env = dataset.recover_environment(converter, true, "human")?;

    // Sequence of actions in the episode
    let actions = replay_buffer.whole_actions();

    // Apply the actions to the environment
    env.reset(None)?;

    for ix in 0..actions.action.shape()[0] {
        let act = actions.get(ix);
        let _ = env.step(&act);
    }

    Ok(())
}
