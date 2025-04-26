# crates
cargo test -p border-core
cargo test -p border-py-gym-env
cargo test -p border-async-trainer
cargo test -p border-atari-env
cargo test -p border-candle-agent
cargo test -p border-tch-agent
cargo test -p border-policy-no-backend --features=tch

# gym examples
cd examples/gym/dqn_cartpole; cargo test; cd ../../..
cd examples/gym/sac_pendulum; cargo test; cd ../../..
cd examples/gym/sac_fetch_reach; cargo test; cd ../../..
cd examples/gym/dqn_cartpole_tch; cargo test; cd ../../..
cd examples/gym/sac_pendulum_tch; cargo test; cd ../../..

# d4rl examples
cd examples/d4rl/bc_pen; cargo test; cd ../../..
cd examples/d4rl/awac_pen; cargo test; cd ../../..
cd examples/d4rl/iql_pen; cargo test; cd ../../..
