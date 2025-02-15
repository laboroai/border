cargo test -p border-core
cargo test -p border-py-gym-env
cargo test -p border-async-trainer
cargo test -p border-atari-env
cargo test -p border-candle-agent
cargo test -p border-tch-agent
cargo test -p border-policy-no-backend --features=border-tch-agent
cd examples/gym/dqn_cartpole; cargo test; cd ../../..
cd examples/gym/sac_pendulum; cargo test; cd ../../..
cd examples/gym/sac_fetch_reach; cargo test; cd ../../..
cd examples/gym/dqn_cartpole_tch; cargo test; cd ../../..

# cargo test --example dqn_cartpole_tch --features=tch
# cargo test --example iqn_cartpole_tch --features=tch
# cargo test --example sac_pendulum_tch --features=tch
# cargo test --example convert_sac_policy_to_edge --features="border-tch-agent tch"
# cargo test --example pendulum_edge
