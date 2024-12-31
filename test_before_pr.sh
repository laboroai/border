cargo test -p border-core
cargo test -p border-py-gym-env
cargo test --example dqn_cartpole_tch --features=tch
cargo test --example iqn_cartpole_tch --features=tch
cargo test --example sac_pendulum_tch --features=tch
cargo test --example dqn_cartpole --features=candle-core
cargo test --example sac_pendulum --features=candle-core
cargo test --example convert_sac_policy_to_edge --features="border-tch-agent tch"
cargo test --example pendulum_edge
cd border-async-trainer; cargo test; cd ..
cd border-atari-env; cargo test; cd ..
cd border-candle-agent; cargo test; cd ..
cd border-tch-agent; cargo test; cd ..
cd border-policy-no-backend; cargo test --features=border-tch-agent; cd ..
cd border-py-gym-env; cargo test; cd ..
