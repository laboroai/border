## Gym

You need to set PYTHONPATH as `PYTHONPATH=./border-py-gym-env/examples`.

### DQN

```bash
cargo run --example dqn_cartpole_tch --features="tch"
```

```bash
cargo run --example dqn_cartpole --features="candle-core,cuda,cudnn"
```

### SAC

```bash
cargo run --example sac_pendulum_tch --features="tch"
```

```bash
cargo run --example sac_lunarlander_cont_tch --features="tch"
```

```bash
cargo run --example sac_pendulum --features="candle-core,cuda,cudnn"
```

```bash
cargo run --example sac_lunarlander_cont --features="candle-core,cuda,cudnn"
```

### border-policy-no-backend

`convert_sac_policy_to_edge` converts model parameters obtained with `sac_pendulum_tch`.

```bash
cargo run --example convert_sac_policy_to_edge
```

The converted model parameters can be used with border-policy-no-backend crate.

```bash
cargo run --example pendulum_edge
```
