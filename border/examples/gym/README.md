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
