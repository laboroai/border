# Mujoco environment

This directory contains examples using Mujoco environments.

## tch agent

```bash
cargo run --release --example sac_mujoco_tch --features=tch -- --env ant --mlflow
```

`env` option can be `ant`, `cheetah`, `walker`, or `hopper`.

## candle agent

```bash
cargo run --release --example sac_mujoco --features=candle-core,cuda,cudnn -- --env ant --mlflow
```
