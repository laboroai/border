# Atari environment

This directory contains examples using Atari environments.

## tch agent

```bash
cargo run --release --example dqn_atari_tch --features=tch -- pong --mlflow
```

## candle agent

```bash
cargo run --release --example dqn_atari --features=candle-core,cuda,cudnn -- pong --mlflow
```
