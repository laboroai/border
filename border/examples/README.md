## Gym

### Random policy

```bash
cargo run --example random_cartpole
```

```bash
cargo run --example random_lunarlander_cont
```

```bash
cargo run --example random_ant
```

### DQN

```bash
cargo run --example dqn_cartpole --features="tch"
```

### SAC

```bash
cargo run --example sac_pendulum --features="tch"
```

```bash
cargo run --example sac_lunarlander_cont --features="tch"
```

## Atari

### DQN

* Training

  ```bash
  PYTHONPATH=./border/examples cargo run --release --example dqn_atari -- PongNoFrameskip-v4
  ```

* Evaluation

  ```bash
  PYTHONPATH=./border/examples cargo run --example dqn_atari -- PongNoFrameskip-v4 --play ./examples/model/dqn_PongNoFrameskip-v4
  ```

* Evaluation with pretrained models, downloaded from google drive

  ```bash
  PYTHONPATH=./border/examples cargo run --example dqn_atari -- PongNoFrameskip-v4 --play-gdrive
  ```

### IQN

* Evaluation with pretrained models

  ```bash
  PYTHONPATH=./border/examples cargo run --example iqn_atari -- PongNoFrameskip-v4 --play-gdrive
  ```

  ```bash
  PYTHONPATH=./border/examples cargo run --example iqn_atari -- SeaquestNoFrameskip-v4 --play-gdrive
  ```

## PyBullet Env Ant-v0

* Training

  ```bash
  $ cargo run --example sac_ant
  ```

  <img src="https://drive.google.com/uc?id=16TEKfby6twCP6PxYoSlBqzOPEwVk1o4Q" width="256">

* Evaluation

  ```bash
  cargo run --example sac_ant -- --play=$./border/examples/model/sac_ant
  ```

* Evaluation with a pretrained model, downloaded from google drive

  ```bash
  cargo run --example sac_ant -- --play-gdrive
  ```
