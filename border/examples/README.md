## Gym

### Random policy

```bash
PYTHONPATH=./border-py-gym-env/examples cargo run --example random_cartpole
```

```bash
PYTHONPATH=./border-py-gym-env/examples cargo run --example random_lunarlander_cont
```

```bash
PYTHONPATH=./border-py-gym-env/examples cargo run --example random_ant
```

### DQN

```bash
PYTHONPATH=./border-py-gym-env/examples cargo run --example dqn_cartpole --features="tch"
```

### SAC

```bash
PYTHONPATH=./border-py-gym-env/examples cargo run --example sac_pendulum --features="tch"
```

```bash
PYTHONPATH=./border-py-gym-env/examples cargo run --example sac_lunarlander_cont --features="tch"
```

## PyBullet Env Ant-v0

* Training

  ```bash
  PYTHONPATH=./border-py-gym-env/examples cargo run --release --example sac_ant --features="tch"
  ```

* Evaluation

  ```bash
  PYTHONPATH=./border-py-gym-env/examples cargo run --example sac_ant --features="tch" -- --play=$./border/examples/model/sac_ant
  ```

* Evaluation with a pretrained model, downloaded from google drive

  ```bash
  PYTHONPATH=./border-py-gym-env/examples cargo run --example sac_ant --features="tch" -- --play-gdrive
  ```

  <img src="https://drive.google.com/uc?id=16TEKfby6twCP6PxYoSlBqzOPEwVk1o4Q" width="256">

## Atari (python gym)

### DQN

* Pong training, evaluation, evaluation with a pretrained model downloaded from google drive

  ```bash
  PYTHONPATH=./border-py-gym-env/examples cargo run --release --example dqn_atari --features="tch" -- PongNoFrameskip-v4
  ```

  ```bash
  PYTHONPATH=./border-py-gym-env/examples cargo run --release --example dqn_atari --features="tch" -- PongNoFrameskip-v4 --play ./examples/model/dqn_PongNoFrameskip-v4
  ```

  ```bash
  PYTHONPATH=./border-py-gym-env/examples cargo run --release --example dqn_atari --features="tch" -- PongNoFrameskip-v4 --play-gdrive
  ```

### IQN

* Evaluation with pretrained models

  ```bash
  PYTHONPATH=./border-py-gym-env/examples cargo run --example iqn_atari --features="tch" -- PongNoFrameskip-v4 --play-gdrive
  ```

  ```bash
  PYTHONPATH=./border-py-gym-env/examples cargo run --example iqn_atari --features="tch" -- SeaquestNoFrameskip-v4 --play-gdrive
  ```

## Atari (not python)

### Random policy

```bash
cargo run --example random_pong
```
