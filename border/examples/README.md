The following directories contain example scripts.

* `gym` - Classic control environments in [Gymnasium](https://gymnasium.farama.org) based on [border-py-gym-env](https://crates.io/crates/border-py-gym-env).
* `gym-robotics` - A robotic environment (fetch-reach) in [Gymnasium-Robotics](https://robotics.farama.org/) based on [border-py-gym-env](https://crates.io/crates/border-py-gym-env).
* `mujoco` - Mujoco environments in [Gymnasium](https://gymnasium.farama.org) based on [border-py-gym-env](https://crates.io/crates/border-py-gym-env).
* `atari` - Atari environments based on [border-atari-env](https://crates.io/crates/border-atari-env) is a wrapper of [atari-env](https://crates.io/crates/atari-env), which is a part of [gym-rs](https://crates.io/crates/gym-rs).

## Gym

You may need to set PYTHONPATH as `PYTHONPATH=./border-py-gym-env/examples`.

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

## Gym, Mujoco

### SAC

```bash
cargo run --example sac_ant --features="tch"
```

```bash
cargo run --example sac_ant_async --features="tch,border-async-trainer"
```

<img src="https://drive.google.com/uc?id=1yTTvfSursj1CfWaF0WyshQC3zghNxE6r" width="512">

<!-- ## PyBullet Env Ant-v0

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

  <img src="https://drive.google.com/uc?id=16TEKfby6twCP6PxYoSlBqzOPEwVk1o4Q" width="256"> -->

<!-- ## Atari (python gym)

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
  ``` -->

## Atari

See atari subdirectory.

* Random policy

  ```bash
  cargo run --example random_pong
  ```

* DQN Pong

  ```bash
  cargo run --release --example dqn_atari_rs --features=tch -- pong
  ```

* DQN Pong Asynchronous trainer

  ```bash
  cargo run --release --example dqn_atari_async --features="tch,border-async-trainer" -- pong
  ```

  <img src="https://drive.google.com/uc?id=1yC3ZWA96GJgNqkWtqgx7DK9_hRH2BlLe" width="512">

* DQN Asterix Asynchronous trainer

  ```bash
  cargo run --release --example dqn_atari_async --features="tch,border-async-trainer" -- asterix
  ```

* DDQN Asterix Asynchronous trainer

  ```bash
  cargo run --release --example dqn_atari_async --features="tch,border-async-trainer" -- asterix --ddqn
  ```

  <img src="https://drive.google.com/uc?id=1ZGiIksX7Ljn6oLp1hIqoYPPSc70hNEJ6" width="512">
