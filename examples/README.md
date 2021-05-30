## Atari

### DQN

* Training

  ```bash
  PYTHONPATH=./examples cargo run --release --example dqn_atari -- PongNoFrameskip-v4
  ```

* Evaluation

  ```bash
  PYTHONPATH=./examples cargo run --example dqn_atari -- PongNoFrameskip-v4 --play ./examples/model/dqn_PongNoFrameskip-v4
  ```

* Evaluation with pretrained models

  ```bash
  PYTHONPATH=./examples cargo run --example dqn_atari -- PongNoFrameskip-v4 --play-gdrive
  ```

### IQN

* Evaluation with pretrained models

  ```bash
  PYTHONPATH=./examples cargo run --example iqn_atari -- PongNoFrameskip-v4 --play-gdrive
  ```

  ```bash
  PYTHONPATH=./examples cargo run --example iqn_atari -- SeaquestNoFrameskip-v4 --play-gdrive
  ```

## PyBullet Env Ant-v0

* Training

  ```bash
  $ cargo run --example sac_ant
  ```

  <img src="https://drive.google.com/uc?id=16TEKfby6twCP6PxYoSlBqzOPEwVk1o4Q" width="256">

* Testing

  ```bash
  $ cargo run --example sac_ant -- --play=$REPO/examples/model/sac_ant
  ```

* Testing with downloading a pretrained model

  ```bash
  $ cargo run --example sac_ant -- --play-gdrive
  ```

You can download a pretrained model from [here](https://drive.google.com/uc?export=download&id=1fdAVJLgFY2v0BDyE-xGt7mxpa8GXa9aX).

## FreewayNoFrameskip-v4

<img src="https://drive.google.com/uc?export=view&id=1KUXN4GpL_lrwNJ4synSH9P1ROT3ljVAD" width="256">
