# Docker container for RL training

This directory contains scripts to build and run a docker container for RL training.

## Build

```bash
cd $REPO/docker/aarch64_headless
sh build.sh
```

## Run

### DQN

* Cartpole

  ```bash
  cd $REPO/docker/aarch64_headless
  sh run.sh "cargo run --example dqn_cartpole --features='tch'"
  ```

  * Use a directory not mounted on the host as a cargo target directory,
    making compile faster on Mac, where access to mounted directories is slow.

    ```bash
    cd $REPO/docker/aarch64_headless
    sh run.sh "CARGO_TARGET_DIR=/home/ubuntu/target cargo run --example dqn_cartpole --features='tch'"
    ```
