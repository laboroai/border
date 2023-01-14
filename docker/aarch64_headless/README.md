# Docker container for training

This directory contains scripts to build and run a docker container for training.

## Build

The following command creates a container image locally, named `border_headless`.

```bash
cd $REPO/docker/aarch64_headless
sh build.sh
```

# Build document

The following commands builds the document and places it as `$REPO/doc`.

## Run

The following commands runs a program for training an agent.
The trained model will be saved in `$REPO/border/examples/model` directory,
which is mounted in the container.

### DQN

* Cartpole

  ```bash
  cd $REPO/docker/aarch64_headless
  sh run.sh "source /home/ubuntu/venv/bin/activate && cargo run --example dqn_cartpole --features='tch'"
  ```

  * Use a directory, not mounted on the host, as a cargo target directory,
    making compile faster on Mac, where access to mounted directories is slow.

    ```bash
    cd $REPO/docker/aarch64_headless
    sh run.sh "source /home/ubuntu/venv/bin/activate && CARGO_TARGET_DIR=/home/ubuntu/target cargo run --example dqn_cartpole --features='tch'"
    ```
