# Docker container for training

This directory contains scripts to build and run a docker container for training.

## Build a Docker image

```bash
# cd $REPO/docker/aarch64_headless
sh build.sh
```

## Run training

The following commands runs a program for training an agent.
The trained model will be saved in `$REPO/border/examples/model` directory.

### DQN

* Cartpole

  Note that this command starts an MLflow server, accessible via a web browser at $IP:8080 without any authentication.

  ```bash
  # cd $REPO/docker/amd64_headless
  sh dqn_cartpole.sh
  ```

## Start MLflow server for checking logs

```bash
# cd $REPO/docker/amd64_headless
sh mlflow.sh
```
