# DQN on cartpole environment

## Tensorboard

The model parameters and TFRecords will be saved in `./model` directory.

```bash
cargo run --release
```

## MLflow tracking

Before executing the below command, you may run a MLflow tracking server at `$REPO/mlruns`.
The model parameters will be saved in the directory coresponding to the MLflow run id
under the `$REPO/mlruns` directory.

```bash
export MLFLOW_DEFAULT_ARTIFACT_ROOT=$REPO/mlruns
cargo run --release -- --mlflow
```
