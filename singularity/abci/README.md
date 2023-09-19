# ABCI

This directory contains scripts to build and run a singularity container for training on [ABCI](https://abci.ai/).

## Build

```bash
sh build.sh
```

## Run

```bash
qsub -g group [option] dqn_cartpole.sh
```
