# ABCI

This directory contains scripts to build and run a singularity container for training on [ABCI](https://abci.ai/).

## Build

Build the SIF image in an interactive node. 

```bash
# in $HOME
git clone https://github.com/taku-y/border.git
```

```bash
cd border/singularity/abci
sh build.sh
```

## Run

```bash
cd dqn_cartpole
qsub -g [group_id] dqn_cartpole.sh
```
