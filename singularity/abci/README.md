# ABCI

This directory contains scripts to build and run a singularity container for training on [ABCI](https://abci.ai/).

## Run training

### Build singularity container image

Build the SIF image in an interactive node. 

```bash
# in $HOME
git clone https://github.com/taku-y/border.git
```

```bash
cd border/singularity/abci
sh build.sh
```

### Run

```bash
cd dqn_cartpole
qsub -g [group_id] dqn_cartpole.sh
```

## Open MLflow in an interactive note

### Portforward

```bash
ssh -i identity_file -L 10022:es:22 -l user_name as.abci.ai
```

```bash
ssh -N -L 8080:host_name:8080 -l user_name -i identity_file -p 10022 localhost
```

### Login interactive node

```bash
ssh -i identity_file -p 10022 -l user_name localhost
```

```bash
qrsh -g group_name -l rt_F=1
```

### Run MLflow server

```bash
module load python/3.10
source venv/bin/activate
mlflow server --host 0.0.0.0 --port 8080
```

Access `localhost:8080` in your browser to show MLflow UI.
