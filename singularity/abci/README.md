# ABCI

This directory contains scripts to build and run a singularity container for training on [ABCI](https://abci.ai/).

## Preparation

### Login to the interactive node

```bash
# Login to the access server
ssh -i $ABCI_IDENTITY_FILE -L 10022:es:22 -l $ABCI_USER_NAME as.abci.ai
```

```bash
# Login to the interactive node
ssh -i $ABCI_IDENTITY_FILE -p 10022 -l $ABCI_USER_NAME localhost
```

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

### Install MLflow (optional)

```bash
cd $HOME
load module python/3.10
python3 -m venv venv
source venv/bin/activate
pip3 install mlflow
```

### Install AutoROM (optional)

```bash
cd $HOME
source venv/bin/activate
pip3 install autorom
mkdir atari_rom
AutoROM --install-dir atari_rom
```

## Run training

### Submit training job

```bash
cd dqn_cartpole
qsub -g [group_id] dqn_cartpole.sh
```

## Open MLflow in an interactive note

### Login to the compute node

```bash
qrsh -g $ABCI_GROUP_NAME -l rt_F=1
```

### Run MLflow server

```bash
module load python/3.10
source venv/bin/activate
cd border
mlflow server --host 0.0.0.0 --port 8080
```

### Portforward

```bash
ssh -N -L 8080:host_name:8080 -l $ABCI_USER_NAME -i $ABCI_IDENTITY_FILE -p 10022 localhost
```

Access `localhost:8080` in your browser to show MLflow UI.
