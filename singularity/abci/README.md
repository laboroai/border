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
qsub -g $ABCI_GROUP_ID dqn_cartpole.sh
```

## Open MLflow UI in local

When you would like to see the mlflow log,
run the following command in an interactive node.

```bash
cd mlflow
qsub -g $ABCI_GROUP_ID mlflow.sh
```

The script submits commands into an compute node. The submitted commands
are to show the name of host and start a mlflow tracking server. 
Several seconds after starting the script, a log file will be created
in `mlflow` directory. In that log file, the name of the host, where the
server is running, is included. The host name will be used for
port forwarding.

In another local terminal, run the following commands:

```bash
export MLFLOW_HOST_NAME=host_name
ssh -N -L 8080:$MLFLOW_HOST_NAME:8080 -l $ABCI_USER_NAME -i $ABCI_IDENTITY_FILE -p 10022 localhost
```

`host_name` is the name of the host where the mlflow tracking server is
running. Accessing `localhost:8080` with your browser, you can see the
mlflow tracking server UI.
