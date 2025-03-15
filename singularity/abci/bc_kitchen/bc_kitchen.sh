#!/bin/bash

#$-l rt_G.small=1
#$-l h_rt=24:00:00
#$-j y
#$-cwd

source $HOME/.bashrc
PATH_TO_BORDER=$HOME/border
source /etc/profile.d/modules.sh
module load singularitypro
cd $PATH_TO_BORDER/singularity/abci
sh run.sh "mlflow server --host 127.0.0.1 --port 8080 & \
        sleep 5 && \
        cargo run --release --example bc_kitchen --features=candle,cuda,cudnn -- --eval-episodes 100 --mlflow"
