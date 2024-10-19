#!/bin/bash

#$-l rt_C.small=1
#$-l h_rt=01:00:00
#$-j y
#$-cwd

source $HOME/.bashrc
PATH_TO_BORDER=$HOME/border
source /etc/profile.d/modules.sh
module load singularitypro
cd $PATH_TO_BORDER/singularity/abci
sh run.sh "hostname & mlflow server --host 0.0.0.0 --port 8080 & sleep 3600"
