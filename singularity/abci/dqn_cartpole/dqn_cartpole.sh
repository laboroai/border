#!/bin/bash

#$-l rt_G.small=1
#$-j y
#$-cwd

source $HOME/.bashrc
PATH_TO_BORDER=$HOME/border
source /etc/profile.d/modules.sh
module load singularitypro
cd $PATH_TO_BORDER/singularity/abci
sh run.sh "cargo run --release --example dqn_cartpole --features=candle-core,cuda,cudnn"
