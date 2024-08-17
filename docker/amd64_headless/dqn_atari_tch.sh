rm -f log_dqn_atari_tch.txt
sh run_detach.sh "
    source /root/venv/bin/activate && \
    pip3 install autorom && \
    mkdir $HOME/atari_rom && \
    AutoROM --install-dir $HOME/atari_rom --accept-license && \
    cd /home/ubuntu/border && \
    mlflow server --host 0.0.0.0 --port 8080 & \
    sleep 5 && \
    cd /home/ubuntu/border; \
    source /root/venv/bin/activate && \
    LIBTORCH_USE_PYTORCH=1 \
    LD_LIBRARY_PATH=/root/venv/lib/python3.10/site-packages/torch/lib \
    ATARI_ROM_DIR=$HOME/atari_rom \
    cargo run --release --example dqn_atari_tch --features=tch \
    -- ${1} --mlflow > \
    $HOME/border/docker/amd64_headless/log_dqn_atari_tch_${1}.txt 2>&1"
