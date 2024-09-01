sh run_detach.sh "
    cd /home/ubuntu/border && \
    source /root/venv/bin/activate && \
    mlflow server --host 0.0.0.0 --port 8080 & \
    sleep 5 && \
    cd /home/ubuntu/border; \
    source /root/venv/bin/activate && \
    LIBTORCH_USE_PYTORCH=1 \
    PYTHONPATH=/home/ubuntu/border/border-py-gym-env/examples \
    cargo run --release --example dqn_cartpole --features=candle-core,cuda,cudnn \
    -- --train --mlflow > \
    $HOME/border/docker/amd64_headless/log_dqn_cartpole.txt 2>&1"
