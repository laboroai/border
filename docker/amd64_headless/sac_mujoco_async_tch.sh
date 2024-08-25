rm -f log_sac_mujoco_async_tch_${1}.txt
sh run_detach.sh "
    source /root/venv/bin/activate && \
    cd /home/ubuntu/border && \
    mlflow server --host 0.0.0.0 --port 8080 & \
    sleep 5 && \
    cd /home/ubuntu/border; \
    source /root/venv/bin/activate && \
    LIBTORCH_USE_PYTORCH=1 \
    LD_LIBRARY_PATH=/root/venv/lib/python3.10/site-packages/torch/lib \
    PYTHONPATH=/home/ubuntu/border/border-py-gym-env/examples \
    cargo run --release --example sac_mujoco_async_tch --features=tch,border-async-trainer \
    -- --env ${1} --mlflow --n-actors 6 > \
    $HOME/border/docker/amd64_headless/log_sac_mujoco_async_tch_${1}.txt 2>&1"
