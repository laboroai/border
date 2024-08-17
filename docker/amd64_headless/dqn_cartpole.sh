sh run.sh "mlflow server --host 127.0.0.1 --port 8080 & \
        sleep 5 && \
        cargo run --release --example dqn_cartpole --features=candle-core,cuda,cudnn \
        -- --train --mlflow > \
        $HOME/border/docker/amd64_headless/log_dqn_cartpole.txt"
