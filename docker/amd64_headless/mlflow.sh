sh run.sh "
    cd /home/ubuntu/border && \
    source /root/venv/bin/activate && \
    mlflow server --host 0.0.0.0 --port 8080
"