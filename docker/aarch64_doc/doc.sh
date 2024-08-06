docker run -it --rm \
    --name border_headless \
    --shm-size=512m \
    --volume="$(pwd)/../..:/home/ubuntu/border" \
    border_headless bash -l -c \
    "cd /home/ubuntu/border; source /root/venv/bin/activate; LIBTORCH_USE_PYTORCH=1 LD_LIBRARY_PATH=$HOME/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH CARGO_TARGET_DIR=/home/ubuntu/target cargo doc --no-deps --document-private-items; cp -r /home/ubuntu/target/doc ."
