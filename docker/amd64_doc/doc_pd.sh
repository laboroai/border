podman run -it --rm \
    --name border_doc \
    --shm-size=512m \
    --volume="$(pwd)/../..:/home/ubuntu/border" \
    border_doc bash -l -c \
    "cd /home/ubuntu/border; source /root/venv/bin/activate; LIBTORCH_USE_PYTORCH=1 CARGO_TARGET_DIR=/home/ubuntu/target cargo doc --no-deps --document-private-items; cp -r /home/ubuntu/target/doc ."
