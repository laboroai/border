docker run -it --rm \
    --name border_headless \
    --shm-size=512m \
    --volume="$(pwd)/../..:/home/ubuntu/border" \
    border_doc bash -l -c \
    "CARGO_TARGET_DIR=/home/ubuntu/target cargo doc --no-deps --document-private-items; cp -r /home/ubuntu/target/doc ."
