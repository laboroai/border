docker run -it --rm \
    --name border_headless \
    --shm-size=512m \
    --volume="$(pwd)/../..:/home/ubuntu/border" \
    border_headless bash -l -c "$@"
