docker run -td  \
    --name border \
    -p 6080:6080 \
    --shm-size=512m \
    --volume="$(pwd)/../..:/root/border" \
    border
