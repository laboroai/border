#!/bin/bash
# nvidia-docker run -it --rm \
#     --env="DISPLAY" \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     --volume="/home/taku-y:/home/taku-y" \
#     --name my_pybullet my_pybullet bash

docker run -td  \
    --name border \
    -p 6080:6080 \
    --shm-size=512m \
    --volume="$(pwd)/../..:/root/border" \
    border
