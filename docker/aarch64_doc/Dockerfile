FROM ubuntu:focal-20221130

ENV DEBIAN_FRONTEND noninteractive
RUN echo "Set disable_coredump false" >> /etc/sudo.conf
RUN apt-get update -q && \
    apt-get upgrade -yq && \
    apt-get install -yq wget curl git build-essential vim sudo libssl-dev
RUN apt install -y -q libclang-dev zip cmake llvm pkg-config libx11-dev libxkbcommon-dev

# # swig
# RUN apt install -y swig

# # python
# RUN apt install -y python3.8 python3.8-dev python3.8-distutils python3.8-venv python3-pip

# # headers required for building libtorch
# RUN apt install -y libgoogle-glog-dev libgflags-dev

# # llvm, mesa for robosuite
# RUN apt install -y llvm libosmesa6-dev

# # Used for Mujoco
# RUN apt install -y patchelf libglfw3 libglfw3-dev

# Cleanup
RUN rm -rf /var/lib/apt/lists/*

# COPY test_mujoco_py.py /test_mujoco_py.py
# RUN chmod 777 /test_mujoco_py.py

# Add user
RUN useradd --create-home --home-dir /home/ubuntu --shell /bin/bash --user-group --groups adm,sudo ubuntu && \
    echo ubuntu:ubuntu | chpasswd && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Use bash
RUN mv /bin/sh /bin/sh_tmp && ln -s /bin/bash /bin/sh

# User settings
USER ubuntu

# rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc

USER root
RUN rm /bin/sh && mv /bin/sh_tmp /bin/sh

USER ubuntu
WORKDIR /home/ubuntu/border
