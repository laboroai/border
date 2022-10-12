FROM dorowu/ubuntu-desktop-lxde-vnc:focal-arm64

ENV DEBIAN_FRONTEND noninteractive
RUN echo "Set disable_coredump false" >> /etc/sudo.conf
RUN apt-get update -q && \
    apt-get upgrade -yq && \
    apt-get install -yq wget curl git build-essential vim sudo libssl-dev

# lsb-release locales bash-completion tzdata gosu && \
# RUN rm -rf /var/lib/apt/lists/*

# clang
RUN apt install -y -q libclang-dev

# sdl
RUN apt update -y && \
    apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive && \
    apt install -y -q --no-install-recommends \
        libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-net-dev libsdl2-ttf-dev \
        libsdl-dev libsdl-image1.2-dev

# zip
RUN apt install -y zip

# swig
RUN apt install -y swig

# python
RUN apt install -y python3.8 python3.8-dev python3.8-distutils python3.8-venv python3-pip

# cmake
RUN apt install -y cmake

# headers required for building libtorch
RUN apt install -y libgoogle-glog-dev libgflags-dev

# llvm, mesa for robosuite
RUN apt install -y llvm libosmesa6-dev

# Used for Mujoco
RUN apt install -y patchelf libglfw3 libglfw3-dev

# Cleanup
RUN rm -rf /var/lib/apt/lists/*

COPY test_mujoco_py.py /test_mujoco_py.py
RUN chmod 777 /test_mujoco_py.py

# Add user
RUN useradd --create-home --home-dir /home/ubuntu --shell /bin/bash --user-group --groups adm,sudo ubuntu && \
    echo ubuntu:ubuntu | chpasswd && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
COPY lxterminal.conf /lxterminal.conf
COPY desktop.conf /desktop.conf
RUN chmod 777 /*.conf

# User desktop configs
USER ubuntu
RUN mkdir -p /home/ubuntu/.config/lxterminal && \
    cp /lxterminal.conf /home/ubuntu/.config/lxterminal/
RUN mkdir -p /home/ubuntu/.config/lxsession/LXDE && \
    cp /desktop.conf /home/ubuntu/.config/lxsession/LXDE/

# rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# python
RUN pip3 install --upgrade pip
RUN pip3 install pyyaml typing-extensions
RUN pip3 install torch==1.9.0
RUN pip3 install ipython jupyterlab
RUN pip3 install numpy==1.21.3
# RUN pip3 install gym[box2d]==0.25.1
RUN pip3 install gym[box2d]==0.17.1
RUN pip3 install pybullet==2.7.1
RUN pip3 install robosuite==1.3.2
RUN pip3 install -U 'mujoco-py<2.2,>=2.1'

# # border
# RUN cd $HOME && git clone https://github.com/taku-y/border.git
RUN cd $HOME && mkdir -p .border/model

# # Mujoco aarch64 binary
# RUN cd $HOME && \
#     mkdir .mujoco && \
#     cd .mujoco && \
#     wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-aarch64.tar.gz
# RUN cd $HOME/.mujoco && \
#     tar zxf mujoco-2.1.1-linux-aarch64.tar.gz && \
#     mkdir -p mujoco210/bin && \
#     ln -sf $PWD/mujoco-2.1.1/lib/libmujoco.so.2.1.1 $PWD/mujoco210/bin/libmujoco210.so && \
#     ln -sf $PWD/mujoco-2.1.1/lib/libglewosmesa.so $PWD/mujoco210/bin/libglewosmesa.so && \
#     ln -sf $PWD/mujoco-2.1.1/include/ $PWD/mujoco210/include && \
#     ln -sf $PWD/mujoco-2.1.1/model/ $PWD/mujoco210/model
# RUN cp /test_mujoco_py.py $HOME

# PyBullet
RUN cd $HOME && \
    git clone https://github.com/benelot/pybullet-gym.git && \
    cd pybullet-gym && \

    git checkout -b tmp 79396fccc1d0e170a092b50137148da48ee2edab && \
    pip install -e .

# .bashrc
RUN echo 'export LIBTORCH=$HOME/.local/lib/python3.8/site-packages/torch' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LIBTORCH/lib' >> ~/.bashrc
RUN echo 'export LIBTORCH_CXX11_ABI=0' >> ~/.bashrc
RUN echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
RUN echo 'export PYTHONPATH=$HOME/border/border-py-gym-env/examples:$PYTHONPATH' >> ~/.bashrc
# RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc

USER root
ENV USER ubuntu
