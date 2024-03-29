FROM dorowu/ubuntu-desktop-lxde-vnc:focal

ENV DEBIAN_FRONTEND noninteractive
RUN echo "Set disable_coredump false" >> /etc/sudo.conf
RUN sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 4EB27DB2A3B88B8B
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
COPY test_dmc_viewer.py /test_dmc_viewer.py
RUN chmod 777 /test_mujoco_py.py
RUN chmod 777 /test_dmc_viewer.py

# Use bash
RUN mv /bin/sh /bin/sh_tmp && ln -s /bin/bash /bin/sh

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
RUN cd /home/ubuntu && python -m venv venv
RUN source /home/ubuntu/venv/bin/activate && pip3 install --upgrade pip
RUN source /home/ubuntu/venv/bin/activate && pip3 install pyyaml typing-extensions
RUN source /home/ubuntu/venv/bin/activate && pip3 install torch==1.12.0
RUN source /home/ubuntu/venv/bin/activate && pip3 install ipython jupyterlab
RUN source /home/ubuntu/venv/bin/activate && pip3 install numpy==1.21.3
RUN source /home/ubuntu/venv/bin/activate && pip3 install gymnasium[box2d]==0.29.0
RUN source /home/ubuntu/venv/bin/activate && pip3 install gymnasium-robotics==1.2.2
# RUN source /home/ubuntu/venv/bin/activate && pip3 install robosuite==1.3.2
# RUN source /home/ubuntu/venv/bin/activate && pip3 install -U 'mujoco-py<2.2,>=2.1'
# RUN source /home/ubuntu/venv/bin/activate && pip3 install dm-control==1.0.8
# RUN source /home/ubuntu/venv/bin/activate && pip3 install pyrender==0.1.45
# RUN source /home/ubuntu/venv/bin/activate && pip3 install dm2gym==0.2.0

# border
RUN cd $HOME && mkdir -p .border/model

# Mujoco amd64 binary
RUN cd $HOME && \
    mkdir .mujoco && \
    cd .mujoco && \
    wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
RUN cd $HOME/.mujoco && \
    tar zxf mujoco-2.1.1-linux-x86_64.tar.gz && \
    mkdir -p mujoco210/bin && \
    ln -sf $PWD/mujoco-2.1.1/lib/libmujoco.so.2.1.1 $PWD/mujoco210/bin/libmujoco210.so && \
    ln -sf $PWD/mujoco-2.1.1/lib/libglewosmesa.so $PWD/mujoco210/bin/libglewosmesa.so && \
    ln -sf $PWD/mujoco-2.1.1/include/ $PWD/mujoco210/include && \
    ln -sf $PWD/mujoco-2.1.1/model/ $PWD/mujoco210/model
RUN cp /*.py $HOME

# # PyBulletGym
# RUN source /home/ubuntu/venv/bin/activate && pip3 install pybullet==3.2.5
# RUN source /home/ubuntu/venv/bin/activate && pip3 install pybullet==2.7.1
# RUN source /home/ubuntu/venv/bin/activate && \
#     cd $HOME && \
#     git clone https://github.com/bulletphysics/bullet3.git && \
#     cd bullet3 && \
#     git checkout -b tmp 2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5 && \
#     ./build_cmake_pybullet_double.sh
# RUN cd $HOME && \
#     git clone https://github.com/benelot/pybullet-gym.git && \
#     cd pybullet-gym && \
#     git checkout -b tmp bc68201c8101c4e30dde95f425647a0709ee2f29 && \
#     source /home/ubuntu/venv/bin/activate && \
#     pip install -e .
# # Tweaks for version incompatibility of gym and pybullet-gym
# RUN sed -i 's/return state, sum(self.rewards), bool(done), {}/return state, sum(self.rewards), bool(done), bool(done), {}/g' /home/ubuntu/pybullet-gym/pybulletgym/envs/roboschool/envs/locomotion/walker_base_env.py
# RUN sed -i 's/id='\''AntPyBulletEnv-v0'\'',/id='\''AntPyBulletEnv-v0'\'', order_enforce=False,/g' /home/ubuntu/pybullet-gym/pybulletgym/envs/__init__.py

# .bashrc
RUN echo 'export LIBTORCH=$HOME/venv/lib/python3.8/site-packages/torch' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LIBTORCH/lib' >> ~/.bashrc
RUN echo 'export LIBTORCH_CXX11_ABI=0' >> ~/.bashrc
RUN echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
RUN echo 'export PYTHONPATH=$HOME/border/border-py-gym-env/examples:$PYTHONPATH' >> ~/.bashrc
RUN echo 'export CARGO_TARGET_DIR=$HOME/target' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
RUN echo 'export MUJOCO_GL=glfw' >> ~/.bashrc
RUN echo 'source $HOME/venv/bin/activate' >> ~/.bashrc
# RUN echo 'export PYTHONPATH=$HOME/bullet3/build_cmake/examples/pybullet:$PYTHONPATH' >> ~/.bashrc

USER root
RUN rm /bin/sh && mv /bin/sh_tmp /bin/sh

ENV USER ubuntu
