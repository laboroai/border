FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive
RUN echo "Set disable_coredump false" >> /etc/sudo.conf
RUN apt-get update -q && \
    apt-get upgrade -yq && \
    apt-get install -yq wget
RUN apt-get install -yq curl
RUN apt-get install -yq git
RUN apt-get install -yq build-essential
RUN apt-get install -yq vim
# RUN apt-get install -yq sudo
RUN apt-get install -yq libssl-dev

# clang
RUN apt install -y -q libclang-dev

# sdl
RUN apt update -y && \
    apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive && \
    apt install -y -q --no-install-recommends \
        libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-net-dev libsdl2-ttf-dev \
        libsdl-image1.2-dev libsdl1.2-dev

# zip
RUN apt install -y zip

# swig
RUN apt install -y swig

# python
RUN apt install -y python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip

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

# Use bash
RUN mv /bin/sh /bin/sh_tmp && ln -s /bin/bash /bin/sh

# rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# python
RUN cd /root && python3 -m venv venv
RUN source /root/venv/bin/activate && pip3 install --upgrade pip
RUN source /root/venv/bin/activate && pip3 install pyyaml typing-extensions
RUN source /root/venv/bin/activate && pip3 install torch==2.3.0 --index-url https://download.pytorch.org/whl/test/cu118 --timeout 300
RUN source /root/venv/bin/activate && pip3 install ipython jupyterlab
RUN source /root/venv/bin/activate && pip3 install numpy==1.21.3
RUN source /root/venv/bin/activate && pip3 install mujoco==2.3.7
RUN source /root/venv/bin/activate && pip3 install gymnasium[box2d]==0.29.0
RUN source /root/venv/bin/activate && pip3 install gymnasium-robotics==1.2.2
RUN source /root/venv/bin/activate && pip3 install tensorboard==2.16.2
RUN source /root/venv/bin/activate && pip3 install mlflow==2.11.1
RUN source /root/venv/bin/activate && pip3 install tabulate==0.9.0
RUN source /root/venv/bin/activate && pip3 install mlflow-export-import==1.2.0
# RUN source /home/ubuntu/venv/bin/activate && pip3 install robosuite==1.3.2
# RUN source /home/ubuntu/venv/bin/activate && pip3 install -U 'mujoco-py<2.2,>=2.1'
# RUN source /home/ubuntu/venv/bin/activate && pip3 install dm-control==1.0.8
# RUN source /home/ubuntu/venv/bin/activate && pip3 install pyrender==0.1.45
# RUN source /home/ubuntu/venv/bin/activate && pip3 install dm2gym==0.2.0

# border
RUN cd $HOME && mkdir -p .border/model

# # PyBulletGym
# RUN source /home/ubuntu/venv/bin/activate && pip3 install pybullet==3.2.5
# # RUN source /home/ubuntu/venv/bin/activate && pip3 install pybullet==2.7.1
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
RUN echo 'export LIBTORCH=$HOME/venv/lib/python3.10/site-packages/torch' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LIBTORCH/lib' >> ~/.bashrc
RUN echo 'export LIBTORCH_CXX11_ABI=0' >> ~/.bashrc
RUN echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
RUN echo 'export PYTHONPATH=$HOME/border/border-py-gym-env/examples:$PYTHONPATH' >> ~/.bashrc
RUN echo 'export CARGO_TARGET_DIR=$HOME/target' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
RUN echo 'export MUJOCO_GL=glfw' >> ~/.bashrc
RUN echo 'source $HOME/venv/bin/activate' >> ~/.bashrc
RUN echo 'export RUSTFLAGS="-C target-feature=+fp16"' >> ~/.bashrc

RUN rm /bin/sh && mv /bin/sh_tmp /bin/sh

# USER root
# WORKDIR /home/ubuntu/border

# ENV USER ubuntu
# CMD ["/bin/bash", "-l", "-c"]
# CMD source /home/ubuntu/.bashrc
