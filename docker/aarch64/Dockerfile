FROM --platform=linux/aarch64 ubuntu:22.04

# Adapted from https://qiita.com/takahashiakari/items/f096e5bcdfecf3d5ba90
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y && apt install --no-install-recommends -y xfce4 xfce4-goodies tigervnc-standalone-server novnc websockify sudo xterm init systemd snapd vim net-tools curl wget git tzdata
RUN apt update -y && apt install -y dbus-x11 x11-utils x11-xserver-utils x11-apps
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:mozillateam/ppa -y
RUN echo 'Package: *' >> /etc/apt/preferences.d/mozilla-firefox
RUN echo 'Pin: release o=LP-PPA-mozillateam' >> /etc/apt/preferences.d/mozilla-firefox
RUN echo 'Pin-Priority: 1001' >> /etc/apt/preferences.d/mozilla-firefox
RUN echo 'Unattended-Upgrade::Allowed-Origins:: "LP-PPA-mozillateam:jammy";' | tee /etc/apt/apt.conf.d/51unattended-upgrades-firefox
RUN apt update -y && apt install -y firefox
RUN apt update -y && apt install -y xubuntu-icon-theme
RUN touch /root/.Xauthority
EXPOSE 5901
EXPOSE 6080

ENV DEBIAN_FRONTEND noninteractive
RUN echo "Set disable_coredump false" >> /etc/sudo.conf
RUN apt-get update -q && \
    apt-get upgrade -yq && \
    apt-get install -yq wget curl git build-essential vim sudo libssl-dev

# clang
RUN apt install -y -q libclang-dev

# sdl
RUN apt update -y && \
    apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive && \
    apt install -y -q --no-install-recommends \
        libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-net-dev libsdl2-ttf-dev \
        libsdl-image1.2-dev

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

COPY test_mujoco_py.py /test_mujoco_py.py
COPY test_dmc_viewer.py /test_dmc_viewer.py
RUN chmod 777 /test_mujoco_py.py
RUN chmod 777 /test_dmc_viewer.py

# Use bash
RUN mv /bin/sh /bin/sh_tmp && ln -s /bin/bash /bin/sh

# rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# python
RUN cd /root && python3 -m venv venv
RUN source /root/venv/bin/activate && pip3 install --upgrade pip
RUN source /root/venv/bin/activate && pip3 install pyyaml typing-extensions
RUN source /root/venv/bin/activate && pip3 install torch==1.12.0
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

CMD bash -c "vncserver -localhost no -SecurityTypes None -geometry 1024x768 --I-KNOW-THIS-IS-INSECURE && openssl req -new -subj "/C=JP" -x509 -days 365 -nodes -out self.pem -keyout self.pem && websockify -D --web=/usr/share/novnc/ --cert=self.pem 6080 localhost:5901 && tail -f /dev/null"
