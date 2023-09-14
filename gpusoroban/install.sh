export DEBIAN_FRONTEND=noninteractive
sudo apt update -qy
sudo apt install --no-install-recommends -y xfce4 xfce4-goodies tigervnc-standalone-server novnc websockify sudo xterm init systemd snapd vim net-tools curl wget git tzdata
sudo apt install -y dbus-x11 x11-utils x11-xserver-utils x11-apps
sudo apt install software-properties-common -y
sudo apt install -y build-essential libssl-dev swig cmake tmux htop

# clang
sudo apt install -y -q libclang-dev

# sdl
sudo apt install -y -q --no-install-recommends \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-net-dev libsdl2-ttf-dev \
    libsdl2-dev libsdl-image1.2-dev

# python
# sudo apt install -y python3.8 python3.8-dev python3.8-distutils python3.8-venv python3-pip
sudo apt install -y python3 python3-dev python3-distutils python3-venv python3-pip

# headers required for building libtorch
sudo apt install -y libgoogle-glog-dev libgflags-dev

# llvm, mesa for robosuite
sudo apt install -y llvm libosmesa6-dev

# Used for Mujoco
sudo apt install -y patchelf libglfw3 libglfw3-dev

# Cleanup
sudo rm -rf /var/lib/apt/lists/*

# rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# python
cd /home/user && python3 -m venv venv
source /home/user/venv/bin/activate && pip3 install --upgrade pip
source /home/user/venv/bin/activate && pip3 install pyyaml typing-extensions
source /home/user/venv/bin/activate && pip3 install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
source /home/user/venv/bin/activate && pip3 install ipython jupyterlab
source /home/user/venv/bin/activate && pip3 install numpy==1.21.3
source /home/user/venv/bin/activate && pip3 install mujoco==2.3.7
source /home/user/venv/bin/activate && pip3 install gymnasium[box2d]==0.29.0
source /home/user/venv/bin/activate && pip3 install gymnasium-robotics==1.2.2
source /home/user/venv/bin/activate && pip3 install tensorboard

# # Mujoco amd64 binary
# cd $HOME && \
#     mkdir .mujoco && \
#     cd .mujoco && \
#     wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
# cd $HOME/.mujoco && \
#     tar zxf mujoco-2.1.1-linux-x86_64.tar.gz && \
#     mkdir -p mujoco210/bin && \
#     ln -sf $PWD/mujoco-2.1.1/lib/libmujoco.so.2.1.1 $PWD/mujoco210/bin/libmujoco210.so && \
#     ln -sf $PWD/mujoco-2.1.1/lib/libglewosmesa.so $PWD/mujoco210/bin/libglewosmesa.so && \
#     ln -sf $PWD/mujoco-2.1.1/include/ $PWD/mujoco210/include && \
#     ln -sf $PWD/mujoco-2.1.1/model/ $PWD/mujoco210/model
# cp /*.py $HOME

echo 'export LIBTORCH=$HOME/venv/lib/python3.10/site-packages/torch' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LIBTORCH/lib' >> ~/.bashrc
echo 'export LIBTORCH_CXX11_ABI=0' >> ~/.bashrc
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export PYTHONPATH=$HOME/border/border-py-gym-env/examples:$PYTHONPATH' >> ~/.bashrc
echo 'source "$HOME/.cargo/env"' >> ~/.bashrc
echo 'source $HOME/venv/bin/activate' >> ~/.bashrc
echo 'export ATARI_ROM_DIR=$HOME/atari_rom' >> ~/.bashrc
echo 'alias tml="tmux list-sessions"' >> ~/.bashrc
echo 'alias tma="tmux a -t"' >> ~/.bashrc
echo 'alias tms="tmux new -s"' >> ~/.bashrc
