DEBIAN_FRONTEND=noninteractive
sudo apt-get update -q
sudo apt-get upgrade -yq
sudo apt-get install -yq wget curl git build-essential vim sudo libssl-dev zip swig cmake tmux

# clang
sudo apt install -y -q libclang-dev

# sdl
sudo apt install -y -q --no-install-recommends \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-net-dev libsdl2-ttf-dev \
    libsdl-dev libsdl-image1.2-dev

# python
sudo apt install -y python3.8 python3.8-dev python3.8-distutils python3.8-venv python3-pip

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
cd /home/user && python3.8 -m venv venv
source /home/user/venv/bin/activate && pip3 install --upgrade pip
source /home/user/venv/bin/activate && pip3 install pyyaml typing-extensions
source /home/user/venv/bin/activate && pip3 install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
source /home/user/venv/bin/activate && pip3 install ipython jupyterlab
source /home/user/venv/bin/activate && pip3 install numpy==1.21.3
source /home/user/venv/bin/activate && pip3 install gym[box2d]==0.26.2
source /home/user/venv/bin/activate && pip3 install robosuite==1.3.2
source /home/user/venv/bin/activate && pip3 install -U 'mujoco-py<2.2,>=2.1'
source /home/user/venv/bin/activate && pip3 install pyrender==0.1.45
source /home/user/venv/bin/activate && pip3 install dm2gym==0.2.0

echo 'export LIBTORCH=$HOME/venv/lib/python3.8/site-packages/torch' >> ~/.bashrc
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
