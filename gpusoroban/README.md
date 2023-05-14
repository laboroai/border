# Script for installing required libraries on Ubuntu

The procedure shown below has been tested on `nvd5-1ub` instance.

```bash
#cd $HOME/
sudo apt install git
git clone https://github.com/taku-y/border.git
cd border/soroban
bash install.sh
source ~/.bashrc
cd $HOME/border
RUST_LOG=info PYTHONPATH=./border-py-gym-env/examples cargo run --example random_cartpole
```

## Copy remote file to local

```bash
scp -r -i ~/.ssh/mykey.txt -P 20122 user@localhost:/home/user/border/border/examples/model/dqn_pong border/examples/model
```

## Install Atari ROM (optional)

If you want to use Atari Learning Environment (ALE), you need to get Atari ROM.
An easy way to do this is to use [AutoROM](https://pypi.org/project/AutoROM/) Python package.
For `border-atari-env` crate, we set `ATARI_ROM_DIR` to the directory including the ROMs.

```bash
pip install autorom
mkdir $HOME/atari_rom
AutoROM --install-dir $HOME/atari_rom
export ATARI_ROM_DIR=$HOME/atari_rom
```
