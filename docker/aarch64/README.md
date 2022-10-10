# Docker container

There are scripts to test the crates in this repository.
The scripts in this directory are tested on M2 Macbook Air.

* [Build and run image](# Build and run image)
* [Install Atari ROM (optional)](#Install Atari ROM (optional))
* [Run examples on GUI](#Run examples on GUI)

## Build and run image

* Run the following command in this directory to build the docker image:

  ```bash
  sh build.sh
  ```

* Run the following command to start the docker container:

  ```bash
  sh run.sh
  ```

You can use GUI in the container with a web browser with `localhost:6080`.
The password for user `ubuntu` is `ubuntu`.

* Run the following command to stop and remove the container:

  ```bash
  sh remove.sh
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

Then you can check if `border-atari-env` runs properly by the following command:

```bash
cd $HOME/border
cargo run --example random_pong
```

## Run examples on GUI

* On the GUI via web browser, open a terminal and run the following command to run an example:

  ```bash
  cd $HOME/border
  cargo run --example dqn_cartpole --features=tch
  ```

<!-- Use robosuite in future
python -m robosuite.demos.demo_random_action -->
