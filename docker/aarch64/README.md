# Docker container

There are scripts to test the crates in this repository.
The scripts in this directory are tested on M2 Macbook Air.

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

## Run examples on GUI

* On the GUI via web browser, open a terminal and run the following command to run an example:

  ```bash
  cd border
  cargo run --example dqn_cartpole --features=tch
  ```

<!-- Use robosuite in future
python -m robosuite.demos.demo_random_action -->
