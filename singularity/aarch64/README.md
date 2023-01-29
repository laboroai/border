# Singularity container for training

This directory contains scripts to build and run a docker container for training.

## Build and run on Mac

The following steps has been tested on M2 Macbook Air.

1. Install and launch [UTM](https://mac.getutm.app/).
2. Download the pre-build UTM image of [this repository](https://github.com/manuparra/singularitycontainers-on-m1-arm64#change-keyboard-layout).
3. Launch the VM.
4. Change the settings for sharing the directory of this repository on the host.
  * Stop the VM, then right click the VM on UTM GUI -> (Stop) -> Edit -> Sharing
    * Directory Share Mode -> SPICE WebDAV
    * Path -> (repository of border)
  * See [here](https://docs.getutm.app/guest-support/linux/) for information of sharing host directories.
5. Login the VM.
  * If you want to use ssh with your favorite terminal, see [this comment](https://github.com/utmapp/UTM/discussions/2535#discussioncomment-4440754).
6. Mount the local directory of this repository:
  ```bash
  # Install davfs2 to mount host directories in the VM
  # https://www.hiroom2.com/2018/05/05/ubuntu-1804-davfs2-ja/ (in Japanese)
  sudo apt install -y davfs2

  sudo mount -t davfs http://127.0.0.1:9843 /home/ska/border
  ```
7. In the VM, cd /mnt/border/singularity/aarch64
8. `sh run.sh` (TODO: more specific description)
