# Changelog

## v0.0.6 (20??-??-??)

### Added

* Docker files (`border`).
* Singularity files (`border`)
* Script for GPUSOROBAN (#67)
* `Evaluator` trait in `border-core` (#70). It can be used to customize evaluation logic in `Trainer`.
* Example of asynchronous trainer for native Atari environment and DQN (`border/examples`).

### Changed

* Bump the version of tch-rs to 0.8.0 (`border-tch-agent`).
* Rename agents as following the convention in Rust (`border-tch-agent`).
* Bump the version of gym to 0.26 (`border-py-gym-env`)
* Remove the type parameter for array shape of gym environments (`border-py-gym-env`)
* Interface of Python-Gym interface (`border-py-gym-env`)
