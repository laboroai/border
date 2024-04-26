# Changelog

## v0.0.7 (20??-??-??)

### Added

* Support MLflow tracking (`border-mlflow-tracking`) (https://github.com/taku-y/border/issues/2)
* Add candle agent (`border-candle-agent`) (https://github.com/taku-y/border/issues/1)
* Split policy trait into two traits, one for sampling (`Policy`) and the other for configuration (`Configurable`) (https://github.com/taku-y/border/issues/12)

### Changed

* Take `self` in the signature of `push()` method of replay buffer (`border-core`)
* Fix a bug in `MlpConfig` (`border-tch-agent`)
* Bump the version of tch to 0.10.0 (`border-tch-agent`)
* Change the name of trait `StepProcessorBase` to `StepProcessor` (`border-core`)

## v0.0.6 (2023-09-19)

### Added

* Docker files (`border`).
* Singularity files (`border`)
* Script for GPUSOROBAN (#67)
* `Evaluator` trait in `border-core` (#70). It can be used to customize evaluation logic in `Trainer`.
* Example of asynchronous trainer for native Atari environment and DQN (`border/examples`).
* Move tensorboard recorder into a separate crate (`border-tensorboard`)

### Changed

* Bump the version of tch-rs to 0.8.0 (`border-tch-agent`).
* Rename agents as following the convention in Rust (`border-tch-agent`).
* Bump the version of gym to 0.26 (`border-py-gym-env`)
* Remove the type parameter for array shape of gym environments (`border-py-gym-env`)
* Interface of Python-Gym interface (`border-py-gym-env`)
