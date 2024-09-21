# Changelog

## v0.0.8 (2024-??-??)

* Add crate `border-minari`, which is a wrapper of [Minari](https://minari.farama.org).
* Bump the version of candle to 0.6.0 (`border-candle-agent`).

## v0.0.7 (2024-09-01)

### Added

* Support MLflow tracking (`border-mlflow-tracking`) (https://github.com/taku-y/border/issues/2).
* Add candle agent (`border-candle-agent`) (https://github.com/taku-y/border/issues/1).
* Add `Trainer::train_offline()` method for offline training (`border-core`) (https://github.com/taku-y/border/issues/18).
* Add crate `border-policy-no-backend`.

### Changed

* Take `self` in the signature of `push()` method of replay buffer (`border-core`).
* Fix a bug in `MlpConfig` (`border-tch-agent`).
* Bump the version of tch to 0.16.0 (`border-tch-agent`).
* Change the name of trait `StepProcessorBase` to `StepProcessor` (`border-core`).
* Change the environment API to include terminate/truncate flags (`border-core`) (https://github.com/taku-y/border/issues/10).
* Split policy trait into two traits, one for sampling (`Policy`) and the other for configuration (`Configurable`) (https://github.com/taku-y/border/issues/12).

## v0.0.6 (2023-09-19)

### Added

* Docker files (`border`).
* Singularity files (`border`).
* Script for GPUSOROBAN (#67).
* `Evaluator` trait in `border-core` (#70). It can be used to customize evaluation logic in `Trainer`.
* Example of asynchronous trainer for native Atari environment and DQN (`border/examples`).
* Move tensorboard recorder into a separate crate (`border-tensorboard`).

### Changed

* Bump the version of tch-rs to 0.8.0 (`border-tch-agent`).
* Rename agents as following the convention in Rust (`border-tch-agent`).
* Bump the version of gym to 0.26 (`border-py-gym-env`).
* Remove the type parameter for array shape of gym environments (`border-py-gym-env`).
* Interface of Python-Gym interface (`border-py-gym-env`).
