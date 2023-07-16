# Changelog

## v0.0.6 (20??-??-??)

### Added

* Add Docker files (`border`).
* Add Singularity files (`border`)
* Add script for GPUSOROBAN (#67)
* Add `Evaluator` trait in `border-core` (#70). It can be used to customize evaluation logic in `Trainer`.

### Changed

* Bump the version of tch-rs to 0.8.0 (`border-tch-agent`).
* Rename agents as following the convention in Rust (`border-tch-agent`).
* Bump the version of gym to 0.26 (`border-py-gym-env`)
* Remove the type parameter for array shape of gym environments (`border-py-gym-env`)
