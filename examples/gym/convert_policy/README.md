# Convert model parameters

This program converts model parameters of an SAC agent, trained with tch, to another format 
which is readable using Rust's standard library.

The below command loads model parameters from `../sac_pendulum_tch/model/best`, converts its format, then saves as `./model/mlp.bincode`. It will be used in `pendulum_std` example.

```bash
cargo run
```
