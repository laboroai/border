## PyBullet Env Ant-v0

* Training

  ```bash
  $ RUST_LOG=info cargo run --example sac_ant
  ```

  <img src="https://drive.google.com/uc?id=1d9UJCtz31iX2XYo_FaqVx8ZSkRLa8eMI" width="256">

* Testing

  ```bash
  $ cargo run --example sac_ane -- --play=$REPO/examples/model/sac_ant
  ```
