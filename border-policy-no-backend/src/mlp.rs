use crate::Mat;
use serde::{Deserialize, Serialize};

#[cfg(feature = "tch")]
use tch::nn::VarStore;

#[derive(Clone, Debug, Deserialize, Serialize)]
/// Multilayer perceptron with ReLU activation function.
pub struct Mlp {
    /// Weights of layers.
    ws: Vec<Mat>,

    /// Biases of layers.
    bs: Vec<Mat>,
}

impl Mlp {
    pub fn forward(&self, x: &Mat) -> Mat {
        let n_layers = self.ws.len();
        let mut x = x.clone();
        for i in 0..n_layers {
            x = self.ws[i].matmul(&x).add(&self.bs[i]);
            if i != n_layers - 1 {
                x = x.relu();
            }
        }
        x.tanh()
    }

    #[cfg(feature = "tch")]
    pub fn from_varstore(vs: &VarStore, w_names: &[&str], b_names: &[&str]) -> Self {
        let vars = vs.variables();
        let ws: Vec<Mat> = w_names
            .iter()
            .map(|name| vars[&name.to_string()].copy().into())
            .collect();
        let bs: Vec<Mat> = b_names
            .iter()
            .map(|name| vars[&name.to_string()].copy().into())
            .collect();

        Self { ws, bs }
    }
}
