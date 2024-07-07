use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Mat {
    pub data: Vec<f32>,
    pub shape: Vec<i32>,
}

#[cfg(feature = "border-tch-agent")]
impl From<tch::Tensor> for Mat {
    fn from(x: tch::Tensor) -> Self {
        let shape: Vec<i32> = x.size().iter().map(|e| *e as i32).collect();
        let (n, shape) = match shape.len() {
            1 => (shape[0] as usize, vec![shape[0], 1]),
            2 => ((shape[0] * shape[1]) as usize, shape),
            _ => panic!("Invalid matrix size: {:?}", shape),
        };
        let mut data: Vec<f32> = vec![0f32; n];
        x.f_copy_data(&mut data, n).unwrap();
        Self { data, shape }
    }
}

impl Mat {
    pub fn matmul(&self, x: &Mat) -> Self {
        let (m, l, n) = (
            self.shape[0] as usize,
            self.shape[1] as usize,
            x.shape[1] as usize,
        );
        let mut data = vec![0.0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let kk = i * n as usize + j;
                for k in 0..l as usize {
                    data[kk] += self.data[i * l + k] * x.data[k * n + j];
                }
            }
        }

        Self {
            shape: vec![m as _, n as _],
            data,
        }
    }

    pub fn add(&self, x: &Mat) -> Self {
        if self.shape[0] != x.shape[0] || self.shape[1] != x.shape[1] {
            panic!(
                "Trying to add matrices of different sizes: {:?}",
                (&self.shape, &x.shape)
            );
        }

        let data = self
            .data
            .iter()
            .zip(x.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        Mat {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn relu(&self) -> Self {
        let data = self
            .data
            .iter()
            .map(|a| match *a < 0. {
                true => 0.,
                false => *a,
            })
            .collect();

        Self {
            data,
            shape: self.shape.clone(),
        }
    }
}

impl From<Vec<f32>> for Mat {
    fn from(x: Vec<f32>) -> Self {
        let shape = vec![x.len() as i32, 1];
        Self {
            shape,
            data: x,
        }
    }
}
