use super::{arrayd_to_tensor, tensor_to_arrayd, TensorBatch};
use candle_core::{DType, Tensor, D};
use ndarray::ArrayD;

#[derive(Clone, Debug)]
/// Action.
pub enum NdarrayAct {
    Continuous(ArrayD<f32>),
    Discrete(ArrayD<i64>),
}

impl border_core::Act for NdarrayAct {}

impl Into<Tensor> for NdarrayAct {
    fn into(self) -> Tensor {
        match self {
            Self::Continuous(array) => arrayd_to_tensor::<_, f32>(array, true).unwrap(),
            Self::Discrete(array) => {
                let t = arrayd_to_tensor::<_, i64>(array, true).unwrap();
                t.unsqueeze(D::Minus1).unwrap()
            }
        }
    }
}

impl From<Tensor> for NdarrayAct {
    fn from(t: Tensor) -> Self {
        match t.dtype() {
            DType::F32 => Self::Continuous(tensor_to_arrayd(t, true).unwrap()),
            DType::I64 => Self::Discrete(tensor_to_arrayd(t, true).unwrap()),
            _ => panic!(),
        }
    }
}

impl From<NdarrayAct> for TensorBatch {
    fn from(act: NdarrayAct) -> Self {
        let tensor = act.into();
        TensorBatch::from_tensor(tensor)
    }
}
