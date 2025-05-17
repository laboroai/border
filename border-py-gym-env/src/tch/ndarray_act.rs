use super::{arrayd_to_tensor, tensor_to_arrayd, TensorBatch};
use ndarray::ArrayD;
use tch::{Kind, Tensor};

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
            Self::Continuous(array) => arrayd_to_tensor::<_, f32>(array, true),
            Self::Discrete(array) => arrayd_to_tensor::<_, i64>(array, true),
        }
    }
}

impl From<Tensor> for NdarrayAct {
    fn from(t: Tensor) -> Self {
        match t.kind() {
            Kind::Float => Self::Continuous(tensor_to_arrayd(t, true)),
            Kind::Int64 => Self::Discrete(tensor_to_arrayd(t.flatten(0, -1), true)),
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
