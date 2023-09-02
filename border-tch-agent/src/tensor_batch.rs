use border_core::{Shape, replay_buffer::SubBatch};
use tch::{Tensor, Device};
use std::marker::PhantomData;

/// Adds capability of constructing [Tensor] with a static method.
pub trait ZeroTensor {
    /// Constructs zero tensor.
    fn zeros(shape: &[i64]) -> Tensor;
}

impl ZeroTensor for u8 {
    fn zeros(shape: &[i64]) -> Tensor {
        Tensor::zeros(&shape, (tch::kind::Kind::Uint8, Device::Cpu))
    }
}

impl ZeroTensor for i32 {
    fn zeros(shape: &[i64]) -> Tensor {
        Tensor::zeros(&shape, (tch::kind::Kind::Int, Device::Cpu))
    }
}

impl ZeroTensor for f32 {
    fn zeros(shape: &[i64]) -> Tensor {
        Tensor::zeros(&shape, tch::kind::FLOAT_CPU)
    }
}

impl ZeroTensor for i64 {
    fn zeros(shape: &[i64]) -> Tensor {
        Tensor::zeros(&shape, (tch::kind::Kind::Int64, Device::Cpu))
    }
}

/// A buffer consisting of a [Tensor](tch::Tensor).
///
/// Type parameter `D` is the data type of the buffer.
/// S is the shape of items in the buffer.
pub struct TensorSubBatch<S, D> {
    buf: Tensor,
    capacity: i64,
    phantom: PhantomData<(S, D)>,
}

impl<S, D> Clone for TensorSubBatch<S, D> {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.copy(),
            capacity: self.capacity,
            phantom: PhantomData
        }
    }
}

impl<S, D> TensorSubBatch<S, D>
where
    S: Shape,
    D: 'static + Copy + tch::kind::Element + ZeroTensor,
{
    pub fn from_tensor(t: Tensor) -> Self {
        let capacity = t.size()[0] as _;
        Self {
            buf: t,
            capacity,
            phantom: PhantomData,
        }
    }
}

impl<S, D> SubBatch for TensorSubBatch<S, D>
where
    S: Shape,
    D: 'static + Copy + tch::kind::Element + ZeroTensor,
{
    fn new(capacity: usize) -> Self {
        let capacity = capacity as i64;
        let mut shape: Vec<_> = S::shape().to_vec().iter().map(|e| *e as i64).collect();
        shape.insert(0, capacity);
        let buf = D::zeros(shape.as_slice());

        Self {
            buf,
            capacity,
            phantom: PhantomData,
        }
    }

    fn push(&mut self, index: usize, data: Self) {
        let index = index as i64;
        let val: Tensor = data.buf;
        let batch_size = val.size()[0];
        debug_assert_eq!(&val.size()[1..], &self.buf.size()[1..]);

        for i_ in 0..batch_size {
            let i = (i_ + index) % self.capacity;
            self.buf.get(i).copy_(&val.get(i_));
        }
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        let ixs = ixs.iter().map(|&ix| ix as i64).collect::<Vec<_>>();
        let batch_indexes = Tensor::of_slice(&ixs);
        let buf = self.buf.index_select(0, &batch_indexes);
        Self {
            buf,
            capacity: ixs.len() as i64,
            phantom: PhantomData,
        }
    }
}

impl<S, D> From<TensorSubBatch<S, D>> for Tensor {
    fn from(b: TensorSubBatch<S, D>) -> Self {
        b.buf
    }
}
