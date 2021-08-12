//! Replay buffer.
use std::marker::PhantomData;
use tch::{Device, Tensor};
mod base;
pub use base::{ReplayBuffer, TchBatch, TchBuffer, TchBufferOnDevice};
use border_core::Shape;

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
/// Type parameter `D` is the data type of the buffer and one of `u8` or `f32`.
/// S is the shape of the buffer, excepting the first dimension, which is for minibatch.
/// Type parameter `T` is the data stored in the buffer.
pub struct TchTensorBuffer<D, S, T>
where
    D: 'static + Copy + tch::kind::Element + ZeroTensor,
    S: Shape,
    T: Into<Tensor>,
{
    buf: Tensor,
    capacity: i64,
    model_device: tch::Device,
    phantom: PhantomData<(D, S, T)>,
}

impl<D, S, T> TchBuffer for TchTensorBuffer<D, S, T>
where
    D: 'static + Copy + tch::kind::Element + ZeroTensor,
    S: Shape,
    T: Clone + Into<Tensor>,
{
    type Item = T;
    type SubBatch = Tensor;

    /// Creates a buffer for observation or action.
    fn new(capacity: usize, model_device: Device) -> Self {
        let capacity = capacity as i64;
        let mut shape: Vec<_> = S::shape().to_vec().iter().map(|e| *e as i64).collect();
        shape.insert(0, capacity);
        let buf = D::zeros(shape.as_slice()).to(tch::Device::Cpu);

        Self {
            buf,
            capacity,
            model_device,
            phantom: PhantomData,
        }
    }

    /// Push data (`Into<Tensor>`) to the buffer.
    ///
    /// The first dimension of the tensor is the number of samples,
    /// which can be two or more in vectorized environments.
    fn push(&mut self, index: i64, item: &Self::Item) {
        let val: Tensor = item.clone().into();
        let batch_size = val.size()[0];
        debug_assert_eq!(&val.size()[1..], &self.buf.size()[1..]);

        for i_ in 0..batch_size {
            let i = (i_ + index) % self.capacity;
            self.buf.get(i).copy_(&val.get(i_));
        }
    }

    /// Creates minibatch.
    fn batch(&self, batch_indexes: &Tensor) -> Tensor {
        let batch_indexes = batch_indexes.to(self.buf.device());
        self.buf.index_select(0, &batch_indexes).to(self.model_device)
    }
}

impl<D, S, T> TchBufferOnDevice for TchTensorBuffer<D, S, T>
where
    D: 'static + Copy + tch::kind::Element + ZeroTensor,
    S: Shape,
    T: Clone + Into<Tensor>,
{
    fn new_on_device(capacity: usize, device: tch::Device, model_device: tch::Device) -> Self {
        let capacity = capacity as i64;
        let mut shape: Vec<_> = S::shape().to_vec().iter().map(|e| *e as i64).collect();
        shape.insert(0, capacity);
        let buf = D::zeros(shape.as_slice()).to(device);

        Self {
            buf,
            capacity,
            model_device,
            phantom: PhantomData,
        }
    }
}
