use border_core::replay_buffer::SubBatch;
use candle_core::{error::Result, DType, Device, Tensor};

/// Adds capability of constructing [Tensor] with a static method.
pub trait ZeroTensor {
    /// Constructs zero tensor.
    fn zeros(shape: &[usize]) -> Result<Tensor>;
}

impl ZeroTensor for u8 {
    fn zeros(shape: &[usize]) -> Result<Tensor> {
        Tensor::zeros(shape, DType::U8, &Device::Cpu)
    }
}

impl ZeroTensor for f32 {
    fn zeros(shape: &[usize]) -> Result<Tensor> {
        Tensor::zeros(shape, DType::F32, &Device::Cpu)
    }
}

impl ZeroTensor for i64 {
    fn zeros(shape: &[usize]) -> Result<Tensor> {
        Tensor::zeros(shape, DType::I64, &Device::Cpu)
    }
}

/// A buffer consisting of a [`Tensor`].
///
/// The internal buffer is `Vec<Tensor>`.
#[derive(Clone, Debug)]
pub struct TensorSubBatch {
    buf: Vec<Tensor>,
    capacity: usize,
}

impl TensorSubBatch {
    pub fn from_tensor(t: Tensor) -> Self {
        let capacity = t.dims()[0] as _;
        Self {
            buf: vec![t],
            capacity,
        }
    }
}

impl SubBatch for TensorSubBatch {
    fn new(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            capacity: capacity,
        }
    }

    /// Pushes given data.
    ///
    /// if ix + data.buf.len() exceeds the self.capacity,
    /// the tail samples in data is placed in the head of the buffer of self.
    fn push(&mut self, ix: usize, data: Self) {
        if self.buf.len() == self.capacity {
            for (i, sample) in data.buf.into_iter().enumerate() {
                let ix_ = (ix + i) % self.capacity;
                self.buf[ix_] = sample;
            }
        } else if self.buf.len() < self.capacity {
            for (i, sample) in data.buf.into_iter().enumerate() {
                if self.buf.len() < self.capacity {
                    self.buf.push(sample);
                } else {
                    let ix_ = (ix + i) % self.capacity;
                    self.buf[ix_] = sample;
                }
            }
        } else {
            panic!("The length of the buffer is SubBatch is larger than its capacity.");
        }
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        let buf = ixs.iter().map(|&ix| self.buf[ix].clone()).collect();
        Self {
            buf,
            capacity: ixs.len(),
        }
    }
}

impl From<TensorSubBatch> for Tensor {
    fn from(b: TensorSubBatch) -> Self {
        Tensor::cat(&b.buf[..], 0).unwrap()
    }
}
