use border_core::generic_replay_buffer::BatchBase;
use candle_core::{error::Result, DType, Device, Tensor};

/// Adds capability of constructing [`Tensor`] with a static method.
///
/// [`Tensor`]: https://docs.rs/candle-core/0.4.1/candle_core/struct.Tensor.html
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
///
/// [`Tensor`]: https://docs.rs/candle-core/0.4.1/candle_core/struct.Tensor.html
#[derive(Clone, Debug)]
pub struct TensorBatch {
    buf: Vec<Tensor>,
    capacity: usize,
}

impl TensorBatch {
    pub fn from_tensor(t: Tensor) -> Self {
        let capacity = t.dims()[0] as _;
        assert_eq!(capacity, 1);
        Self {
            buf: vec![t],
            capacity,
        }
    }
}

impl BatchBase for TensorBatch {
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

impl From<TensorBatch> for Tensor {
    fn from(b: TensorBatch) -> Self {
        Tensor::cat(&b.buf[..], 0).unwrap()
    }
}

impl From<Tensor> for TensorBatch {
    fn from(t: Tensor) -> TensorBatch {
        assert_eq!(t.dims()[0], 1);
        TensorBatch {
            buf: vec![t],
            capacity: 1,
        }
    }
}
