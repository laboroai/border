use border_core::generic_replay_buffer::BatchBase;
use candle_core::{error::Result, DType, Device, IndexOp, Tensor};

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
    buf: Option<Tensor>,
    capacity: usize,
}

impl TensorBatch {
    pub fn from_tensor(t: Tensor) -> Self {
        let capacity = t.dims()[0] as _;
        Self {
            buf: Some(t),
            capacity,
        }
    }

    pub fn to(&mut self, device: &Device) -> Result<()> {
        if let Some(buf) = &self.buf {
            self.buf = Some(buf.to_device(device)?);
        }
        Ok(())
    }
}

impl BatchBase for TensorBatch {
    fn new(capacity: usize) -> Self {
        Self {
            buf: None,
            capacity: capacity,
        }
    }

    /// Pushes given data.
    ///
    /// If the internal buffer is empty, it will be initialized with the shape
    /// `[capacity, data.buf.dims()[1..]]`.
    fn push(&mut self, index: usize, data: Self) {
        if data.buf.is_none() {
            return;
        }

        let batch_size = data.buf.as_ref().unwrap().dims()[0];
        if batch_size == 0 {
            return;
        }

        if self.buf.is_none() {
            let mut shape = data.buf.as_ref().unwrap().dims().to_vec();
            shape[0] = self.capacity;
            let dtype = data.buf.as_ref().unwrap().dtype();
            let device = Device::Cpu;
            self.buf = Some(Tensor::zeros(shape, dtype, &device).unwrap());
        }

        if index + batch_size > self.capacity {
            let batch_size = self.capacity - index;
            let data = &data.buf.unwrap();
            let data1 = data.i((..batch_size,)).unwrap();
            let data2 = data.i((batch_size..,)).unwrap();
            self.buf
                .as_mut()
                .unwrap()
                .slice_set(&data1, 0, index)
                .unwrap();
            self.buf.as_mut().unwrap().slice_set(&data2, 0, 0).unwrap();
        } else {
            self.buf
                .as_mut()
                .unwrap()
                .slice_set(&data.buf.unwrap(), 0, index)
                .unwrap();
        }
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        let capacity = ixs.len();
        let ixs = {
            let device = self.buf.as_ref().unwrap().device();
            let ixs = ixs.iter().map(|x| *x as u32).collect();
            Tensor::from_vec(ixs, &[capacity], device).unwrap()
        };
        let buf = Some(self.buf.as_ref().unwrap().index_select(&ixs, 0).unwrap());
        Self { buf, capacity }
    }
}

impl From<TensorBatch> for Tensor {
    fn from(b: TensorBatch) -> Self {
        b.buf.unwrap()
    }
}
