use border_core::generic_replay_buffer::BatchBase;
use tch::{Device, Tensor};

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

/// A buffer consisting of a [`Tensor`].
///
/// The internal buffer of this struct has the shape of `[n_capacity, shape[1..]]`,
/// where `shape` is obtained from the data pushed at the first time via
/// [`TensorBatch::push`] method. `[1..]` means that the first axis of the
/// given data is ignored as it might be batch size.
pub struct TensorBatch {
    buf: Option<Tensor>,
    capacity: i64,
}

impl Clone for TensorBatch {
    fn clone(&self) -> Self {
        let buf = match self.buf.is_none() {
            true => None,
            false => Some(self.buf.as_ref().unwrap().copy()),
        };

        Self {
            buf,
            capacity: self.capacity,
        }
    }
}

impl TensorBatch {
    pub fn from_tensor(t: Tensor) -> Self {
        let capacity = t.size()[0] as _;
        Self {
            buf: Some(t),
            capacity,
        }
    }
}

impl BatchBase for TensorBatch {
    fn new(capacity: usize) -> Self {
        // let capacity = capacity as i64;
        // let mut shape: Vec<_> = S::shape().to_vec().iter().map(|e| *e as i64).collect();
        // shape.insert(0, capacity);
        // let buf = D::zeros(shape.as_slice());

        Self {
            buf: None,
            capacity: capacity as _,
        }
    }

    /// Pushes given data.
    ///
    /// If the internal buffer is empty, it will be initialized with the shape
    /// `[capacity, data.buf.size()[1..]]`.
    fn push(&mut self, index: usize, data: Self) {
        if data.buf.is_none() {
            return;
        }

        let batch_size = data.buf.as_ref().unwrap().size()[0];
        if batch_size == 0 {
            return;
        }

        if self.buf.is_none() {
            let mut shape = data.buf.as_ref().unwrap().size().clone();
            shape[0] = self.capacity;
            let kind = data.buf.as_ref().unwrap().kind();
            let device = tch::Device::Cpu;
            self.buf = Some(Tensor::zeros(&shape, (kind, device)));
        }

        let index = index as i64;
        let val: Tensor = data.buf.as_ref().unwrap().copy();

        for i_ in 0..batch_size {
            let i = (i_ + index) % self.capacity;
            self.buf.as_ref().unwrap().get(i).copy_(&val.get(i_));
        }
    }

    fn sample(&self, ixs: &Vec<usize>) -> Self {
        let ixs = ixs.iter().map(|&ix| ix as i64).collect::<Vec<_>>();
        let batch_indexes = Tensor::of_slice(&ixs);
        let buf = Some(self.buf.as_ref().unwrap().index_select(0, &batch_indexes));
        Self {
            buf,
            capacity: ixs.len() as i64,
        }
    }
}

impl From<TensorBatch> for Tensor {
    fn from(b: TensorBatch) -> Self {
        b.buf.unwrap()
    }
}
