//! Utilities.
// use crate::model::ModelBase;
use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::VarMap;
use log::trace;
use serde::{Deserialize, Serialize};
mod named_tensors;
mod quantile_loss;
use border_core::record::{Record, RecordValue};
pub use named_tensors::NamedTensors;
pub use quantile_loss::quantile_huber_loss;
use std::convert::TryFrom;

/// Critic loss type.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub enum CriticLoss {
    /// Mean squared error.
    Mse,

    /// Smooth L1 loss.
    SmoothL1,
}

/// Apply soft update on variables.
///
/// Variables are identified by their names.
/// 
/// dest = tau * src + (1.0 - tau) * dest
pub fn track(dest: &VarMap, src: &VarMap, tau: f64) -> Result<()> {
    trace!("dest");
    let dest = dest.data().lock().unwrap();
    trace!("src");
    let src = src.data().lock().unwrap();

    dest.iter().for_each(|(k_dest, v_dest)| {
        let v_src = src.get(k_dest).unwrap();
        let t_src = v_src.as_tensor();
        let t_dest = v_dest.as_tensor();
        let t_dest = ((tau * t_src).unwrap() + (1.0 - tau) * t_dest).unwrap();
        v_dest.set(&t_dest).unwrap();
    });

    Ok(())
}

// /// Concatenates slices.
// pub fn concat_slices(s1: &[i64], s2: &[i64]) -> Vec<i64> {
//     let mut v = Vec::from(s1);
//     v.append(&mut Vec::from(s2));
//     v
// }

/// Interface for handling output dimensions.
pub trait OutDim {
    /// Returns the output dimension.
    fn get_out_dim(&self) -> i64;

    /// Sets the  output dimension.
    fn set_out_dim(&mut self, v: i64);
}

#[test]
fn test_track() -> Result<()> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Init;

    let tau = 0.7;
    let t_src = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &Device::Cpu)?;
    let t_dest = Tensor::from_slice(&[4.0f32, 5.0, 6.0], (3,), &Device::Cpu)?;
    let t = ((tau * &t_src).unwrap() + (1.0 - tau) * &t_dest).unwrap();

    let vm_src = {
        let vm = VarMap::new();
        let init = Init::Randn {
            mean: 0.0,
            stdev: 1.0,
        };
        vm.get((3,), "var1", init, DType::F32, &Device::Cpu)?;
        vm.data().lock().unwrap().get("var1").unwrap().set(&t_src)?;
        vm
    };
    let vm_dest = {
        let vm = VarMap::new();
        let init = Init::Randn {
            mean: 0.0,
            stdev: 1.0,
        };
        vm.get((3,), "var1", init, DType::F32, &Device::Cpu)?;
        vm.data()
            .lock()
            .unwrap()
            .get("var1")
            .unwrap()
            .set(&t_dest)?;
        vm
    };
    track(&vm_dest, &vm_src, tau)?;

    let t_ = vm_dest
        .data()
        .lock()
        .unwrap()
        .get("var1")
        .unwrap()
        .as_tensor()
        .clone();

    println!("{:?}", t);
    println!("{:?}", t_);
    assert!((t - t_)?.abs()?.sum(0)?.to_scalar::<f32>()? < 1e-32);

    Ok(())
}

/// See <https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html>.
pub fn smooth_l1_loss(x: &Tensor, y: &Tensor) -> Result<Tensor, candle_core::Error> {
    let device = x.device();
    let d = (x - y)?.abs()?;
    let m1 = d.lt(1.0)?.to_dtype(DType::F32)?.to_device(&device)?;
    let m2 = Tensor::try_from(1f32)?
        .to_device(&device)?
        .broadcast_sub(&m1)?;
    (((0.5 * m1)? * d.powf(2.0))? + m2 * (d - 0.5))?.mean_all()
}

/// Returns the standard deviation of a tensor.
pub fn std(t: &Tensor) -> f32 {
    t.broadcast_sub(&t.mean_all().unwrap())
        .unwrap()
        .powf(2f64)
        .unwrap()
        .mean_all()
        .unwrap()
        .sqrt()
        .unwrap()
        .to_vec0::<f32>()
        .unwrap()
}

/// Returns the mean and standard deviation of the parameters.
pub fn param_stats(varmap: &VarMap) -> Record {
    let mut record = Record::empty();

    for (k, v) in varmap.data().lock().unwrap().iter() {
        let m: f32 = v.mean_all().unwrap().to_vec0().unwrap();
        let k_mean = format!("{}_mean", &k);
        record.insert(k_mean, RecordValue::Scalar(m));

        let m: f32 = std(v.as_tensor());
        let k_std = format!("{}_std", &k);
        record.insert(k_std, RecordValue::Scalar(m));
    }

    record
}
