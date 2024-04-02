//! Utilities.
// use crate::model::ModelBase;
use anyhow::Result;
use candle_core::{Tensor, DType};
use candle_nn::VarMap;
use log::trace;
use serde::{Deserialize, Serialize};
mod named_tensors;
mod quantile_loss;
pub use named_tensors::NamedTensors;
pub use quantile_loss::quantile_huber_loss;
use std::convert::TryFrom;

/// Critic loss type.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub enum CriticLoss {
    /// Mean squared error.
    MSE,

    /// Smooth L1 loss.
    SmoothL1,
}

// /// Apply soft update on a model.
// ///
// /// Variables are identified by their names.
// pub fn track<M: ModelBase>(dest: &mut M, src: &mut M, tau: f64) {
//     let src = &mut src.get_var_store().variables();
//     let dest = &mut dest.get_var_store().variables();
//     debug_assert_eq!(src.len(), dest.len());

//     let names = src.keys();
//     tch::no_grad(|| {
//         for name in names {
//             let src = src.get(name).unwrap();
//             let dest = dest.get_mut(name).unwrap();
//             dest.copy_(&(tau * src + (1.0 - tau) * &*dest));
//         }
//     });
//     trace!("soft update");
// }

/// Apply soft update on model parameters.
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

/// See https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
pub fn smooth_l1_loss(x: &Tensor, y: &Tensor) -> Result<Tensor, candle_core::Error> {
    let d = (x - y)?.abs()?;
    let m1 = d.lt(1.0)?.to_dtype(DType::F32)?;
    let m2 = Tensor::try_from(1f32)?.broadcast_sub(&m1)?;
    (((0.5 * m1)? * d.powf(2.0))? + m2 * (d - 0.5))?.mean_all()
}
