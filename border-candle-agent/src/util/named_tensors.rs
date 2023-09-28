use candle_core::Tensor;
use candle_nn::VarMap;
use std::{collections::HashMap, iter::FromIterator};
// use tch::{nn::VarStore, Device::Cpu, Tensor};

/// Named tensors to send model parameters using a channel.
pub struct NamedTensors {
    pub named_tensors: HashMap<String, Tensor>,
}

impl NamedTensors {
    /// Copy data of [`VarMap`] to CPU.
    pub fn copy_from(vs: &VarMap) -> Self {
        unimplemented!();

        // let src = vs.variables();

        // tch::no_grad(|| NamedTensors {
        //     named_tensors: HashMap::from_iter(src.iter().map(|(k, v)| {
        //         let v = v.detach().to(Cpu).data();
        //         (k.clone(), v)
        //     })),
        // })
    }

    /// Copy named tensors to [`VarMap`].
    pub fn copy_to(&self, vs: &mut VarMap) {
        unimplemented!();

        // let src = &self.named_tensors;
        // let dest = &mut vs.variables();
        // // let device = vs.device();
        // debug_assert_eq!(src.len(), dest.len());

        // tch::no_grad(|| {
        //     for (name, src) in src.iter() {
        //         let dest = dest.get_mut(name).unwrap();
        //         dest.copy_(src);
        //     }
        // });
    }
}

impl Clone for NamedTensors {
    fn clone(&self) -> Self {
        unimplemented!();

        // let src = &self.named_tensors;

        // tch::no_grad(|| NamedTensors {
        //     named_tensors: HashMap::from_iter(src.iter().map(|(k, v)| {
        //         let v = v.detach().to(Cpu).data();
        //         (k.clone(), v)
        //     })),
        // })
    }
}

#[cfg(test)]
mod test {
    use super::NamedTensors;
    use std::convert::{TryFrom, TryInto};
    // use tch::{
    //     nn::{self, Module},
    //     Device::Cpu,
    //     Tensor,
    // };

    #[test]
    fn test_named_tensors() {
        // tch::manual_seed(42);

        // let tensor1 = Tensor::try_from(vec![1., 2., 3.])
        //     .unwrap()
        //     .internal_cast_float(false);

        // let vs1 = nn::VarStore::new(Cpu);
        // let model1 = nn::seq()
        //     .add(nn::linear(&vs1.root() / "layer1", 3, 8, Default::default()))
        //     .add(nn::linear(&vs1.root() / "layer2", 8, 2, Default::default()));

        // let mut vs2 = nn::VarStore::new(tch::Device::cuda_if_available());
        // let model2 = nn::seq()
        //     .add(nn::linear(&vs2.root() / "layer1", 3, 8, Default::default()))
        //     .add(nn::linear(&vs2.root() / "layer2", 8, 2, Default::default()));
        // let device = vs2.device();

        // let t1: Vec<f64> = model1.forward(&tensor1).try_into().unwrap();
        // let t2: Vec<f64> = model2.forward(&tensor1.to(device)).try_into().unwrap();

        // let nt = NamedTensors::copy_from(&vs1);
        // nt.copy_to(&mut vs2);

        // let t3: Vec<f64> = model2.forward(&tensor1.to(device)).try_into().unwrap();

        // for i in 0..2 {
        //     assert!((t1[i] - t2[i]).abs() >= t1[i].abs() * 0.001);
        //     assert!((t1[i] - t3[i]).abs() < t1[i].abs() * 0.001);
        // }
        // // println!("{:?}", t1);
        // // println!("{:?}", t2);
        // // println!("{:?}", t3);
    }
}
