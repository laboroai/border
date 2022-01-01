use std::{collections::HashMap, iter::FromIterator};
use tch::{nn::VarStore, Device::Cpu, Tensor};

/// Named tensors to send model parameters using a channel.
pub struct NamedTensors {
    pub named_tensors: HashMap<String, Tensor>,
}

impl NamedTensors {
    /// Copy data of VarStore to CPU.
    pub fn copy_from(vs: &VarStore) -> Self {
        let src = vs.variables();

        tch::no_grad(|| NamedTensors {
            named_tensors: HashMap::from_iter(src.iter().map(|(k, v)| {
                let v = v.detach().to(Cpu).data();
                (k.clone(), v)
            })),
        })
    }

    /// Copy named tensors to [VarStore].
    pub fn copy_to(&self, vs: &mut VarStore) {
        let src = &self.named_tensors;
        let dest = &mut vs.variables();
        debug_assert_eq!(src.len(), dest.len());
    
        tch::no_grad(|| {
            for (name, src) in src.iter() {
                let dest = dest.get_mut(name).unwrap();
                dest.copy_(src);
            }
        });    
    }
}

impl Clone for NamedTensors {
    fn clone(&self) -> Self {
        let src = &self.named_tensors;

        tch::no_grad(|| NamedTensors {
            named_tensors: HashMap::from_iter(src.iter().map(|(k, v)| {
                let v = v.detach().to(Cpu).data();
                (k.clone(), v)
            })),
        })
    }
}
