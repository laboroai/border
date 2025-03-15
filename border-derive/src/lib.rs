//! Derive macros for implementing `border_core::Act` and
//! `border_core::generic_replay_buffer::BatchBase`.
//!
//! # Examples
//!
//! ## Newtype for `BorderAtariAct`
//!
//! ```
//! # use border_core::Act;
//! # use border_derive::Act;
//! # use border_atari_env::BorderAtariAct;
//! #
//! #[derive(Clone, Debug, Act)]
//! struct MyAct(BorderAtariAct);
//! ```
//!
//! The above code will generate the following implementation:
//!
//! ```
//! # use border_core::Act;
//! # use border_derive::Act;
//! # use border_atari_env::BorderAtariAct;
//! #
//! #[derive(Clone, Debug)]
//! struct MyAct(BorderAtariAct);
//! impl border_core::Act for MyAct {
//!     fn len(&self) -> usize {
//!         self.0.len()
//!     }
//! }
//! impl Into<BorderAtariAct> for MyAct {
//!     fn into(self) -> BorderAtariAct {
//!         self.0
//!     }
//! }
//! /// The following code is generated when features="tch" is enabled.
//! impl From<MyAct> for tch::Tensor {
//!     fn from(act: MyAct) -> tch::Tensor {
//!         let v = vec![act.0.act as i64];
//!         let t: tch::Tensor = std::convert::TryFrom::<Vec<i64>>::try_from(v).unwrap();
//!         t.unsqueeze(0)
//!     }
//! }
//! impl From<tch::Tensor> for MyAct {
//!     fn from(t: tch::Tensor) -> Self {
//!         let data: Vec<i64> = {
//!             let t = t.to_dtype(tch::Kind::Int64, false, true);
//!             let n = t.numel();
//!             let mut data = vec![0i64; n];
//!             t.f_copy_data(&mut data, n).unwrap();
//!             data
//!         };
//!         MyAct(BorderAtariAct::new(data[0] as u8))
//!     }
//! }
//! ```
//!
//! ## Newtype for `GymContinuousAct`
//!
//! ```
//! # use border_core::Act;
//! # use border_derive::Act;
//! # use border_py_gym_env::GymContinuousAct;
//! #
//! #[derive(Clone, Debug, Act)]
//! struct MyAct(GymContinuousAct);
//! ```
//!
//! The above code will generate the following implementation:
//! ```
//! # use border_core::Act;
//! # use border_derive::Act;
//! # use border_py_gym_env::GymContinuousAct;
//! #
//! #[derive(Clone, Debug)]
//! struct MyAct(GymContinuousAct);
//! impl border_core::Act for MyAct {
//!     fn len(&self) -> usize {
//!         self.0.len()
//!     }
//! }
//! impl Into<GymContinuousAct> for MyAct {
//!     fn into(self) -> GymContinuousAct {
//!         self.0
//!     }
//! }
//!  /// The following code is generated when features="tch" is enabled.
//! impl From<MyAct> for tch::Tensor {
//!     fn from(act: MyAct) -> tch::Tensor {
//!         let v = act.0.act.iter().map(|e| *e as f32).collect::<Vec<_>>();
//!         let t: tch::Tensor = std::convert::TryFrom::<Vec<f32>>::try_from(v).unwrap();
//!         t.unsqueeze(0)
//!     }
//! }
//! impl From<tch::Tensor> for MyAct {
//!     /// `t` must be a 1-dimentional tensor of `f32`.
//!     fn from(t: tch::Tensor) -> Self {
//!         let shape = t.size()[1..].iter().map(|x| *x as usize).collect::<Vec<_>>();
//!         use std::convert::TryInto;
//!         let act: Vec<f32> = t.try_into().unwrap();
//!         let act = ndarray::Array1::<f32>::from(act)
//!             .into_shape(ndarray::IxDyn(&shape))
//!             .unwrap();
//!         MyAct(GymContinuousAct::new(act))
//!     }
//! }
//! ```
//!
//! ## Newtype for `GymDiscreteAct`
//!
//! ```
//! # use border_core::Act;
//! # use border_derive::Act;
//! # use border_py_gym_env::GymDiscreteAct;
//! #
//! #[derive(Clone, Debug, Act)]
//! struct MyAct(GymDiscreteAct);
//! ```
//!
//! The above code will generate the following implementation:
//! ```
//! # use border_core::Act;
//! # use border_derive::Act;
//! # use border_py_gym_env::GymDiscreteAct;
//! #
//! #[derive(Clone, Debug)]
//! struct MyAct(GymDiscreteAct);
//! impl border_core::Act for MyAct {
//!     fn len(&self) -> usize {
//!         self.0.len()
//!     }
//! }
//! impl Into<GymDiscreteAct> for MyAct {
//!     fn into(self) -> GymDiscreteAct {
//!         self.0
//!     }
//! }
//! impl From<MyAct> for tch::Tensor {
//!     fn from(act: MyAct) -> tch::Tensor {
//!         let v = act.0.act.iter().map(|e| *e as i64).collect::<Vec<_>>();
//!         let t: tch::Tensor = std::convert::TryFrom::<Vec<i64>>::try_from(v).unwrap();
//!         t.unsqueeze(0)
//!     }
//! }
//! impl From<tch::Tensor> for MyAct {
//!     fn from(t: tch::Tensor) -> Self {
//!         use std::convert::TryInto;
//!         let data: Vec<i64> = t.try_into().unwrap();
//!         let data: Vec<_> = data.iter().map(|e| *e as i32).collect();
//!         MyAct(GymDiscreteAct::new(data))
//!     }
//! }
//! ```
//!
//! ## Newtype for `TensorBatch`
//!
//! ```
//! # use border_derive::BatchBase;
//! # use border_tch_agent::TensorBatch;
//! #
//! #[derive(Clone, BatchBase)]
//! struct MyBatch(TensorBatch);
//! ```
//!
//! The above code will generate the following implementation:
//!
//! ```
//! # use border_derive::BatchBase;
//! # use border_tch_agent::TensorBatch;
//! #
//! #[derive(Clone)]
//! struct ObsBatch(TensorBatch);
//! impl border_core::generic_replay_buffer::BatchBase for ObsBatch {
//!     fn new(capacity: usize) -> Self {
//!         Self(TensorBatch::new(capacity))
//!     }
//!     fn push(&mut self, i: usize, data: Self) {
//!         self.0.push(i, data.0)
//!     }
//!     fn sample(&self, ixs: &Vec<usize>) -> Self {
//!         let buf = self.0.sample(ixs);
//!         Self(buf)
//!     }
//! }
//! impl From<TensorBatch> for ObsBatch {
//!     fn from(obs: TensorBatch) -> Self {
//!         ObsBatch(obs)
//!     }
//! }
//! impl From<ObsBatch> for tch::Tensor {
//!     fn from(b: ObsBatch) -> Self {
//!         b.0.into()
//!     }
//! }
//! ```
//!
//! [`border_core::Obs`]: border_core::Obs

mod act;
mod obs;
mod subbatch;
use proc_macro::{self, TokenStream};

/// Implements `border_core::Obs` for the newtype that wraps
/// PyGymEnvObs or BorderAtariObs.
#[deprecated]
#[proc_macro_derive(Obs, attributes(my_trait))]
pub fn derive1(input: TokenStream) -> TokenStream {
    obs::derive(input)
}

/// Implements `border_core::generic_replay_buffer::BatchBase` for the newtype.
#[proc_macro_derive(BatchBase, attributes(my_trait))]
pub fn derive2(input: TokenStream) -> TokenStream {
    subbatch::derive(input)
}

/// Implements [`border_core::Act`] for the newtype.
///
/// [`border_core::Act`]: https://docs.rs/border-core/latest/border_core/trait.Act.html
#[proc_macro_derive(Act, attributes(my_trait))]
pub fn derive3(input: TokenStream) -> TokenStream {
    act::derive(input)
}

fn get_field_type(data: syn::Data) -> syn::Type {
    let field_type = match data {
        syn::Data::Struct(data) => match data.fields {
            syn::Fields::Unnamed(fields) => {
                let tmp = fields.unnamed[0].ty.clone();
                tmp
            }
            _ => {
                unimplemented!();
            }
        },
        _ => {
            unimplemented!();
        }
    };

    field_type
}

fn get_type_str(ty: syn::Type, hint: &str) -> String {
    let msg = "Failed to find the type name.".to_string() + hint;
    match ty {
        syn::Type::Path(p) => p
            .path
            .segments
            .first()
            .expect(&msg.as_str())
            .ident
            .to_string(),
        _ => {
            panic!("{}", msg);
        }
    }
}
