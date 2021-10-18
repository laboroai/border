use super::{get_field_type, get_type_str};
use proc_macro::{self, TokenStream};
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input);
    // let opts = Opts::from_derive_input(&input).expect("Wrong options");
    let DeriveInput { ident, data, .. } = input;
    let field_type = get_field_type(data);
    let field_type_str = get_type_str(
        field_type.clone(),
        "The item for deriving Obs must be a new type like MyObs(PyGymEnvObs)",
    );

    let output = if field_type_str == "PyGymEnvObs" {
        py_gym_env_obs(ident, field_type)
    } else {
        panic!("Deriving Obs support PyGymEnvObs, given {:?}", field_type_str);
    };

    output.into()
}

fn py_gym_env_obs(ident: proc_macro2::Ident, field_type: syn::Type) -> proc_macro2::TokenStream {
    #[cfg(not(feature = "tch"))]
    let output = quote! {
        impl border_core::Obs for #ident {
            fn dummy(n: usize) -> Self {
                Obs(PyGymEnvObs::dummy(n))
            }

            fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
                Obs(self.0.merge(obs_reset.0, is_done))
            }

            fn len(&self) -> usize {
                self.0.len()
            }
        }

        impl From<#field_type> for #ident {
            fn from(obs: #field_type) -> Self {
                #ident(obs)
            }
        }
    };

    #[cfg(feature = "tch")]
    let output = quote! {
        impl border_core::Obs for #ident {
            fn dummy(n: usize) -> Self {
                Obs(PyGymEnvObs::dummy(n))
            }

            fn merge(self, obs_reset: Self, is_done: &[i8]) -> Self {
                Obs(self.0.merge(obs_reset.0, is_done))
            }

            fn len(&self) -> usize {
                self.0.len()
            }
        }

        impl From<#field_type> for #ident {
            fn from(obs: #field_type) -> Self {
                #ident(obs)
            }
        }

        impl From<#ident> for tch::Tensor {
            fn from(obs: #ident) -> tch::Tensor {
                tch::Tensor::try_from(obs.0).unwrap()
            }
        }
    };

    output
}