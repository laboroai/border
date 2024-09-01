use super::{get_field_type, get_type_str};
use proc_macro::{self, TokenStream};
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input);
    let DeriveInput { ident, data, .. } = input;
    let field_type = get_field_type(data);
    let field_type_str = get_type_str(
        field_type.clone(),
        "The item for deriving Obs must be a new type like MyObs(BorderAtariObs)",
    );

    // let output = if field_type_str == "PyGymEnvObs" {
    //     py_gym_env_obs(ident, field_type)
    let output = if field_type_str == "BorderAtariObs" {
        atari_env_obs(ident, field_type)
    } else {
        panic!(
            "Deriving Obs supports PyGymEnvObs or BorderAtariObs, given {:?}",
            field_type_str
        );
    };

    output.into()
}

// fn py_gym_env_obs(ident: proc_macro2::Ident, field_type: syn::Type) -> proc_macro2::TokenStream {
//     // #[cfg(not(feature = "tch"))]
//     #[allow(unused_mut)]
//     let mut output = common(ident.clone(), field_type.clone());

//     #[cfg(feature = "tch")]
//     output.extend(quote! {
//         use std::convert::TryFrom as _;

//         impl From<#ident> for tch::Tensor {
//             fn from(obs: #ident) -> tch::Tensor {
//                 // `PyGymEnvObs` implements Into<Tensor> when feature = "tch"
//                 tch::Tensor::try_from(obs.0).unwrap()
//             }
//         }
//     }.into_iter());

//     output
// }

fn atari_env_obs(ident: proc_macro2::Ident, field_type: syn::Type) -> proc_macro2::TokenStream {
    // #[cfg(not(feature = "tch"))]
    #[allow(unused_mut)]
    let mut output = quote! {
        impl border_core::Obs for #ident {
            fn dummy(n: usize) -> Self {
                Obs(BorderAtariObs::dummy(n))
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
    output.extend(
        quote! {
            use std::convert::TryFrom as _;

            impl From<#ident> for tch::Tensor {
                fn from(obs: #ident) -> tch::Tensor {
                    // `BorderAtariObs` implements Into<Tensor> when feature = "tch"
                    tch::Tensor::try_from(obs.0).unwrap()
                }
            }
        }
        .into_iter(),
    );

    output
}

// fn common(ident: proc_macro2::Ident, field_type: syn::Type) -> proc_macro2::TokenStream {
//     quote! {
//         impl border_core::Obs for #ident {
//             fn dummy(n: usize) -> Self {
//                 Obs(PyGymEnvObs::dummy(n))
//             }

//             fn len(&self) -> usize {
//                 self.0.len()
//             }
//         }

//         impl From<#field_type> for #ident {
//             fn from(obs: #field_type) -> Self {
//                 #ident(obs)
//             }
//         }
//     }
// }
