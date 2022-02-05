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
        "The item for deriving SubBatch must be a new type like SubBatch(TensorSubBatch<ObsShape, f32>)",
    );

    let output = if field_type_str == "TensorSubBatch" {
        tensor_sub_batch(ident, field_type)
    } else {
        panic!(
            "Deriving ObsBatch support TensorSubBatch, given {:?}",
            field_type_str
        );
    };

    output.into()
}

fn tensor_sub_batch(ident: proc_macro2::Ident, field_type: syn::Type) -> proc_macro2::TokenStream {
    #[allow(unused_mut)]
    let mut output = quote! {
        impl border_core::replay_buffer::SubBatch for #ident {
            fn new(capacity: usize) -> Self {
                Self(TensorSubBatch::new(capacity))
            }

            fn push(&mut self, i: usize, data: &Self) {
                self.0.push(i, &data.0)
            }

            fn sample(&self, ixs: &Vec<usize>) -> Self {
                let buf = self.0.sample(ixs);
                Self(buf)
            }
        }

        impl From<#field_type> for #ident {
            fn from(obs: #field_type) -> Self {
                #ident(obs)
            }
        }
    };

    #[cfg(feature = "tch")]
    output.extend(quote! {
        impl From<#ident> for tch::Tensor {
            fn from(b: #ident) -> Self {
                b.0.into()
            }
        }
    }.into_iter());

    output
}
