mod obs;
mod subbatch;
mod act;
use proc_macro::{self, TokenStream};

#[proc_macro_derive(Obs, attributes(my_trait))]
pub fn derive1(input: TokenStream) -> TokenStream {
    obs::derive(input)
}

#[proc_macro_derive(SubBatch, attributes(my_trait))]
pub fn derive2(input: TokenStream) -> TokenStream {
    subbatch::derive(input)
}

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
