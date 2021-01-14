use std::{path::Path, error::Error};
use tch::{Tensor, nn};

pub trait ModelBase {
    fn backward_step(&mut self, loss: &Tensor);

    fn get_var_store(&mut self) -> &mut nn::VarStore;

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>>;

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>>;
}

pub trait Model1: ModelBase {
    type Input;
    type Output;

    fn forward(&self, xs: &Self::Input) -> Self::Output;
}

// pub trait MultiheadModel {
//     fn forward(&self, xs: &Tensor) -> (Tensor, Tensor);

//     fn backward_step(&mut self, loss: &Tensor);

//     fn get_var_store(&mut self) -> &mut nn::VarStore;

//     fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>>;

//     fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>>;
// }

pub trait Model2: ModelBase {
    type Input1;
    type Input2;
    type Output;

    fn forward(&self, x1s: &Self::Input1, x2s: &Self::Input2) -> Self::Output;
}
