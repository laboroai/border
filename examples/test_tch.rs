use kind::FLOAT_CPU;
use pyo3::*;
use ndarray::*;
use tch::*;
use lrr::agents::tch::py_gym_env::*;

fn try_from(value: ArrayD<f32>) -> Result<Tensor, TchError> {
    // TODO: Replace this with `?` once it works with `std::option::ErrorNone`
    let slice = match value.as_slice() {
        None => return Err(TchError::Convert("cannot convert to slice".to_string())),
        Some(v) => v,
    };
    let tn = Tensor::f_of_slice(slice)?;
    let shape: Vec<i64> = value.shape().iter().map(|s| *s as i64).collect();
    Ok(tn.f_reshape(&shape)?)
}

fn main() {
    let nd = ndarray::arr2(&[[1f32, 2.], [3., 4.]]).into_dyn();
    let tensor = try_from(nd.clone()).unwrap();

    println!("{:?}", nd);
    tensor.print();
    // println!("{:?}", tensor.values());
    // let a = Tensor::zeros(&[4, 5, 6, 7], FLOAT_CPU);
    // let b = a.flatten(0, 1);
    // println!("{:?}", a.size());
    // println!("{:?}", b.size());
    // let a = Array2::zeros((2, 3));
    // let b = Tensor::
}