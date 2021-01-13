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

fn elemwise_multiply() {
    println!("=== elemwise_muptiply ===");

    let m = Tensor::of_slice(&[1 as f32, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let n = 0.5 as f32 * Tensor::ones(&[2, 2], tch::kind::FLOAT_CPU);
    m.print();
    (&m * n).print();
    println!("{:?}", m.size());
}

fn reduce_with_negative_index() {
    println!("=== reduce_with_negative_index ===");

    let z = Tensor::randn(&[2, 3, 4, 5, 6], tch::kind::FLOAT_CPU);
    // let tmp = &z.size().as_slice()[..=2];
    // let tmp2 = &[-1i64];
    // let tmp3 = [tmp, tmp2].concat().as_slice();
    let z_reduce = z.view([&z.size().as_slice()[..=2], &[-1i64]].concat().as_slice());
    let z_reduce = z_reduce.sum1(&[-1], false, tch::Kind::Float);
    
    println!("{:?}", z.size());
    // println!("{:?}", (1..(z.size().len())).collect::<Vec<_>>());
    println!("{:?}", z_reduce.size());
}

fn main() {
    // let nd = ndarray::arr2(&[[1f32, 2.], [3., 4.]]).into_dyn();
    // let tensor = try_from(nd.clone()).unwrap();

    // println!("{:?}", nd);
    // tensor.print();
    // println!("{:?}", tensor.values());
    // let a = Tensor::zeros(&[4, 5, 6, 7], FLOAT_CPU);
    // let b = a.flatten(0, 1);
    // println!("{:?}", a.size());
    // println!("{:?}", b.size());
    // let a = Array2::zeros((2, 3));
    // let b = Tensor::
    elemwise_multiply();
    reduce_with_negative_index();
}