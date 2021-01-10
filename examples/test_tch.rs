use kind::FLOAT_CPU;
use tch::*;

fn main() {
    let a = Tensor::zeros(&[4, 5, 6, 7], FLOAT_CPU);
    let b = a.flatten(0, 1);
    println!("{:?}", a.size());
    println!("{:?}", b.size());
}