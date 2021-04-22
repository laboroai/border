use std::io;
use tch::{Tensor, IndexOp};
use border::agent::tch::util::quantile_huber_loss;

fn main() {
    let tau = Tensor::of_slice(
        (1..10).map(|x| 0.1 * x as f32).collect::<Vec<_>>().as_slice()
    );
    let x = Tensor::vstack(&(0..tau.size()[0])
        .map(|_| Tensor::of_slice(
            (0..40).map(|x| (x as f32 - 20.) / 10.).collect::<Vec<_>>().as_slice()
        )).collect::<Vec<_>>().as_slice()
    );
    let x_tgt = x.zeros_like();

    let data = quantile_huber_loss(&x, &x_tgt, &tau).tr();
    let mut wtr = csv::Writer::from_writer(io::stdout());

    (0..data.size()[0])
        .map(|i| data.i(i))
        .map(|t| Vec::<f32>::from(&t).iter().map(|x| x.to_string()).collect::<Vec<_>>())
        .for_each(|v| wtr.write_record(&v).unwrap());
}