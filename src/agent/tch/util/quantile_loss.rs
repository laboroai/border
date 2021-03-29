//! Quantile loss.
use tch::Tensor;

/// Returns the quantile huber loss.
///
/// The first dimension of `x` corresponds to quantiles `tau`.
pub fn quantile_huber_loss(x: &Tensor, tau: &Tensor) -> Tensor {
    debug_assert_eq!(x.size().len(), 2);
    debug_assert_eq!(tau.size().len(), 1);

    let tau = tau.unsqueeze(-1);
    let x_abs = x.abs();
    let loss = (&x_abs - 0.5).where1(&x_abs.gt(1.0), &(0.5 * x.pow(2.0)));

    (tau - Tensor::where4(&x.detach().lt(0.0), 1., 0.)).abs() * loss
}
