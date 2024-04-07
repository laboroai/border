//! Quantile loss.
use candle_core::Tensor;

/// Returns the quantile huber loss.
///
/// `x` and `tau` has the same shape.
pub fn quantile_huber_loss(_x: &Tensor, _tau: &Tensor) -> Tensor {
    // TODO: implements this
    panic!();
    // debug_assert_eq!(x.size().as_slice(), tau.size().as_slice());

    // let lt_0 = &x.lt(0.0).detach();
    // let loss = x.smooth_l1_loss(&Tensor::zeros_like(x), tch::Reduction::None, 1.0);
    // (tau - Tensor::where_scalar(lt_0, 1., 0.)).abs() * loss
}
