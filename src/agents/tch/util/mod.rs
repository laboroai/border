use log::trace;
use crate::agents::Model;

pub fn track(dest: &mut impl Model, src: &mut impl Model, tau: f64) {
    let src = &mut src.get_var_store();
    let dest = &mut dest.get_var_store();
    tch::no_grad(|| {
        for (dest, src) in dest
            .trainable_variables()
            .iter_mut()
            .zip(src.trainable_variables().iter())
        {
            dest.copy_(&(tau * src + (1.0 - tau) * &*dest));
        }
    });
    trace!("soft update");
}
