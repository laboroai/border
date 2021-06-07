use border_core::shape;
use border_py_gym_env::{newtype_obs, newtype_act_d};

shape!(ObsShape, [1, 2, 3]);
newtype_obs!(Obs, ObsFilter, ObsShape, f32, f32);
newtype_act_d!(ActD, ActDFilter);
newtype_act_d!(ActC, ActCFilter);

fn main() {
}
