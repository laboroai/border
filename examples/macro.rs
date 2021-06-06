use border_py_gym_env::{newtype_obs, newtype_act_d, shape};

shape!(ObsShape, [1, 2, 3]);
newtype_obs!(Obs, ObsFilter, ObsShape, f32, f32);
newtype_act_d!(Act, ActFilter);

fn main() {
}
