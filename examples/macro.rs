use border_py_gym_env::{newtype_obs, shape};

shape!(ObsShape, [1, 2, 3]);
// newtype_obs!(Obs, ObsShape, f32, f32);
newtype_obs!(Obs, ObsFilter, ObsShape, f32, f32);

fn main() {
}
