use border_derive::Act;
use border_py_gym_env::GymContinuousAct;

#[allow(dead_code)]
#[derive(Clone, Debug, Act)]
struct MyAct(GymContinuousAct);

fn main() {}
