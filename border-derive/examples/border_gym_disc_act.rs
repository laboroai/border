use border_derive::Act;
use border_py_gym_env::GymDiscreteAct;

#[allow(dead_code)]
#[derive(Clone, Debug, Act)]
struct MyAct(GymDiscreteAct);

fn main() {}
