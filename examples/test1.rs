use std::cell::RefCell;
use ndarray::{Array2}; //, s};

use lrr::core::{Obs, Info, Env, Policy, Sampler};

// ----------

struct MyAct (f32);

#[derive(Clone)]
struct MyObs (f32);

impl Obs for MyObs {
    fn new() -> Self {
        MyObs(1.0)
    }
}

struct MyInfo ();

impl Info for MyInfo {}

struct MyPolicy ();

impl Policy<MyEnv> for MyPolicy {
    fn sample(&self, _: &MyObs) -> MyAct {
        MyAct(1.0)
    }
}

struct MyEnv {
    state: RefCell<Array2::<f32>>,
}

impl Env for MyEnv {
    type Obs  = MyObs;
    type Act  = MyAct;
    type Info = MyInfo;

    fn step(&self, _a: &MyAct) -> (MyObs, f32, bool, MyInfo) {
        let mut state = self.state.borrow_mut();
        state[[0, 0]] += 1.0;
        (MyObs(state[[0, 0]]), 0.0, true, MyInfo{})
    }

    fn reset(&self) -> MyObs {
        MyObs(0.0)
    }
}

impl MyEnv {
    fn new() -> Self {
        MyEnv {
            state: RefCell::new(Array2::<f32>::zeros((3, 4))),
        }
    }
}

// ----------

fn main() {
    let env = MyEnv::new();
    let pi = MyPolicy {};
    let sampler = Sampler::new(env, pi);

    sampler.sample(10);

    println!("finished!");
}
