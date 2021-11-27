use border_core::{ReplayBufferBase};
use std::marker::PhantomData;
use crossbeam_channel::{Sender, Receiver, bounded};

// /// Commands that [ReplayBufferProxy] sends and receives.
// pub ReplayBufferCommand

// /// A wrapper of replay buffer for asynchronous trainer.
// pub struct ReplayBufferProxy<B: ReplayBufferBase> {

// }

#[test]
fn send_tensor() {
    use std::thread;
    use tch::Tensor;

    let (s, r) = bounded(1);

    thread::spawn(move || {
        let t = Tensor::of_slice(&[1, 2, 3]);
        s.send(t).unwrap();
    });

    let msg = r.recv().unwrap();
    println!("{:?}", msg);
}
