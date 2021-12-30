use border_core::{ReplayBufferBase};
use std::marker::PhantomData;
// use crossbeam_channel::{Sender, Receiver, bounded};

// /// Commands that [ReplayBufferProxy] sends and receives.
// pub ReplayBufferCommand

/// Configuration of [ReplayBufferProxy].
pub struct ReplayBufferProxiConfig {
}

/// A wrapper of replay buffer for asynchronous trainer.
pub struct ReplayBufferProxy<B: ReplayBufferBase> {
    phantom: PhantomData<B>,
}

impl<B: ReplayBufferBase> ReplayBufferBase for ReplayBufferProxy<B> {
    type Config = ReplayBufferProxiConfig;
    type PushedItem = B::PushedItem;
    type Batch = B::Batch;

    fn build(config: &Self::Config) -> Self {
        unimplemented!();
    }

    fn len(&self) -> usize {
        unimplemented!();
    }

    fn batch(&self, size: usize) -> anyhow::Result<Self::Batch> {
        unimplemented!();
    }

    fn push(&mut self, tr: Self::PushedItem) {
        unimplemented!();
    }

    fn update_priority(&mut self, ixs: &Option<Vec<usize>>, td_err: &Option<Vec<f32>>) {
        unimplemented!();
    }
}

// #[test]
// fn send_tensor() {
//     use std::thread;
//     use tch::Tensor;

//     let (s, r) = bounded(1);

//     thread::spawn(move || {
//         let t = Tensor::of_slice(&[1, 2, 3]);
//         s.send(t).unwrap();
//     });

//     let msg = r.recv().unwrap();
//     println!("{:?}", msg);
// }
