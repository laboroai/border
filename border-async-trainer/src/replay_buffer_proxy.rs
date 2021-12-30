use border_core::{ReplayBufferBase};
use std::marker::PhantomData;
// use crossbeam_channel::{Sender, Receiver, bounded};

// /// Commands that [ReplayBufferProxy] sends and receives.
// pub ReplayBufferCommand

/// Configuration of [ReplayBufferProxy].
#[derive(Clone, Debug)]
pub struct ReplayBufferProxyConfig {
}

/// A wrapper of replay buffer for asynchronous trainer.
pub struct ReplayBufferProxy<B: ReplayBufferBase> {
    phantom: PhantomData<B>,
}

impl<B: ReplayBufferBase> ReplayBufferBase for ReplayBufferProxy<B> {
    type Config = ReplayBufferProxyConfig;
    type PushedItem = B::PushedItem;
    type Batch = B::Batch;

    fn build(_config: &Self::Config) -> Self {
        Self {
            phantom: PhantomData,
        }
    }

    fn push(&mut self, _tr: Self::PushedItem) {
    }

    fn len(&self) -> usize {
        unimplemented!();
    }

    fn batch(&self, _size: usize) -> anyhow::Result<Self::Batch> {
        unimplemented!();
    }

    fn update_priority(&mut self, _ixs: &Option<Vec<usize>>, _td_err: &Option<Vec<f32>>) {
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
