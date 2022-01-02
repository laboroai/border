use crate::PushedItemMessage;
use border_core::{ReplayBufferBase};
use crossbeam_channel::Sender;
use std::marker::PhantomData;

/// Configuration of [ReplayBufferProxy].
#[derive(Clone, Debug)]
pub struct ReplayBufferProxyConfig {}

/// A wrapper of replay buffer for asynchronous trainer.
pub struct ReplayBufferProxy<R: ReplayBufferBase> {
    id: usize,

    /// Sender of [PushedItemMessage].
    sender: Sender<PushedItemMessage<R::PushedItem>>,

    phantom: PhantomData<R>,
}

impl<R: ReplayBufferBase> ReplayBufferProxy<R> {
    pub fn build_with_sender(
        id: usize,
        _config: &ReplayBufferProxyConfig,
        sender: Sender<PushedItemMessage<R::PushedItem>>,
    ) -> Self {
        Self {
            id,
            sender,
            phantom: PhantomData,
        }
    }
}

impl<R: ReplayBufferBase> ReplayBufferBase for ReplayBufferProxy<R> {
    type Config = ReplayBufferProxyConfig;
    type PushedItem = R::PushedItem;
    type Batch = R::Batch;

    fn build(_config: &Self::Config) -> Self {
        unimplemented!();
    }

    fn push(&mut self, tr: Self::PushedItem) {
        let msg = PushedItemMessage {
            id: self.id,
            pushed_item: tr,
        };
        self.sender.send(msg).unwrap();
    }

    fn len(&self) -> usize {
        unimplemented!();
    }

    fn batch(&mut self, _size: usize) -> anyhow::Result<Self::Batch> {
        unimplemented!();
    }

    fn update_priority(&mut self, _ixs: &Option<Vec<usize>>, _td_err: &Option<Vec<f32>>) {
        unimplemented!();
    }
}
