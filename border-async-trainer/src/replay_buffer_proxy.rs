use crate::BatchMessage;
use border_core::ReplayBufferBase;
use crossbeam_channel::Sender;
use std::marker::PhantomData;

/// Configuration of [ReplayBufferProxy].
#[derive(Clone, Debug)]
pub struct ReplayBufferProxyConfig {}

/// A wrapper of replay buffer for asynchronous trainer.
pub struct ReplayBufferProxy<R: ReplayBufferBase> {
    /// Sender of [BatchMessage].
    sender: Sender<BatchMessage<R::Batch>>,

    phantom: PhantomData<R>,
}

impl<R: ReplayBufferBase> ReplayBufferProxy<R> {
    pub fn build_with_sender(
        _config: &ReplayBufferProxyConfig,
        sender: Sender<BatchMessage<R::Batch>>,
    ) -> Self {
        Self {
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

    fn push(&mut self, _tr: Self::PushedItem) {}

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
