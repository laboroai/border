use crate::PushedItemMessage;
use anyhow::Result;
use border_core::{ExperienceBufferBase, ReplayBufferBase};
use crossbeam_channel::Sender;
use std::marker::PhantomData;

/// Configuration of [`ReplayBufferProxy`].
#[derive(Clone, Debug)]
pub struct ReplayBufferProxyConfig {
    /// Number of samples buffered until sent to the trainer.
    ///
    /// Here, a sample corresponds to a `R::PushedItem` for [`ReplayBufferProxy`]`<R>`.
    pub n_buffer: usize,
}

/// A wrapper of replay buffer for asynchronous trainer.
pub struct ReplayBufferProxy<R: ReplayBufferBase> {
    id: usize,

    /// Sender of [PushedItemMessage].
    sender: Sender<PushedItemMessage<R::PushedItem>>,

    /// Number of samples buffered until sent to the trainer.
    n_buffer: usize,

    /// Buffer of `R::PushedItem`s.
    buffer: Vec<R::PushedItem>,

    phantom: PhantomData<R>,
}

impl<R: ReplayBufferBase> ReplayBufferProxy<R> {
    pub fn build_with_sender(
        id: usize,
        config: &ReplayBufferProxyConfig,
        sender: Sender<PushedItemMessage<R::PushedItem>>,
    ) -> Self {
        let n_buffer = config.n_buffer;
        Self {
            id,
            sender,
            n_buffer,
            buffer: Vec::with_capacity(n_buffer),
            phantom: PhantomData,
        }
    }
}

impl<R: ReplayBufferBase> ExperienceBufferBase for ReplayBufferProxy<R> {
    type PushedItem = R::PushedItem;

    fn push(&mut self, tr: Self::PushedItem) -> Result<()> {
        self.buffer.push(tr);
        if self.buffer.len() == self.n_buffer {
            let mut buffer = Vec::with_capacity(self.n_buffer);
            std::mem::swap(&mut self.buffer, &mut buffer);

            let msg = PushedItemMessage {
                id: self.id,
                pushed_items: buffer,
            };

            match self.sender.try_send(msg) {
                Ok(()) => {}
                Err(_e) => {
                    return Err(crate::BorderAsyncTrainerError::SendMsgForPush)?;
                }
            }
        }

        Ok(())
    }

    fn len(&self) -> usize {
        unimplemented!();
    }
}

impl<R: ReplayBufferBase> ReplayBufferBase for ReplayBufferProxy<R> {
    type Config = ReplayBufferProxyConfig;
    type Batch = R::Batch;

    fn build(_config: &Self::Config) -> Self {
        unimplemented!();
    }

    fn batch(&mut self, _size: usize) -> anyhow::Result<Self::Batch> {
        unimplemented!();
    }

    fn update_priority(&mut self, _ixs: &Option<Vec<usize>>, _td_err: &Option<Vec<f32>>) {
        unimplemented!();
    }
}
