//! Simple generic replay buffer.
use super::{Batch, SimpleReplayBufferConfig, SubBatch};
use crate::{Batch as BatchBase, ReplayBufferBase};
use fastrand::Rng;

/// A simple generic replay buffer.
pub struct SimpleReplayBuffer<O, A>
where
    O: SubBatch,
    A: SubBatch,
{
    capacity: usize,
    i: usize,
    size: usize,
    obs: O,
    act: A,
    next_obs: O,
    reward: Vec<f32>,
    is_done: Vec<i8>,
    rng: Rng,
}

impl<O, A> SimpleReplayBuffer<O, A>
where
    O: SubBatch,
    A: SubBatch,
{
    fn push_reward(&mut self, i: usize, b: &Vec<f32>) {
        let mut j = i;
        for r in b.iter() {
            self.reward[j] = *r;
            j += 1;
            if j == self.capacity {
                j = 0;
            }
        }
    }

    fn push_is_done(&mut self, i: usize, b: &Vec<i8>) {
        let mut j = i;
        for d in b.iter() {
            self.is_done[j] = *d;
            j += 1;
            if j == self.capacity {
                j = 0;
            }
        }
    }

    fn sample_reward(&self, ixs: &Vec<usize>) -> Vec<f32> {
        ixs.iter().map(|ix| self.reward[*ix]).collect()
    }

    fn sample_is_done(&self, ixs: &Vec<usize>) -> Vec<i8> {
        ixs.iter().map(|ix| self.is_done[*ix]).collect()
    }
}

impl<O, A> ReplayBufferBase for SimpleReplayBuffer<O, A>
where
    O: SubBatch,
    A: SubBatch,
{
    type Config = SimpleReplayBufferConfig;
    type PushedItem = Batch<O, A>;
    type Batch = Batch<O, A>;

    fn build(config: &Self::Config) -> Self {
        let capacity = config.capacity;

        Self {
            capacity,
            i: 0,
            size: 0,
            obs: O::new(capacity),
            act: A::new(capacity),
            next_obs: O::new(capacity),
            reward: vec![0.; capacity],
            is_done: vec![0; capacity],
            rng: Rng::with_seed(config.seed),
        }
    }

    fn len(&self) -> usize {
        if self.i < self.capacity {
            self.i
        } else {
            self.capacity
        }
    }

    fn batch(&self, size: usize, _beta: Option<f32>) -> anyhow::Result<Self::Batch> {
        let ixs = (0..size)
            .map(|_| self.rng.usize(..self.size))
            .collect::<Vec<_>>();

        Ok(Self::Batch {
            obs: self.obs.sample(&ixs),
            act: self.act.sample(&ixs),
            next_obs: self.next_obs.sample(&ixs),
            reward: self.sample_reward(&ixs),
            is_done: self.sample_is_done(&ixs),
        })
    }

    fn push(&mut self, tr: Self::PushedItem) {
        let len = tr.len();
        let (obs, act, next_obs, reward, is_done) = tr.unpack();
        self.obs.push(self.i, &obs);
        self.act.push(self.i, &act);
        self.next_obs.push(self.i, &next_obs);
        self.push_reward(self.i, &reward);
        self.push_is_done(self.i, &is_done);

        self.i = (self.i + len) % self.capacity;
        self.size += len;
        if self.size >= self.capacity {
            self.size = self.capacity;
        }
    }
}
