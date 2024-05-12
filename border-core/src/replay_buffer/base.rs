//! Simple generic replay buffer.
mod iw_scheduler;
mod sum_tree;
use super::{config::PerConfig, SimpleReplayBufferConfig, StdBatch, SubBatch};
use crate::{ExperienceBufferBase, ReplayBufferBase, StdBatchBase};
use anyhow::Result;
pub use iw_scheduler::IwScheduler;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use sum_tree::SumTree;
pub use sum_tree::WeightNormalizer;

struct PerState {
    sum_tree: SumTree,
    iw_scheduler: IwScheduler,
}

impl PerState {
    fn new(capacity: usize, per_config: &PerConfig) -> Self {
        Self {
            sum_tree: SumTree::new(capacity, per_config.alpha, per_config.normalize),
            iw_scheduler: IwScheduler::new(
                per_config.beta_0,
                per_config.beta_final,
                per_config.n_opts_final,
            ),
        }
    }
}

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
    is_terminated: Vec<i8>,
    is_truncated: Vec<i8>,
    rng: StdRng,
    per_state: Option<PerState>,
}

impl<O, A> SimpleReplayBuffer<O, A>
where
    O: SubBatch,
    A: SubBatch,
{
    #[inline]
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

    #[inline]
    fn push_is_terminated(&mut self, i: usize, b: &Vec<i8>) {
        let mut j = i;
        for d in b.iter() {
            self.is_terminated[j] = *d;
            j += 1;
            if j == self.capacity {
                j = 0;
            }
        }
    }

    fn push_is_truncated(&mut self, i: usize, b: &Vec<i8>) {
        let mut j = i;
        for d in b.iter() {
            self.is_truncated[j] = *d;
            j += 1;
            if j == self.capacity {
                j = 0;
            }
        }
    }

    fn sample_reward(&self, ixs: &Vec<usize>) -> Vec<f32> {
        ixs.iter().map(|ix| self.reward[*ix]).collect()
    }

    fn sample_is_terminated(&self, ixs: &Vec<usize>) -> Vec<i8> {
        ixs.iter().map(|ix| self.is_terminated[*ix]).collect()
    }

    fn sample_is_truncated(&self, ixs: &Vec<usize>) -> Vec<i8> {
        ixs.iter().map(|ix| self.is_truncated[*ix]).collect()
    }

    /// Sets priorities for the added samples.
    fn set_priority(&mut self, batch_size: usize) {
        let sum_tree = &mut self.per_state.as_mut().unwrap().sum_tree;
        let max_p = sum_tree.max();

        for j in 0..batch_size {
            let i = (self.i + j) % self.capacity;
            sum_tree.add(i, max_p);
        }
    }
}

impl<O, A> ExperienceBufferBase for SimpleReplayBuffer<O, A>
where
    O: SubBatch,
    A: SubBatch,
{
    type PushedItem = StdBatch<O, A>;

    fn len(&self) -> usize {
        self.size
    }

    fn push(&mut self, tr: Self::PushedItem) -> Result<()> {
        let len = tr.len(); // batch size
        let (obs, act, next_obs, reward, is_terminated, is_truncated, _, _) = tr.unpack();
        self.obs.push(self.i, obs);
        self.act.push(self.i, act);
        self.next_obs.push(self.i, next_obs);
        self.push_reward(self.i, &reward);
        self.push_is_terminated(self.i, &is_terminated);
        self.push_is_truncated(self.i, &is_truncated);

        if self.per_state.is_some() {
            self.set_priority(len)
        };

        self.i = (self.i + len) % self.capacity;
        self.size += len;
        if self.size >= self.capacity {
            self.size = self.capacity;
        }

        Ok(())
    }
}

impl<O, A> ReplayBufferBase for SimpleReplayBuffer<O, A>
where
    O: SubBatch,
    A: SubBatch,
{
    type Config = SimpleReplayBufferConfig;
    type Batch = StdBatch<O, A>;

    fn build(config: &Self::Config) -> Self {
        let capacity = config.capacity;
        let per_state = match &config.per_config {
            Some(per_config) => Some(PerState::new(capacity, per_config)),
            None => None,
        };

        Self {
            capacity,
            i: 0,
            size: 0,
            obs: O::new(capacity),
            act: A::new(capacity),
            next_obs: O::new(capacity),
            reward: vec![0.; capacity],
            is_terminated: vec![0; capacity],
            is_truncated: vec![0; capacity],
            // rng: Rng::with_seed(config.seed),
            rng: StdRng::seed_from_u64(config.seed as _),
            per_state,
        }
    }

    fn batch(&mut self, size: usize) -> anyhow::Result<Self::Batch> {
        let (ixs, weight) = if let Some(per_state) = &self.per_state {
            let sum_tree = &per_state.sum_tree;
            let beta = per_state.iw_scheduler.beta();
            let (ixs, weight) = sum_tree.sample(size, beta);
            let ixs = ixs.iter().map(|&ix| ix as usize).collect();
            (ixs, Some(weight))
        } else {
            let ixs = (0..size)
                // .map(|_| self.rng.usize(..self.size))
                .map(|_| (self.rng.next_u32() as usize) % self.size)
                .collect::<Vec<_>>();
            let weight = None;
            (ixs, weight)
        };

        Ok(Self::Batch {
            obs: self.obs.sample(&ixs),
            act: self.act.sample(&ixs),
            next_obs: self.next_obs.sample(&ixs),
            reward: self.sample_reward(&ixs),
            is_terminated: self.sample_is_terminated(&ixs),
            is_truncated: self.sample_is_truncated(&ixs),
            ix_sample: Some(ixs),
            weight,
        })
    }

    fn update_priority(&mut self, ixs: &Option<Vec<usize>>, td_errs: &Option<Vec<f32>>) {
        if let Some(per_state) = &mut self.per_state {
            let ixs = ixs
                .as_ref()
                .expect("ixs should be Some(_) in update_priority().");
            let td_errs = td_errs
                .as_ref()
                .expect("td_errs should be Some(_) in update_priority().");
            for (&ix, &td_err) in ixs.iter().zip(td_errs.iter()) {
                per_state.sum_tree.update(ix, td_err);
            }
            per_state.iw_scheduler.add_n_opts();
        }
    }
}
