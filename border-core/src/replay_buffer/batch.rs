//! A generic implementation of [`StdBatchBase`](crate::StdBatchBase).
use super::SubBatch;
use crate::{StdBatchBase, PushedItemBase, util::shuffle};
use rand::{Rng, thread_rng};

/// A generic implementation of [`StdBatchBase`](`crate::StdBatchBase`).
pub struct StdBatch<O, A>
where
    O: SubBatch,
    A: SubBatch,
{
    /// Observations.
    pub obs: O,

    /// Actions.
    pub act: A,

    /// Next observations.
    pub next_obs: O,

    /// Rewards.
    pub reward: Vec<f32>,

    /// Done flags.
    pub is_done: Vec<i8>,

    /// Priority weights.
    pub weight: Option<Vec<f32>>,

    /// Sample indices.
    pub ix_sample: Option<Vec<usize>>,
}

impl<O, A> StdBatchBase for StdBatch<O, A>
where
    O: SubBatch,
    A: SubBatch,
{
    type ObsBatch = O;
    type ActBatch = A;

    fn unpack(
        self,
    ) -> (
        Self::ObsBatch,
        Self::ActBatch,
        Self::ObsBatch,
        Vec<f32>,
        Vec<i8>,
        Option<Vec<usize>>,
        Option<Vec<f32>>,
    ) {
        (
            self.obs,
            self.act,
            self.next_obs,
            self.reward,
            self.is_done,
            self.ix_sample,
            self.weight,
        )
    }

    fn len(&self) -> usize {
        self.reward.len()
    }

    fn obs(&self) -> &Self::ObsBatch {
        &self.obs
    }

    fn act(&self) -> &Self::ActBatch {
        &self.act
    }

    fn next_obs(&self) -> &Self::ObsBatch {
        &self.next_obs
    }

    fn reward(&self) -> &Vec<f32> {
        &self.reward
    }

    fn is_done(&self) -> &Vec<i8> {
        &self.is_done
    }

    fn weight(&self) -> &Option<Vec<f32>> {
        &self.weight
    }

    fn ix_sample(&self) -> &Option<Vec<usize>> {
        &self.ix_sample
    }

    fn empty() -> Self {
        Self {
            obs: O::new(0),
            act: A::new(0),
            next_obs: O::new(0),
            reward: vec![],
            is_done: vec![],
            ix_sample: None,
            weight: None,
        }
    }
}

impl<O, A> PushedItemBase for StdBatch<O, A>
where
    O: SubBatch,
    A: SubBatch,
{
    fn size(&self) -> usize {
        self.reward.len()
    }

    fn shuffle_and_chunk(self, n: usize) -> Vec<Self> {
        let batch_size = self.reward.len();
        let seed: [u8; 32] = thread_rng().gen();
        
        let mut obs_iter = shuffle(self.obs.into_vec(), seed.clone()).into_iter();
        let mut act_iter = shuffle(self.act.into_vec(), seed.clone()).into_iter();
        let mut next_obs_iter = shuffle(self.next_obs.into_vec(), seed.clone()).into_iter();
        let mut reward_iter = shuffle(self.reward, seed.clone()).into_iter();
        let mut is_done_iter = shuffle(self.is_done, seed.clone()).into_iter();
        let mut weight_iter = match self.weight {
            Some(x) => shuffle(x.into_iter().map(|e| Some(e)).collect::<Vec<_>>(), seed.clone()).into_iter(),
            None => vec![None; batch_size].into_iter(),
        };
        let mut ix_sample_iter = match self.ix_sample {
            Some(x) => shuffle(x.into_iter().map(|e| Some(e)).collect::<Vec<_>>(), seed.clone()).into_iter(),
            None => vec![None; batch_size].into_iter(),
        };

        let chunk_size = (batch_size as f64 / n as f64).ceil() as usize;

        (0..n).into_iter().map(|_| {
            Self {
                obs: O::concat(obs_iter.by_ref().take(chunk_size).collect::<Vec<_>>()),
                act: A::concat(act_iter.by_ref().take(chunk_size).collect::<Vec<_>>()),
                next_obs: O::concat(next_obs_iter.by_ref().take(chunk_size).collect::<Vec<_>>()),
                reward: reward_iter.by_ref().take(chunk_size).collect(),
                is_done: is_done_iter.by_ref().take(chunk_size).collect(),
                weight: weight_iter.by_ref().take(chunk_size).collect(),
                ix_sample: ix_sample_iter.by_ref().take(chunk_size).collect(),
            }
        }).collect()
    }

    fn concat(vec: Vec<Self>) -> Self {
        let mut obs_vec = Vec::new();
        let mut act_vec = Vec::new();
        let mut next_obs_vec = Vec::new();
        let mut reward = Vec::new();
        let mut is_done = Vec::new();
        let mut weight = Some(Vec::new());
        let mut ix_sample = Some(Vec::new());
        for mut item in vec.into_iter() {
            obs_vec.push(item.obs);
            act_vec.push(item.act);
            next_obs_vec.push(item.next_obs);
            reward.append(&mut item.reward);
            is_done.append(&mut item.is_done);
            match item.weight {
                Some(mut x) => weight.as_mut().unwrap().append(&mut x),
                None => weight = None,
            };
            match item.ix_sample {
                Some(mut x) => ix_sample.as_mut().unwrap().append(&mut x),
                None => ix_sample = None,
            };
        }
        
        Self {
            obs: O::concat(obs_vec),
            act: A::concat(act_vec),
            next_obs: O::concat(next_obs_vec),
            reward,
            is_done,
            weight,
            ix_sample,
        }
    }
}


impl<O, A> StdBatch<O, A>
where
    O: SubBatch,
    A: SubBatch,
{
    /// Creates new batch with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            obs: O::new(capacity),
            act: A::new(capacity),
            next_obs: O::new(capacity),
            reward: vec![0.0; capacity],
            is_done: vec![0; capacity],
            ix_sample: None,
            weight: None,
        }
    }
}