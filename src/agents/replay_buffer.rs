use tch::{Tensor, kind::{FLOAT_CPU, INT64_CPU}};

pub struct ReplayBuffer {
    obs: Tensor,
    next_obs: Tensor,
    rewards: Tensor,
    actions: Tensor,
    capacity: usize,
    len: usize,
    i: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize, num_obs: usize, num_actions: usize) -> Self {
        Self {
            obs: Tensor::zeros(&[capacity as _, num_obs as _], FLOAT_CPU),
            next_obs: Tensor::zeros(&[capacity as _, num_obs as _], FLOAT_CPU),
            rewards: Tensor::zeros(&[capacity as _, 1], FLOAT_CPU),
            actions: Tensor::zeros(&[capacity as _, num_actions as _], FLOAT_CPU),
            capacity,
            len: 0,
            i: 0,
        }
    }

    fn push(&mut self, obs: &Tensor, actions: &Tensor, reward: &Tensor, next_obs: &Tensor) {
        let i = self.i % self.capacity;
        self.obs.get(i as _).copy_(obs);
        self.rewards.get(i as _).copy_(reward);
        self.actions.get(i as _).copy_(actions);
        self.next_obs.get(i as _).copy_(next_obs);
        self.i += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    fn random_batch(&self, batch_size: usize) -> Option<(Tensor, Tensor, Tensor, Tensor)> {
        if self.len < 3 {
            return None;
        }

        let batch_size = batch_size.min(self.len - 1);
        let batch_indexes = Tensor::randint((self.len - 2) as _, &[batch_size as _], INT64_CPU);

        let states = self.obs.index_select(0, &batch_indexes);
        let next_states = self.next_obs.index_select(0, &batch_indexes);
        let actions = self.actions.index_select(0, &batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes);

        Some((states, actions, rewards, next_states))
    }
}
