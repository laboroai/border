//! Sum tree for prioritized sampling.
//!
//! Code is adapted from https://github.com/jaromiru/AI-blog/blob/master/SumTree.py and
/// https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

use segment_tree::{SegmentPoint, ops::MinIgnoreNaN};

#[derive(Debug)]
pub struct SumTree {
    eps: f32,
    alpha: f32,
    capacity: usize,
    n_samples: usize,
    tree: Vec<f32>,
    min_tree: SegmentPoint<f32, MinIgnoreNaN>
}

impl SumTree {
    pub fn new(capacity: usize, alpha: f32) -> Self {
        Self {
            eps: 1e-8,
            alpha,
            capacity,
            n_samples: 0,
            tree: vec![0f32; 2 * capacity - 1],
            min_tree: SegmentPoint::build(vec![f32::MAX; capacity], MinIgnoreNaN),
        }
    }

    fn propagate(&mut self, ix: usize, change: f32) {
        let parent = (ix - 1) / 2;
        self.tree[parent] += change;
        if parent != 0 {
            self.propagate(parent, change);
        }
    }

    fn retrieve(&self, ix: usize, s: f32) -> usize {
        let left = 2 * ix + 1;
        let right = left + 1;

        if left >= self.tree.len() {
            return ix;
        }

        if s <= self.tree[left] {
            return self.retrieve(left, s);
        } else {
            return self.retrieve(right, s - self.tree[left]);
        }
    }

    pub fn total(&self) -> f32 {
        return self.tree[0];
    }

    /// Add priority value at `ix`-th element in the sum tree.
    ///
    /// The alpha-th power of the priority value is taken when addition.
    pub fn add(&mut self, ix: usize, p: f32) {
        let p = p.powf(self.alpha) + self.eps;
        self.update(ix + self.capacity - 1, p);
        self.min_tree.modify(ix, p);

        if self.n_samples < self.capacity {
            self.n_samples += 1;
        }
    }

    /// Update priority value at `ix`-th element in the sum tree.
    pub fn update(&mut self, ix: usize, p: f32) {
        let change = p - self.tree[ix];
        self.tree[ix] = p;
        self.propagate(ix, change);
    }

    /// Get the maximal index of the sum tree where the sum of priority values is less than `s`.
    pub fn get(&self, s: f32) -> usize {
        let ix = self.retrieve(0, s);
        ix + 1 - self.capacity
    }

    /// Samples indices for batch and returns normalized weights.
    ///
    /// The weight is $w_i=\left(N^{-1}P(i)^{-1}\right)^{\beta}$
    /// and it will be normalized by $max_i w_i$.
    pub fn sample(&self, batch_size: usize, beta: f32) -> (Vec<usize>, Vec<f32>) {
        let s_max = &self.total() * 0.9999999;
        let indices = (0..batch_size)
            .map(|_| self.get(s_max * fastrand::f32()))
            .collect::<Vec<_>>();

        let p_sum = self.tree[0];
        let n_div = p_sum / self.n_samples as f32;
        let w_max = (n_div / self.min_tree.query(0, self.n_samples)).powf(beta);
        let ws = indices.iter()
            .map(|ix| self.tree[ix + self.capacity - 1])
            .map(|p| (n_div / p).powf(beta) / w_max)
            .collect::<Vec<_>>();

        (indices, ws)
    }
}

#[cfg(test)]
mod tests {
    use super::SumTree;

    #[test]
    fn test_sum_tree_odd() {
        let data = vec![0.5f32, 0.2, 0.8, 0.3, 1.1];
        let mut sum_tree = SumTree::new(8, 1.0);
        for ix in 0..5 {
            sum_tree.add(ix, data[ix]);
        }

        assert_eq!(sum_tree.get(0.0), 0);
        assert_eq!(sum_tree.get(0.4), 0);
        assert_eq!(sum_tree.get(0.5), 0);
        assert_eq!(sum_tree.get(0.6), 1);
        assert_eq!(sum_tree.get(1.2), 2);
        assert_eq!(sum_tree.get(1.6), 3);
        assert_eq!(sum_tree.get(2.0), 4);
        assert_eq!(sum_tree.get(2.8), 4);

        let (ixs, ws) = sum_tree.sample(10, 1.0);
        println!("{:?}", ixs);
        println!("{:?}", ws);
        println!();

        let n_samples = 100000;
        let (ixs, _) = sum_tree.sample(n_samples, 1.0);
        (0..5).for_each(|ix| {
            let p = data[ix] / sum_tree.total() * (n_samples as f32);
            let n = ixs.iter().filter(|&&e| e == ix).collect::<Vec<_>>().len();
            println!("ix={:?}: {:?} (p={:?})", ix, n, p);
        })
    }
}
