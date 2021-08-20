//! Sum tree for prioritized sampling.
//!
//! Code is adapted from https://github.com/jaromiru/AI-blog/blob/master/SumTree.py and
/// https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
use segment_tree::{
    ops::{MaxIgnoreNaN, MinIgnoreNaN},
    SegmentPoint,
};

#[derive(Debug)]
pub struct SumTree {
    eps: f32,
    alpha: f32,
    capacity: usize,
    n_samples: usize,
    tree: Vec<f32>,
    min_tree: SegmentPoint<f32, MinIgnoreNaN>,
    max_tree: SegmentPoint<f32, MaxIgnoreNaN>,
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
            max_tree: SegmentPoint::build(vec![1e-8f32; capacity], MaxIgnoreNaN),
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

        if s <= self.tree[left] || self.tree[right] == 0f32 {
            return self.retrieve(left, s);
        } else {
            return self.retrieve(right, s - self.tree[left]);
        }
    }

    pub fn total(&self) -> f32 {
        return self.tree[0];
    }

    pub fn max(&self) -> f32 {
        self.max_tree
            .query(0, self.max_tree.len())
            .powf(-self.alpha)
    }

    /// Add priority value at `ix`-th element in the sum tree.
    ///
    /// The alpha-th power of the priority value is taken when addition.
    pub fn add(&mut self, ix: usize, p: f32) {
        debug_assert!(ix <= self.n_samples);

        self.update(ix, p);

        if self.n_samples < self.capacity {
            self.n_samples += 1;
        }
    }

    /// Update priority value at `ix`-th element in the sum tree.
    pub fn update(&mut self, ix: usize, p: f32) {
        debug_assert!(ix < self.capacity);

        let p = (p + self.eps).powf(self.alpha);
        self.min_tree.modify(ix, p);
        self.max_tree.modify(ix, p);
        let ix = ix + self.capacity - 1;
        let change = p - self.tree[ix];
        if change.is_nan() {
            println!("{:?}, {:?}", p, self.tree[ix]);
            panic!();
        }
        self.tree[ix] = p;
        self.propagate(ix, change);
    }

    /// Get the maximal index of the sum tree where the sum of priority values is less than `s`.
    pub fn get(&self, s: f32) -> usize {
        let ix = self.retrieve(0, s);
        debug_assert!(ix >= (self.capacity - 1));
        ix + 1 - self.capacity
    }

    /// Samples indices for batch and returns normalized weights.
    ///
    /// The weight is $w_i=\left(N^{-1}P(i)^{-1}\right)^{\beta}$
    /// and it will be normalized by $max_i w_i$.
    pub fn sample(&self, batch_size: usize, beta: f32) -> (Vec<i64>, Vec<f32>) {
        let p_sum = &self.total();
        let ps = (0..batch_size).map(|_| p_sum * fastrand::f32()).collect::<Vec<_>>();
        let indices = ps.iter().map(|&p| self.get(p)).collect::<Vec<_>>();
        // let indices = (0..batch_size)
        //     .map(|_| self.get(p_sum * fastrand::f32()))
        //     .collect::<Vec<_>>();

        let n = self.n_samples as f32 / p_sum;
        let ws = indices
            .iter()
            .map(|ix| self.tree[ix + self.capacity - 1])
            .map(|p| (n * p).powf(-beta))
            .collect::<Vec<_>>();

        // normalizer within all samples
        // let w_max_inv = (n * self.min_tree.query(0, self.n_samples)).powf(beta);

        // normalizer within batch
        let w_max_inv = 1f32 / ws.iter().fold(0.0 / 0.0, |m, v| v.max(m));

        let ws = ws.iter().map(|w| w * w_max_inv).collect::<Vec<f32>>();

        // debug
        // if self.n_samples % 100 == 0 || p_sum.is_nan() || w_max.is_nan() {
        if p_sum.is_nan() || w_max_inv.is_nan() || ws.iter().sum::<f32>().is_nan() {
            println!("self.n_samples: {:?}", self.n_samples);
            println!("p_sum: {:?}", p_sum);
            println!("w_max_inv: {:?}", w_max_inv);
            println!("ps: {:?}", ps);
            println!("indices: {:?}", indices);
            println!("{:?}", ws);
            panic!();
        }

        let ixs = indices.iter().map(|&ix| ix as i64).collect();

        (ixs, ws)
    }
}

#[cfg(test)]
mod tests {
    use super::SumTree;

    #[test]
    fn test_sum_tree_odd() {
        let data = vec![0.5f32, 0.2, 0.8, 0.3, 1.1, 2.5, 3.9];
        let mut sum_tree = SumTree::new(8, 1.0);
        for ix in 0..data.len() {
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

        let n_samples = 1000000;
        let (ixs, _) = sum_tree.sample(n_samples, 1.0);
        debug_assert!(ixs.iter().all(|&ix| ix < data.len() as i64));
        (0..5).for_each(|ix| {
            let p = data[ix] / sum_tree.total() * (n_samples as f32);
            let n = ixs.iter().filter(|&&e| e == ix as i64).collect::<Vec<_>>().len();
            println!("ix={:?}: {:?} (p={:?})", ix, n, p);
        })
    }
}
