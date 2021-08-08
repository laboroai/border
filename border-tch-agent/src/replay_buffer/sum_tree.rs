//! Sum tree for prioritized sampling.
//!
//! Code is adapted from https://github.com/jaromiru/AI-blog/blob/master/SumTree.py

#[derive(Debug)]
pub struct SumTree {
    capacity: usize,
    tree: Vec<f32>,
}

impl SumTree {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            tree: vec![0f32; 2 * capacity - 1],
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
    pub fn add(&mut self, ix: usize, p: f32) {
        self.update(ix + self.capacity - 1, p);
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

    /// Samples indices for batch.
    pub fn sample(&self, batch_size: usize) -> Vec<usize> {
        let s_max = &self.total() * 0.9999999;

        (0..batch_size)
            .map(|_| self.get(s_max * fastrand::f32()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::SumTree;

    #[test]
    fn test_sum_tree_odd() {
        let data = vec![0.5f32, 0.2, 0.8, 0.3, 1.1];
        let mut sum_tree = SumTree::new(8);
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

        let n_samples = 100000;
        let ixs = sum_tree.sample(n_samples);
        (0..5).for_each(|ix| {
            let p = data[ix] / sum_tree.total() * (n_samples as f32);
            let n = ixs.iter().filter(|&&e| e == ix).collect::<Vec<_>>().len();
            println!("ix={:?}: {:?} (p={:?})", ix, n, p);
        })
    }
}
