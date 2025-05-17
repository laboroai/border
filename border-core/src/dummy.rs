//! This module is used for tests.
#[derive(Clone, Debug)]

/// Dummy observation.
pub struct DummyObs;

impl crate::Obs for DummyObs {
    fn len(&self) -> usize {
        unimplemented!();
    }
}

// TODO: Consider to make this work with feature flag tch.
// impl Into<tch::Tensor> for DummyObs {
//     fn into(self) -> tch::Tensor {
//         unimplemented!();
//     }
// }

#[derive(Clone, Debug)]
/// Dummy action.
pub struct DummyAct;

impl crate::Act for DummyAct {
    fn len(&self) -> usize {
        unimplemented!();
    }
}

// TODO: Consider to make this work with feature flag tch.
// impl Into<tch::Tensor> for DummyAct {
//     fn into(self) -> tch::Tensor {
//         unimplemented!();
//     }
// }

// TODO: Consider to make this work with feature flag tch.
// impl From<tch::Tensor> for DummyAct {
//     fn from(_value: tch::Tensor) -> Self {
//         unimplemented!();
//     }
// }

#[derive(Clone)]
/// Dummy inner batch.
pub struct DummyInnerBatch;

// TODO: Consider to make this work with feature flag tch.
// impl Into<tch::Tensor> for DummyInnerBatch {
//     fn into(self) -> tch::Tensor {
//         unimplemented!();
//     }
// }

/// Dummy batch.
pub struct DummyBatch;

impl crate::TransitionBatch for DummyBatch {
    type ObsBatch = DummyInnerBatch;
    type ActBatch = DummyInnerBatch;

    fn len(&self) -> usize {
        unimplemented!();
    }

    fn obs(&self) -> &Self::ObsBatch {
        unimplemented!();
    }

    fn act(&self) -> &Self::ActBatch {
        unimplemented!();
    }

    fn unpack(
        self,
    ) -> (
        Self::ObsBatch,
        Self::ActBatch,
        Self::ObsBatch,
        Vec<f32>,
        Vec<i8>,
        Vec<i8>,
        Option<Vec<usize>>,
        Option<Vec<f32>>,
    ) {
        unimplemented!();
    }
}

/// Dummy replay buffer.
pub struct DummyReplayBuffer;

impl crate::ReplayBufferBase for DummyReplayBuffer {
    type Batch = DummyBatch;
    type Config = usize;

    fn batch(&mut self, _size: usize) -> anyhow::Result<Self::Batch> {
        unimplemented!();
    }

    fn build(_config: &Self::Config) -> Self {
        unimplemented!();
    }

    fn update_priority(&mut self, _ixs: &Option<Vec<usize>>, _td_err: &Option<Vec<f32>>) {
        unimplemented!();
    }
}

#[derive(Clone, Debug)]
/// Dummy info.
pub struct DummyInfo;

impl crate::Info for DummyInfo {}

/// Dummy env.
pub struct DummyEnv;

impl crate::Env for DummyEnv {
    type Config = usize;
    type Act = DummyAct;
    type Obs = DummyObs;
    type Info = DummyInfo;

    fn build(_config: &Self::Config, _seed: i64) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        unimplemented!();
    }

    fn reset(&mut self, _is_done: Option<&Vec<i8>>) -> anyhow::Result<Self::Obs> {
        unimplemented!();
    }

    fn reset_with_index(&mut self, _ix: usize) -> anyhow::Result<Self::Obs> {
        unimplemented!();
    }

    fn step(&mut self, _a: &Self::Act) -> (crate::Step<Self>, crate::record::Record)
    where
        Self: Sized,
    {
        unimplemented!();
    }

    fn step_with_reset(&mut self, _a: &Self::Act) -> (crate::Step<Self>, crate::record::Record)
    where
        Self: Sized,
    {
        unimplemented!();
    }
}
