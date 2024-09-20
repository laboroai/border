//! Behavior cloning (BC) agent implemented with candle.
use super::{BcActionType, BcConfig, BcModel};
use crate::{model::SubModel1, util::OutDim};
use anyhow::Result;
use border_core::{
    record::{Record, RecordValue},
    Agent, Configurable, Env, Policy, ReplayBufferBase, TransitionBatch,
};
use candle_core::{shape::D, DType, Device, Tensor};
use candle_nn::{activation, loss::mse};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{de::DeserializeOwned, Serialize};
use std::convert::TryFrom;
use std::{fs, marker::PhantomData, path::Path};

#[allow(dead_code)]
/// Behavior cloning (BC) agent implemented with candle.
///
/// `P` is the type parameter of the policy model.
pub struct Bc<E, P, R>
where
    P: SubModel1<Output = Tensor>,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    policy_model: BcModel<P>,
    batch_size: usize,
    action_type: BcActionType,
    device: Device,
    record_verbose_level: usize,
    phantom: PhantomData<(E, R)>,
}

impl<E, P, R> Policy<E> for Bc<E, P, R>
where
    E: Env,
    P: SubModel1<Output = Tensor>,
    E::Obs: Into<P::Input>,
    E::Act: From<P::Output>,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    /// Sample an action.
    ///
    /// When `action_type` is set to [`BcActionType::Discrete`], this method returns the action
    /// corresponding to the argmax of the policy model's output tensor.
    /// On the other hand, when `action_type` is set to [`BcActionType::Continuous`], this method
    /// returns the output tensor as is.
    fn sample(&mut self, obs: &E::Obs) -> E::Act {
        let a = self.policy_model.forward(&obs.clone().into()).detach();
        match self.action_type {
            BcActionType::Discrete => {
                let a = a.argmax(D::Minus1).unwrap().to_dtype(DType::I64).unwrap();
                a.into()
            }
            BcActionType::Continuous => a.into(),
        }
    }
}

impl<E, P, R> Configurable for Bc<E, P, R>
where
    E: Env,
    P: SubModel1<Output = Tensor>,
    E::Obs: Into<P::Input>,
    E::Act: From<P::Output>,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
{
    type Config = BcConfig<P>;

    /// Constructs DQN agent.
    fn build(config: Self::Config) -> Self {
        let device: Device = config
            .device
            .expect("No device is given for DQN agent")
            .into();
        let policy_model =
            BcModel::build(config.policy_model_config.clone(), device.clone()).unwrap();

        Self {
            policy_model,
            batch_size: config.batch_size,
            action_type: config.action_type,
            device,
            record_verbose_level: config.record_verbose_level,
            phantom: PhantomData,
        }
    }
}

impl<E, P, R> Agent<E, R> for Bc<E, P, R>
where
    E: Env,
    P: SubModel1<Output = Tensor>,
    R: ReplayBufferBase,
    E::Obs: Into<P::Input>,
    E::Act: From<P::Output>,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<P::Input>,
    <R::Batch as TransitionBatch>::ActBatch: Into<Tensor>,
{
    /// For BC agent, this method does nothing.
    fn train(&mut self) {}

    /// For BC agent, this method does nothing.
    fn eval(&mut self) {}

    /// For BC agent, this method always returns `false`.
    fn is_train(&self) -> bool {
        false
    }

    fn opt(&mut self, buffer: &mut R) {
        self.opt_(buffer);
    }

    fn opt_with_record(&mut self, buffer: &mut R) -> Record {
        let mut record = {
            let record = self.opt_(buffer);

            match self.record_verbose_level >= 2 {
                true => {
                    let record_weights = self.policy_model.param_stats();
                    let record = record.merge(record_weights);
                    record
                }
                false => record,
            }
        };

        record
    }

    /// Save model parameters in the given directory.
    ///
    /// The parameters of the policy_model are saved as `policy_model.pt`.
    fn save_params<T: AsRef<Path>>(&self, path: T) -> Result<()> {
        // TODO: consider to rename the path if it already exists
        fs::create_dir_all(&path)?;
        self.policy_model
            .save(&path.as_ref().join("policy_model.pt").as_path())?;
        Ok(())
    }

    /// Load model parameters in the given directory.
    ///
    /// The parameters of the policy_model are loaded from `policy_model.pt`.
    fn load_params<T: AsRef<Path>>(&mut self, path: T) -> Result<()> {
        self.policy_model
            .load(&path.as_ref().join("policy_model.pt").as_path())?;
        Ok(())
    }
}

impl<E, P, R> Bc<E, P, R>
where
    E: Env,
    P: SubModel1<Output = Tensor>,
    R: ReplayBufferBase,
    P::Config: DeserializeOwned + Serialize + OutDim + std::fmt::Debug + PartialEq + Clone,
    R::Batch: TransitionBatch,
    <R::Batch as TransitionBatch>::ObsBatch: Into<P::Input>,
    <R::Batch as TransitionBatch>::ActBatch: Into<Tensor>,
{
    // Currently, this method supports only continuous action.
    fn opt_(&mut self, buffer: &mut R) -> Record {
        let batch = buffer.batch(self.batch_size).unwrap();
        let (obs, act, _, _, _, _, _, _) = batch.unpack();
        let obs = obs.into();
        let act = act.into().to_device(&self.device).unwrap();
        let loss = match self.action_type {
            BcActionType::Discrete => {
                panic!();
            }
            BcActionType::Continuous => {
                let act_ = self.policy_model.forward(&obs);
                mse(&act_, &act)
            }
        }
        .unwrap();
        self.policy_model.backward_step(&loss).unwrap();

        let mut record = Record::empty();
        record.insert(
            "loss",
            RecordValue::Scalar(
                loss.to_device(&Device::Cpu)
                    .expect("Error when moving loss to CPU")
                    .mean_all()
                    .unwrap()
                    .to_scalar()
                    .unwrap(),
            ),
        );
        record
    }
}
