//! IQN model.
use std::{path::Path, default::Default, marker::PhantomData, error::Error, f64::consts::PI};
use log::{info, trace};
use tch::{Tensor, Kind::Float, Device, nn, nn::{OptimizerConfig, Module, VarStore}};
use super::super::model::{ModelBase, SubModel};

#[allow(clippy::upper_case_acronyms)]
/// Constructs [IQNModel].
///
/// The type parameter `F` represents a submodel for a feature extractor,
/// converting [F::Input] to a feature vector.
/// The type parameter `M` represents a submodel for merging
/// cos-embedded percent points and feature vectors. 
pub struct IQNModelBuilder<F, M> where
    F: SubModel,
    M: SubModel,
{
    feature_dim: i64,
    embed_dim: i64,
    out_dim: i64,
    learning_rate: f64,
    phantom: PhantomData<(F, M)>
}

// impl<F, T, M> Default for IQNModelBuilder<F, T, M> where
//     F: SubModel<Input = T, Output = Tensor>,
//     M: SubModel<Input = Tensor, Output = Tensor>,
impl<F, M> Default for IQNModelBuilder<F, M> where
    F: SubModel,
    M: SubModel,
{
    fn default() -> Self {
        Self {
            feature_dim: 0,
            embed_dim: 0,
            out_dim: 0,
            learning_rate: 0.0,
            phantom: PhantomData
        }
    }
}

impl<F, M> IQNModelBuilder<F, M> where
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>,
{
    /// Sets the dimension of cos-embedding of percent points.
    pub fn embed_dim(mut self, v: i64) -> Self {
        self.embed_dim = v;
        self
    }

    /// Sets the dimension of feature vectors.
    pub fn feature_dim(mut self, v: i64) -> Self {
        self.feature_dim = v;
        self
    }

    /// Sets the dimension of output vectors, i.e., the number of discrete outputs.
    pub fn out_dim(mut self, v: i64) -> Self {
        self.out_dim = v;
        self
    }

    /// Sets the learning rate.
    pub fn learning_rate(mut self, v: f64) -> Self {
        self.learning_rate = v;
        self
    }

    /// Constructs [IQNModel].
    pub fn build(&self, fe_config: F::Config, m_config: M::Config, device: Device) -> IQNModel<F, M> {
        let feature_dim = self.feature_dim;
        let embed_dim = self.embed_dim;
        let out_dim = self.out_dim;
        let learning_rate = self.learning_rate;
        let var_store = nn::VarStore::new(device);

        // Feature extractor
        let psi = F::build(&var_store, fe_config);

        // Cosine embedding
        let phi = IQNModel::<F, M>::cos_embed_nn(&var_store, feature_dim, embed_dim);

        // Merge
        let f = M::build(&var_store, m_config);

        let opt = nn::AdamW::default().build(&var_store, learning_rate).unwrap();

        IQNModel {
            device,
            var_store,
            feature_dim,
            embed_dim,
            out_dim,
            psi,
            phi,
            f,
            learning_rate,
            opt,
            phantom: PhantomData,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
/// Constructs IQN output layer, which takes input features and percent points.
/// It returns action-value quantiles.
///
/// The type parameter `F` is [FeatureExtractor].
pub struct IQNModel<F, M> where 
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>,
{
    device: Device,
    var_store: nn::VarStore,

    // Dimension of the input (feature) vector.
    // The `size()[-1]` of F::Output (Tensor) is feature_dim.
    feature_dim: i64,

    // Dimension of the cosine embedding vector.
    embed_dim: i64,

    // Dimension of the output vector (equal to the number of actions).
    pub(super) out_dim: i64,

    // Feature extractor
    psi: F,

    // Cos embedding
    phi: nn::Sequential,

    // Merge network
    f: M,

    // Optimizer
    learning_rate: f64,
    opt: nn::Optimizer<nn::AdamW>,

    phantom: PhantomData<(F, M)>
}

impl<F, M> IQNModel<F, M> where
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>,
{
    // Cosine embedding.
    fn cos_embed_nn(var_store: &VarStore, feature_dim: i64, embed_dim: i64) -> nn::Sequential {
        let p = &var_store.root();
        let device = p.device();
        nn::seq()
            .add_fn(move |tau| {
                let batch_size = tau.size().as_slice()[0];
                let n_percent_points = tau.size().as_slice()[1];
                let tau = tau.unsqueeze(-1);
                let i = Tensor::range(1, embed_dim, (Float, device)).unsqueeze(0).unsqueeze(0);
                debug_assert_eq!(tau.size().as_slice(), &[batch_size, n_percent_points, 1]);
                debug_assert_eq!(i.size().as_slice(), &[1, 1, embed_dim]);

                let cos = Tensor::cos(&(tau * (PI * i)));
                debug_assert_eq!(cos.size().as_slice(), &[batch_size, n_percent_points, embed_dim]);

                cos.reshape(&[-1, embed_dim])
            })
            .add(nn::linear(p / "iqn_cos_to_feature", embed_dim, feature_dim, Default::default()))
            .add_fn(|x| x.relu())
    }
}

impl<F, M> Clone for IQNModel<F, M> where
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>,
{
    fn clone(&self) -> Self {
        let device = self.device;
        let feature_dim = self.feature_dim;
        let embed_dim = self.embed_dim;
        let out_dim = self.out_dim;
        let learning_rate = self.learning_rate;
        let mut var_store = nn::VarStore::new(device);

        // Feature extractor
        let psi = self.psi.clone_with_var_store(&var_store);

        // Cos-embedding
        let phi = IQNModel::<F, M>::cos_embed_nn(&var_store, feature_dim, embed_dim);

        // Merge
        let f = self.f.clone_with_var_store(&var_store);

        let opt = nn::AdamW::default().build(&var_store, learning_rate).unwrap();

        var_store.copy(&self.var_store).unwrap();

        Self {
            device,
            var_store,
            feature_dim,
            embed_dim,
            out_dim,
            psi,
            phi,
            f,
            learning_rate,
            opt,
            phantom: PhantomData,
        }
    }
}

impl<F, M> IQNModel<F, M> where
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>,
{
    /// Returns the tensor of action-value quantiles.
    ///
    /// * The shape of` psi(x)` (feature vector) is [batch_size, feature_dim].
    /// * The shape of `tau` is [batch_size, n_percent_points].
    /// * The shape of the output is [batch_size, n_percent_points, self.out_dim].
    pub fn forward(&self, x: &F::Input, tau: &Tensor) -> Tensor {
        // Used to check tensor size
        let feature_dim = self.feature_dim;
        let n_percent_points = tau.size().as_slice()[1];

        // Feature extraction
        let psi = self.psi.forward(x);
        let batch_size = psi.size().as_slice()[0];
        debug_assert_eq!(psi.size().as_slice(), &[batch_size, feature_dim]);

        // Cosine embedding of percent points, eq. (4) in the paper
        debug_assert_eq!(tau.size().as_slice(), &[batch_size, n_percent_points]);
        let phi = self.phi.forward(tau);
        debug_assert_eq!(phi.size().as_slice(), &[batch_size * n_percent_points, self.feature_dim]);
        let phi = phi.reshape(&[batch_size, n_percent_points, self.feature_dim]);

        // Merge features and embedded quantiles by elem-wise multiplication
        let psi = psi.unsqueeze(1);
        debug_assert_eq!(psi.size().as_slice(), &[batch_size, 1, self.feature_dim]);
        let m = psi * phi;
        debug_assert_eq!(m.size().as_slice(), &[batch_size, n_percent_points, self.feature_dim]);

        // Action-value
        let a = self.f.forward(&m);
        debug_assert_eq!(a.size().as_slice(), &[batch_size, n_percent_points, self.out_dim]);

        a
    }
}

impl<F, M> ModelBase for IQNModel<F, M> where
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>,
{
    fn backward_step(&mut self, loss: &Tensor) {
        self.opt.backward_step(loss);
    }

    fn get_var_store(&mut self) -> &mut nn::VarStore {
        &mut self.var_store
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.save(&path)?;
        info!("Save IQN model to {:?}", path.as_ref());
        let vs = self.var_store.variables();
        for (name, _) in vs.iter() {
            trace!("Save variable {}", name);
        };
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.var_store.load(&path)?;
        info!("Load IQN model from {:?}", path.as_ref());
        Ok(())
    }
}

#[allow(clippy::upper_case_acronyms)]
/// The way of taking percent points.
pub enum IQNSample {
    /// Samples over percent points `0.05:0.1:0.95`.
    ///
    /// The precent points are constants.
    Const10,

    /// 10 samples from uniform distribution.
    Uniform10,

    /// 8 samples from uniform distribution.
    Uniform8,

    /// 32 samples from uniform distribution.
    Uniform32,
}

impl IQNSample {
    /// Returns samples of percent points.
    pub fn sample(&self, batch_size: i64) -> Tensor {
        match self {
            Self::Const10 => Tensor::of_slice(
                &[0.05_f32, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
            ).unsqueeze(0).repeat(&[batch_size, 1]),
            Self::Uniform10 => Tensor::rand(&[batch_size, 10], tch::kind::FLOAT_CPU),
            Self::Uniform8 => Tensor::rand(&[batch_size, 8], tch::kind::FLOAT_CPU),
            Self::Uniform32 => Tensor::rand(&[batch_size, 32], tch::kind::FLOAT_CPU),
        }
    }

    /// Returns the number of percent points generated by this way.
    pub fn n_percent_points(&self) -> i64 {
        match self {
            Self::Const10 => 10,
            Self::Uniform10 => 10,
            Self::Uniform8 => 8,
            Self::Uniform32 => 32,
        }
    }
}

/// Takes an average over percent points specified by `mode`.
///
/// * `obs` - Observations.
/// * `iqn` - IQN model.
/// * `mode` - The way of taking percent points.
pub(super) fn average<F, M>(batch_size: i64, obs: &F::Input, iqn: &IQNModel<F, M>, mode: &IQNSample, device: Device)
    -> Tensor where
    F: SubModel<Output = Tensor>,
    M: SubModel<Input = Tensor, Output = Tensor>
{
    let tau = mode.sample(batch_size).to(device);
    let averaged_action_value = iqn.forward(obs, &tau).mean1(&[1], false, Float);
    let batch_size = averaged_action_value.size()[0];
    let n_action = iqn.out_dim;
    debug_assert_eq!(averaged_action_value.size().as_slice(), &[batch_size, n_action]);
    averaged_action_value
}

#[cfg(test)]
mod test {
    use std::default::Default;
    use tch::{Tensor, Device, nn};
    use super::*;

    struct IdentityConfig {}

    struct Identity {}

    impl SubModel for Identity {
        type Config = IdentityConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn clone_with_var_store(&self, _var_store: &nn::VarStore) -> Self {
            Self {}
        }

        fn build(_var_store: &VarStore, _config: Self::Config) -> Self {
            Self {}
        }

        fn forward(&self, input: &Self::Input) -> Self::Output {
            input.copy()
        }
    }

    fn iqn_model(feature_dim: i64, embed_dim: i64, out_dim: i64) -> IQNModel<Identity, Identity> {
        let fe_config = IdentityConfig {};
        let m_config = IdentityConfig {};
        let device = Device::Cpu;
        let learning_rate = 1e-4;

        IQNModelBuilder::default()
            .feature_dim(feature_dim)
            .embed_dim(embed_dim)
            .out_dim(out_dim)
            .learning_rate(learning_rate)
            .build(fe_config, m_config, device)
    }

    #[test]
    /// Check shape of tensors in IQNModel.
    fn test_iqn_model() {
        let in_dim = 100;
        let feature_dim = 100;
        let embed_dim = 64;
        let out_dim = 100;
        let n_percent_points = 8;
        let batch_size = 32;

        let model = iqn_model(feature_dim, embed_dim, out_dim);
        let psi = Tensor::rand(&[batch_size, in_dim], tch::kind::FLOAT_CPU);
        let tau = Tensor::rand(&[batch_size, n_percent_points], tch::kind::FLOAT_CPU);
        let _q = model.forward(&psi, &tau);
    }    
}
