use anyhow::Result;
use border_core::{Agent, Configurable};
use border_edge_policy::Mlp;
use border_tch_agent::{
    mlp,
    model::ModelBase,
    sac::{ActorConfig, CriticConfig, SacConfig},
};

const DIM_OBS: i64 = 3;
const DIM_ACT: i64 = 1;

// Dummy types
mod dummy {
    use super::mlp::{Mlp, Mlp2};
    use border_tch_agent::sac::Sac as Sac_;

    #[derive(Clone, Debug)]
    pub struct DummyObs;

    impl border_core::Obs for DummyObs {
        fn dummy(_n: usize) -> Self {
            unimplemented!();
        }

        fn len(&self) -> usize {
            unimplemented!();
        }
    }

    impl Into<tch::Tensor> for DummyObs {
        fn into(self) -> tch::Tensor {
            unimplemented!();
        }
    }

    #[derive(Clone, Debug)]
    pub struct DummyAct;

    impl border_core::Act for DummyAct {
        fn len(&self) -> usize {
            unimplemented!();
        }
    }

    impl Into<tch::Tensor> for DummyAct {
        fn into(self) -> tch::Tensor {
            unimplemented!();
        }
    }

    impl From<tch::Tensor> for DummyAct {
        fn from(_value: tch::Tensor) -> Self {
            unimplemented!();
        }
    }

    #[derive(Clone)]
    pub struct DummyInnerBatch;

    impl Into<tch::Tensor> for DummyInnerBatch {
        fn into(self) -> tch::Tensor {
            unimplemented!();
        }
    }

    pub struct DummyBatch;

    impl border_core::TransitionBatch for DummyBatch {
        type ObsBatch = DummyInnerBatch;
        type ActBatch = DummyInnerBatch;

        fn len(&self) -> usize {
            unimplemented!();
        }

        fn obs(&self) -> &Self::ObsBatch {
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

    pub struct DummyReplayBuffer;

    impl border_core::ReplayBufferBase for DummyReplayBuffer {
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
    pub struct DummyInfo;

    impl border_core::Info for DummyInfo {}

    pub struct DummyEnv;

    impl border_core::Env for DummyEnv {
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

        fn step(&mut self, _a: &Self::Act) -> (border_core::Step<Self>, border_core::record::Record)
        where
            Self: Sized,
        {
            unimplemented!();
        }

        fn step_with_reset(
            &mut self,
            _a: &Self::Act,
        ) -> (border_core::Step<Self>, border_core::record::Record)
        where
            Self: Sized,
        {
            unimplemented!();
        }
    }

    pub type Env = DummyEnv;
    pub type Sac = Sac_<Env, Mlp, Mlp2, DummyReplayBuffer>;
}

use dummy::Sac;

fn create_sac_config() -> SacConfig<mlp::Mlp, mlp::Mlp2> {
    // Omit learning related parameters
    let actor_config = ActorConfig::default()
        .out_dim(DIM_ACT)
        .pi_config(mlp::MlpConfig::new(DIM_OBS, vec![64, 64], DIM_ACT, false));
    let critic_config = CriticConfig::default().q_config(mlp::MlpConfig::new(
        DIM_OBS + DIM_ACT,
        vec![64, 64],
        1,
        false,
    ));
    SacConfig::default()
        .actor_config(actor_config)
        .critic_config(critic_config)
        .device(tch::Device::Cpu)
}

fn main() -> Result<()> {
    let src_path = "./border/examples/gym/model/tch/sac_pendulum/best";
    let dest_path = "";

    // Load Sac model
    let sac = {
        let config = create_sac_config();
        let mut sac = Sac::build(config);
        sac.load_params(src_path)?;
        sac
    };

    // Check variables in the VarStore
    let mlp = {
        let vs = sac.get_policy_net().get_var_store();
        let w_names = ["mlp.al0.weight", "mlp.al1.weight", "ml.weight"];
        let b_names = ["mlp.al0.bias", "mlp.al1.bias", "ml.bias"];
        Mlp::from_varstore(vs, &w_names, &b_names)
    };

    // println!("{:?}", mlp);

    Ok(())
}
