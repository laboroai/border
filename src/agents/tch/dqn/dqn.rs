use std::{error::Error, cell::RefCell, marker::PhantomData, path::Path, fs};
use tch::{no_grad, Kind::Float, Tensor};
use crate::core::{Policy, Agent, Step, Env};
use crate::agents::tch::{ReplayBuffer, TchBuffer, TchBatch};
use crate::agents::tch::model::Model;
use crate::agents::tch::util::track;

pub struct DQN<E, M, O, A> where
    E: Env,
    M: Model + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    n_samples_per_opt: usize,
    n_updates_per_opt: usize,
    n_opts_per_soft_update: usize,
    min_transitions_warmup: usize,
    batch_size: usize,
    qnet: M,
    qnet_tgt: M,
    train: bool,
    phantom: PhantomData<E>,
    prev_obs: RefCell<Option<E::Obs>>,
    replay_buffer: ReplayBuffer<E, O, A>,
    count_samples_per_opt: usize,
    count_opts_per_soft_update: usize,
    discount_factor: f64,
    tau: f64,
}

impl<E, M, O, A> DQN<E, M, O, A> where 
    E: Env,
    M: Model<Input=Tensor, Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(qnet: M, replay_buffer: ReplayBuffer<E, O, A>)
            -> Self {
        let qnet_tgt = qnet.clone();
        DQN {
            qnet,
            qnet_tgt,
            replay_buffer,
            n_samples_per_opt: 1,
            n_updates_per_opt: 1,
            n_opts_per_soft_update: 1,
            min_transitions_warmup: 1,
            batch_size: 1,
            discount_factor: 0.99,
            tau: 0.005,
            count_samples_per_opt: 0,
            count_opts_per_soft_update: 0,
            train: false,
            prev_obs: RefCell::new(None),
            phantom: PhantomData,
        }
    }

    pub fn n_samples_per_opt(mut self, v: usize) -> Self {
        self.n_samples_per_opt = v;
        self
    }

    pub fn n_updates_per_opt(mut self, v: usize) -> Self {
        self.n_updates_per_opt = v;
        self
    }

    pub fn n_opts_per_soft_update(mut self, v: usize) -> Self {
        self.n_opts_per_soft_update = v;
        self
    }

    pub fn min_transitions_warmup(mut self, v: usize) -> Self {
        self.min_transitions_warmup = v;
        self
    }

    pub fn batch_size(mut self, v: usize) -> Self {
        self.batch_size = v;
        self
    }

    pub fn discount_factor(mut self, v: f64) -> Self {
        self.discount_factor = v;
        self
    }

    pub fn tau(mut self, v: f64) -> Self {
        self.tau = v;
        self
    }

    fn push_transition(&mut self, step: Step<E>) {
        let next_obs = step.obs;
        let obs = self.prev_obs.replace(None).unwrap();
        let reward = Tensor::of_slice(&step.reward[..]);
        let not_done = Tensor::from(1f32) - Tensor::of_slice(&step.is_done[..]);
        self.replay_buffer.push(
            &obs,
            &step.act,
            &reward,
            &next_obs,
            &not_done,
        );
        let _ = self.prev_obs.replace(Some(next_obs));
    }

    fn update_qnet(&mut self, batch: TchBatch<E, O, A>) {
        let obs = batch.obs;
        let a = batch.actions;
        let r = batch.rewards;
        let next_obs = batch.next_obs;
        let not_done = batch.not_dones;
        let loss = {
            let pred = {
                let a = a;
                let x = self.qnet.forward(&obs);
                x.gather(-1, &a, false)
            };
            let tgt = no_grad(|| {
                let x = self.qnet_tgt.forward(&next_obs);
                let y = x.argmax(-1, false).unsqueeze(-1);
                let x = x.gather(-1, &y, false);
                r + not_done * self.discount_factor * x
            });
            pred.smooth_l1_loss(&tgt, tch::Reduction::Mean, 1.0)
        };
        self.qnet.backward_step(&loss);
    }

    fn soft_update_qnet_tgt(&mut self) {
        track(&mut self.qnet_tgt, &mut self.qnet, self.tau);
    }
}

impl<E, M, O, A> Policy<E> for DQN<E, M, O, A> where 
    E: Env,
    M: Model<Input=Tensor, Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    fn sample(&self, obs: &E::Obs) -> E::Act {
        let obs = obs.clone().into();
        let a = self.qnet.forward(&obs);
        let a = if self.train {
            a.softmax(-1, Float)
            .multinomial(1, true)
        } else {
            a.argmax(-1, true)
        };
        a.into()
    }
}

impl<E, M, O, A> Agent<E> for DQN<E, M, O, A> where
    E: Env,
    M: Model<Input=Tensor, Output=Tensor> + Clone,
    E::Obs :Into<M::Input>,
    E::Act :From<Tensor>,
    O: TchBuffer<Item = E::Obs, SubBatch = M::Input>,
    A: TchBuffer<Item = E::Act, SubBatch = Tensor>, // Todo: consider replacing Tensor with M::Output
{
    fn train(&mut self) {
        self.train = true;
    }

    fn eval(&mut self) {
        self.train = false;
    }

    fn is_train(&self) -> bool {
        self.train
    }

    fn push_obs(&self, obs: &E::Obs) {
        self.prev_obs.replace(Some(obs.clone()));
    }

    fn observe(&mut self, step: Step<E>) -> bool {
        // Push transition to the replay buffer
        self.push_transition(step);

        // Do optimization 1 step
        self.count_samples_per_opt += 1;
        if self.count_samples_per_opt == self.n_samples_per_opt {
            self.count_samples_per_opt = 0;

            if self.replay_buffer.len() >= self.min_transitions_warmup {
                for _ in 0..self.n_updates_per_opt {
                    let batch = self.replay_buffer.random_batch(self.batch_size).unwrap();
                    self.update_qnet(batch);
                };

                self.count_opts_per_soft_update += 1;
                if self.count_opts_per_soft_update == self.n_opts_per_soft_update {
                    self.count_opts_per_soft_update = 0;
                    self.soft_update_qnet_tgt();
                }
                return true;
            }
        }
        false
    }

    fn save<T: AsRef<Path>>(&self, path: T) -> Result<(), Box<dyn Error>> {
        fs::create_dir(&path)?;
        self.qnet.save(&path.as_ref().join("qnet.pt").as_path())?;
        self.qnet_tgt.save(&path.as_ref().join("qnet_tgt.pt").as_path())?;
        Ok(())
    }

    fn load<T: AsRef<Path>>(&mut self, path: T) -> Result<(), Box<dyn Error>> {
        self.qnet.load(&path.as_ref().join("qnet.pt").as_path())?;
        self.qnet_tgt.load(&path.as_ref().join("qnet_tgt.pt").as_path())?;
        Ok(())
    }
}
