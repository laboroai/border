# TODO

* Image observation
* Logging
  * Loss (learning curve)
  * TD-error
* Composition of action noise
* Vectorized environment
* Documentation
* Add/improve RL methods
  * DDQN
  * SAC
    * Double Q
    * EntCoef auto tuning
  * DDPG
    * Fix adhoc implementation of scaling for pendulum env
      * (See sac_pendulum.rs for action scaling)
  * Prioritized experience replay
  * Parameter noise
* GPU support
* Add more examples
  * PyBullet gym
  * Atari