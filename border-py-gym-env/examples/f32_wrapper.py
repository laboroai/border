import gym
import numpy as np
import inspect


class F32Wrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        name = env.unwrapped.spec.id
        if name == "AntPyBulletEnv-v0":
            self.is_pybullet_env = True
        else:
            self.is_pybullet_env = False

    def reset(self, **kwargs):
        """Resets the environment.

        Currently it assumes box observation.
        """
        obs = self.env.reset(**kwargs)

        if type(obs) == np.ndarray and obs.dtype == np.float64:
            obs = np.array(obs, dtype=np.float32)

        if self.is_pybullet_env:
            obs = (np.array(obs, dtype=np.float32), None)

        return obs

    def step(self, act):
        (obs, reward, terminated, truncated, info) = self.env.step(act)

        if type(obs) == np.ndarray and obs.dtype == np.float64:
            obs = np.array(obs, dtype=np.float32)

        return (obs, reward, terminated, truncated, info)

def make_f32(env_name, render_mode=None):
    if render_mode is not None:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)

    return F32Wrapper(env)
