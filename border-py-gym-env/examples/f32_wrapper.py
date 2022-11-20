import gym
import numpy as np

class F32Wrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        """Resets the environment.

        Currently it assumes box observation.
        """
        obs = self.env.reset(**kwargs)
        if type(obs) == np.ndarray and obs.dtype == np.float64:
            obs = np.array(obs, dtype=np.float32)

        return obs

    def step(self, act):
        (obs, reward, terminated, truncated, info) = self.env.step(act)
        if type(obs) == np.ndarray and obs.dtype == np.float64:
            obs = np.array(obs, dtype=np.float32)

        return (obs, reward, terminated, truncated, info)

def make_f32(env_name, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    return F32Wrapper(env)
