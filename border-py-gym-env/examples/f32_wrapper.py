import gymnasium as gym
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
        """
        obs = self.env.reset(**kwargs)

        if type(obs) == np.ndarray and obs.dtype != np.float32:
            obs = np.array(obs, dtype=np.float32)
        elif type(obs[0]) == np.ndarray and obs[0].dtype == np.float64:
            obs = (np.array(obs[0], dtype=np.float32), obs[1])

        if self.is_pybullet_env:
            obs = (np.array(obs, dtype=np.float32), None)

        if isinstance(obs, tuple) and isinstance(obs[0], dict):
            obs_ = {}
            for (key, value) in obs[0].items():
                value_ = np.array(value, dtype=np.float32) if isinstance(value, np.ndarray) else value
                obs_[key] = value_
            obs = (obs_, obs[1])

        return obs

    def step(self, act):
        (obs, reward, terminated, truncated, info) = self.env.step(act)

        if type(obs) == np.ndarray and obs.dtype == np.float64:
            obs = np.array(obs, dtype=np.float32)

        if isinstance(obs, dict):
            obs_ = {}
            for (key, value) in obs.items():
                value_ = np.array(value, dtype=np.float32) if isinstance(value, np.ndarray) else value_
                obs_[key] = value_
            obs = obs_

        return (obs, reward, terminated, truncated, info)

def make_f32(env_name, render_mode=None):
    if render_mode is not None:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)

    return F32Wrapper(env)
