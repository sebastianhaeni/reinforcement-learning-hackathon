import gym
import numpy as np


class DiscreteLunarEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        pos_x = observation[0]
        pos_y = observation[1]
        vel_y = observation[3]
        limited_ob = np.array([pos_x, pos_y, vel_y])
        return tuple((limited_ob * 10).astype('int'))
