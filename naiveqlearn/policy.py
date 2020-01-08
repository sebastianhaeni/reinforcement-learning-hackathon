from collections import defaultdict

import gym
import numpy as np


class Policy:
    def __init__(self, env: gym.Env, gamma=0.99):
        self.env = env
        self.Q = defaultdict(float)
        self.actions = range(env.action_space.n)
        self.gamma = gamma

    def act(self, state, eps=0.1):
        if eps > 0. and np.random.rand() < eps:
            return self.env.action_space.sample()

        max_q = max([self.Q[state, a] for a in self.actions])
        best_actions = [a for a in self.actions if self.Q[state, a] == max_q]
        return np.random.choice(best_actions)
