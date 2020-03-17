import gym
import numpy as np

from evolution.config import EvolutionConfig
from evolution.policy import EvolutionPolicy

config = EvolutionConfig()
env = gym.make('LunarLander-v2')

policy = EvolutionPolicy(env, config)
weights = np.load('weights_200.npy')
scores = []
for i in range(1000):
    score = policy.run_episode(weights, True)
    scores.append(score)
    print("Episode {}, score: {}".format(i, score))

print("Avg score: {}".format(np.mean(scores)))
