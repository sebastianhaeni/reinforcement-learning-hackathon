import numpy as np

from evolution.config import EvolutionConfig
from evolution.policy import EvolutionPolicy


def train(policy: EvolutionPolicy, config: EvolutionConfig):
    rewards = []
    n_generations = 0

    for gen in range(config.generation_limit):
        gen_rewards = policy.act()
        gen_mean = np.mean(gen_rewards)

        rewards.append(gen_mean)
        n_generations += 1

        print("Generation {}, avg: {}".format(gen, gen_mean))
        if gen_mean >= config.score_requirement:
            break

    np.save('weights', policy.W)
    return rewards, n_generations
