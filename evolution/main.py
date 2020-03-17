import gym
import numpy as np
from gym.wrappers import Monitor

from evolution.config import EvolutionConfig
from evolution.plot import create_plot
from evolution.policy import EvolutionPolicy
from evolution.train import train


def main():
    config = EvolutionConfig()

    env = gym.make('LunarLander-v2')
    env = Monitor(env, '/tmp/evolution', force=True)
    env.seed(config.seed)
    np.random.seed(config.seed)

    policy = EvolutionPolicy(env, config)
    # Continue training
    #policy.W = np.load('weights.npy')

    rewards, n_generations = train(policy, config)

    env.close()

    create_plot(rewards, n_generations)


if __name__ == "__main__":
    main()
