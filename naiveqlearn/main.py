import gym
from gym.wrappers import Monitor

from naiveqlearn.config import Config
from naiveqlearn.plot import plot_from_monitor_results
from naiveqlearn.policy import Policy
from naiveqlearn.train import train
from naiveqlearn.wrapper import DiscreteLunarEnv


def main():
    env = gym.make('LunarLander-v2')
    env = Monitor(env, '/tmp/qlearn', force=True)
    env = DiscreteLunarEnv(env)

    train(Policy(env), Config())

    env.close()
    plot_from_monitor_results('/tmp/qlearn', window=50)


if __name__ == "__main__":
    main()
