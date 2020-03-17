import numpy as np


class EvolutionPolicy:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.W = np.zeros((self.input_size, self.output_size))

    def act(self):
        rewards = np.zeros(self.config.population_size)
        noises = np.random.randn(self.config.population_size, self.input_size, self.output_size)

        # Run with mutated weights and collect rewards
        for j in range(self.config.population_size):
            weight = self.W + self.config.sigma * noises[j]
            rewards[j] = self.run_episode(weight, render=False)

        # Weigh the weights by the rewards they got
        weighted_weights = np.matmul(noises.T, rewards).T

        # Compute new weights
        self.W = self.W + self.config.alpha / (self.config.population_size * self.config.sigma) * weighted_weights

        return rewards

    def run_episode(self, weight, render):
        obs = self.env.reset()
        episode_reward = 0
        done = False
        step = 0
        while not done:
            if render:
                self.env.render()
            action = np.matmul(weight.T, obs)
            action = np.argmax(action)
            obs, reward, done, info = self.env.step(action)
            step += 1
            episode_reward += reward
        return episode_reward
