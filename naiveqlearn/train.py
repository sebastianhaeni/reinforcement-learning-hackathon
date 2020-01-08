import numpy as np

from naiveqlearn.config import Config
from naiveqlearn.plot import plot_learning_curve
from naiveqlearn.policy import Policy


def train(policy: Policy, config: Config):
    reward_history = []
    reward_averaged = []
    step = 0
    alpha = config.alpha
    eps = config.epsilon

    warmup_episodes = config.warmup_episodes
    eps_drop = (config.epsilon - config.epsilon_final) / warmup_episodes

    for n_episode in range(config.n_episodes):
        ob = policy.env.reset()
        done = False
        reward = 0.

        while not done:
            a = policy.act(ob, eps)
            new_ob, r, done, info = policy.env.step(a)

            update_q_value(policy, ob, a, r, new_ob, done, alpha)

            step += 1
            reward += r
            ob = new_ob

        reward_history.append(reward)
        reward_averaged.append(np.average(reward_history[-50:]))

        alpha *= config.alpha_decay
        if eps > config.epsilon_final:
            eps = max(config.epsilon_final, eps - eps_drop)

        if n_episode % config.log_every_episode == 0:
            print("[Episode:{} | step:{}] best:{} avg:{:.4f} alpha:{:.4f} epsilon:{:.4f} len(Q):{}".format(
                n_episode, step, np.max(reward_history),
                np.mean(reward_history[-config.log_every_episode:]), alpha, eps, len(policy.Q)))

    print("[FINAL] Num. episodes: {}, Max reward: {}, Average reward: {}".format(
        len(reward_history), np.max(reward_history), np.mean(reward_history)))

    data_dict = {'reward': reward_history, 'reward_avg50': reward_averaged}
    plot_learning_curve('qlearn', data_dict, xlabel='episode')


def update_q_value(policy, ob, action, reward, new_ob, done, alpha):
    """
    Q(s, a) += alpha * (reward(s, a) + gamma * max Q(s', .) - Q(s, a))
    """
    q_prime = max([policy.Q[new_ob, a] for a in policy.actions])
    policy.Q[ob, action] += alpha * (reward + policy.gamma * q_prime * (1.0 - done) - policy.Q[ob, action])
