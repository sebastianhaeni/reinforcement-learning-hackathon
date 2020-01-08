import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym.wrappers.monitor import load_results

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def plot_learning_curve(filename, value_dict, xlabel='step'):
    fig = plt.figure(figsize=(12, 4 * len(value_dict)))

    for i, (key, values) in enumerate(value_dict.items()):
        ax = fig.add_subplot(len(value_dict), 1, i + 1)
        ax.plot(range(len(values)), values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(key)
        ax.grid('k--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.join(REPO_ROOT, 'figs'), exist_ok=True)
    plt.savefig(os.path.join(REPO_ROOT, 'figs', filename))


def plot_from_monitor_results(monitor_dir, window=10):
    data = load_results(monitor_dir)
    n_episodes = len(data['episode_lengths'])
    assert n_episodes > 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True, sharex='all')

    ax1.plot(range(n_episodes), pd.DataFrame(np.array(data['episode_lengths'])).rolling(window).mean())
    ax1.set_xlabel('episode')
    ax1.set_ylabel('episode length')
    ax1.grid('k--', alpha=0.6)

    ax2.plot(range(n_episodes), pd.DataFrame(np.array(data['episode_rewards'])).rolling(window).mean())
    ax2.set_xlabel('episode')
    ax2.set_ylabel('episode reward')
    ax2.grid('k--', alpha=0.6)

    os.makedirs(os.path.join(REPO_ROOT, 'figs'), exist_ok=True)
    plt.savefig(os.path.join(REPO_ROOT, 'figs', os.path.basename(monitor_dir) + '-monitor'))
