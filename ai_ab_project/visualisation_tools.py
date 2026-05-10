import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
matplotlib.use('Agg')


def plot_rewards(rewards, episode_count):
    plt.plot(episode_count, rewards)
    plt.ylabel('Total reward')
    
    plt.xlabel('Episode number')
    plt.title('Reward History')
    plt.savefig('plots/reward_history_1_more_episodes_even_more_episodes.png')
