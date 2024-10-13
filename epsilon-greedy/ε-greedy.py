import pandas as pd
import numpy as np
import random


class Epsilon_greedy_bandit:
    def __init__(self, epsilon, n, seed):
        self.epsilon = epsilon
        self.n = n
        self.seed = seed
        self.mean_arms = np.zeros(self.n)
        self.mean_arms.fill(np.inf)
        self.arm_counts= np.zeros(self.n)
        random.seed(self.seed)


    def get_arm(self):
        explore = np.random.binomial(1, self.epsilon)
        if explore and np.sum(self.arm_counts) >= n:
           return random.randint(0, self.n - 1)
        else:
            return  np.argmax(self.mean_arms)


    def update_means(self, arm, reward):
        if self.arm_counts[arm] == 0:
            self.mean_arms[arm] = 0

        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        self.mean_arms[arm] = ((n - 1) * self.mean_arms[arm] + reward) / n




data = pd.read_csv('../facebook-ad-campaign-data.csv')

grouped_data = data.groupby('campaign_id').agg({
    'clicks': 'sum',
    'impressions': 'sum'
}).reset_index()
grouped_data['CTR'] = grouped_data['clicks'] / grouped_data['impressions']

# Normalisierte CTR in die Spalte 'CTR' schreiben

n_arms = len(grouped_data)
t_rounds = 100000
e = 0.1
reward_history = np.zeros((100,t_rounds))

#play bandit
for i in range(100):
    bandit = Epsilon_greedy_bandit(e, n_arms, i)

    for n in range(t_rounds):
        arm = bandit.get_arm()
        click_rate = grouped_data['CTR'].iloc[arm]
        reward = np.random.binomial(1, click_rate)
        bandit.update_means(arm, reward)
        reward_history[i][n] = reward

reward_all = np.cumsum(reward_history, axis=1)
cumulative_reward = np.mean(reward_all, axis=0)

np.save('cumulative_reward', cumulative_reward)






