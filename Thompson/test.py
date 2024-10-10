import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Thompson_sampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        self.total_clicks = 0
        self.total_impressions = 0

    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)

        return np.argmax(samples)

    def update(self, arm, reward):
        self.total_clicks += reward
        self.total_impressions += 1
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


data = pd.read_csv('../facebook-ad-campaign-data.csv')

grouped_data = data.groupby('campaign_id').agg({
    'clicks': 'sum',
    'impressions': 'sum'
}).reset_index()
grouped_data['CTR'] = grouped_data['clicks'] / grouped_data['impressions']

# Normalisierte CTR in die Spalte 'CTR' schreiben

alpha = grouped_data['clicks']
beta = grouped_data['impressions']

n_arms = len(grouped_data)
t_rounds = 100000

true_conversion_rates = grouped_data['CTR'].values
reward_history = np.zeros((30,t_rounds))

bandit = Thompson_sampling(n_arms)
for i in range(30):
    for n in range(t_rounds):
        arm = bandit.select_arm()

        reward = np.random.binomial(1, true_conversion_rates[arm])
        reward_history[i][n] = reward

        bandit.update(arm, reward)


mean_rewards = np.mean(reward_history, axis=0)
cumulative_reward = np.cumsum(mean_rewards)
np.save('cumulative_reward', cumulative_reward)