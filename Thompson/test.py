import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Thompson_sampling:
    def __init__(self, n_arms, alpha, beta, seed):
        self.seed = seed
        np.random.seed(self.seed)
        self.n_arms = n_arms
        self.alpha = alpha
        self.beta = beta
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

#alpha = grouped_data['clicks'].values +1
#beta = (grouped_data['impressions'].values - grouped_data['clicks'].values) + 1
alpha = np.ones(3)
beta = np.ones(3)
n_arms = len(grouped_data)
t_rounds = 100000

true_conversion_rates = grouped_data['CTR'].values
reward_history = np.zeros((100,t_rounds))


for i in range(100):
    bandit = Thompson_sampling(n_arms, alpha, beta,i)
    for n in range(t_rounds):
        arm = bandit.select_arm()

        reward = np.random.binomial(1, true_conversion_rates[arm])
        reward_history[i][n] = reward

        bandit.update(arm, reward)

reward_all = np.cumsum(reward_history, axis=1)
cumulative_reward = np.mean(reward_all, axis=0)
np.save('cumulative_reward', cumulative_reward)