import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import argmax
from scipy.stats import beta

class Thompson_Bandit:
    def __init__(self, n_arms, p , q):
        self.n_arms = n_arms
        self.q = q #failures +1
        self.p = p #successes +1
        self.arm_counts = np.zeros(self.n_arms)

    def choose_arm(self):
        sampled_theta = [beta.rvs(a=self.p[i], b=self.q[i]) for i in range(self.n_arms)]
        n = argmax(sampled_theta)
        self.arm_counts[n] += 1
        return n

    def simulate_env(self, arm, true_conversion_rates: np.ndarray):
        return 1 if np.random.rand() <= true_conversion_rates[arm] else 0

    def update(self, arm, reward):
        if reward == 1:
            self.p[arm] += 1
        else:
            self.q[arm] += 1

data_path = '../facebook-ad-campaign-data.csv'
ad_data = pd.read_csv(data_path)

grouped_data = ad_data.groupby('campaign_id').agg({
    'clicks': 'sum',
    'impressions': 'sum'
}).reset_index()

n_arms = len((grouped_data)['campaign_id'].unique())
n_rounds =1000
alpha_params = np.empty(n_arms)
beta_params = np.empty(n_arms)
conversion_rate = np.empty(n_arms)

for i in range(n_arms):
    alpha_params[i] = (grouped_data['clicks'].iloc[i]) + 1
    beta_params[i] = (grouped_data['impressions'].iloc[i]) - grouped_data['clicks'].iloc[i] + 1

    if grouped_data['impressions'].iloc[i] > 0:
        conversion_rate[i] = grouped_data['clicks'].iloc[i] / grouped_data['impressions'].iloc[i]
    else:
        conversion_rate[i] = 0
print(conversion_rate)
ts = Thompson_Bandit(n_arms, alpha_params, beta_params)
rewards = []
for n in range(n_rounds):
    arm = ts.choose_arm()
    reward = ts.simulate_env(arm, conversion_rate)
    ts.update(arm, reward)
    rewards.append(reward)


plt.figure(figsize=(16, 10))
plt.bar(range(n_arms), ts.arm_counts)
plt.title("Häufigkeit der Arm-Auswahl")
plt.xlabel("Arm (Anzeige)")
plt.ylabel("Anzahl der Ziehungen")
plt.savefig('häufigkeit_arme_thompson.pdf')
plt.show()