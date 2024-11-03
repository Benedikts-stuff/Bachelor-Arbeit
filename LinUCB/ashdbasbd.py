import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class UCB_Bandit:
    def __init__(self, n_arms, delta, seed):
        self.seed = seed
        np.random.seed(seed)
        self.n_arms = n_arms
        self.delta = delta
        self.arm_counts = np.zeros(n_arms)
        self.arm_reward_means = np.zeros(n_arms)


    def select_arm(self):
        ucb_of_arms = np.full(self.n_arms, np.inf)
        for i in range(self.n_arms):
            if self.arm_counts[i] == 0:
                continue
            else:
                ucb_of_arms[i] = self.arm_reward_means[i] + np.sqrt((2 * np.log(1 / self.delta)) / self.arm_counts[i])

        return np.argmax(ucb_of_arms)

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        self.arm_reward_means[arm] = ((n - 1) * self.arm_reward_means[arm] + reward) / n





data = pd.read_csv('./data_linear.csv')
grouped_data = data.groupby('campaign_id')['ctr'].mean().reset_index()

n_arms = len(grouped_data)
t_rounds = 500000

delta = 0.1

arm_count = np.zeros(n_arms)

ucb_arms = np.zeros(n_arms)



reward_history = np.zeros((10,t_rounds))
for i in range(10):
    bandit = UCB_Bandit(n_arms, delta, i)
    for n in range(t_rounds):
        arm = bandit.select_arm()
        ctr = grouped_data['ctr'][arm]
        reward  = np.random.binomial(1, ctr)
        bandit.update(arm, reward)
        reward_history[i][n] = reward



# Plot kumulierter Reward
reward_all = np.cumsum(reward_history, axis=1)
cumulative_reward = np.mean(reward_all, axis=0)

np.save('cumulative_reward', cumulative_reward)

plt.figure(figsize=(16, 10))
plt.plot(cumulative_reward)
plt.plot(np.load('cumulative_reward2.npy'))
plt.title("Häufigkeit der Arm-Auswahl")
plt.xlabel("Arm (Anzeige)")
plt.ylabel("Anzahl der Ziehungen")
plt.savefig('häufigkeit_arme_thompson_normal.pdf')
plt.show()