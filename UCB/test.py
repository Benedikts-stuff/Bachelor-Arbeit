import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class UCB_Bandit:
    def __init__(self, n_arms, delta, seed):
        self.seed = seed
        #np.random.seed(seed)
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





data = pd.read_csv('../Budgeted/Experiment/data/facebook-ad-campaign-data.csv')
grouped_data = data.groupby('campaign_id').agg({
    'clicks': 'sum',
    'impressions': 'sum'
}).reset_index()
grouped_data['CTR'] = grouped_data['clicks'] / grouped_data['impressions']

n_arms = len(grouped_data)
t_rounds = 1000

delta = 1/ np.pow(t_rounds,2)

arm_count = np.zeros(n_arms)

ucb_arms = np.zeros(n_arms)



reward_history = np.zeros((1,t_rounds))
for i in range(1):
    bandit = UCB_Bandit(n_arms, delta, i)
    for n in range(t_rounds):
        arm = bandit.select_arm()
        print(arm)
        ctr = grouped_data['CTR'][arm]
        #reward  = np.random.binomial(1, ctr)
        reward = ctr
        bandit.update(arm, reward)
        reward_history[i][n] = reward



# Plot kumulierter Reward
reward_all = np.cumsum(reward_history, axis=1)
cumulative_reward = np.mean(reward_all, axis=0)

np.save('cumulative_reward', cumulative_reward)

x = [0.00024408887246319506 for _ in range(t_rounds)]

cumulative_optimal_reward = np.cumsum(x)
cumulative_regret_UCB = cumulative_optimal_reward - cumulative_reward

plt.figure(figsize=(16, 10))
#plt.plot(cumulative_regret_epsilon, label='Epsilon-Greedy')
plt.plot(cumulative_regret_UCB, label='UCB')
plt.title("Durchschnittlicher regret über die Zeit")
plt.xlabel("Runden")
plt.ylabel("Regret")
plt.legend()
plt.show()