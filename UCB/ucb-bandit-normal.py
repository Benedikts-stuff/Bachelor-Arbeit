import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# UCB-Bandit-Algorithmus-Klasse
class UCB_Bandit:
    def __init__(self, n_arms, delta):
        self.n_arms = n_arms  # Anzahl der Arme
        self.delta = delta  # Parameter für den Konfidenzbonus
        self.arm_counts = np.zeros(n_arms)  # Zählungen für jeden Arm
        self.arm_reward_means = np.zeros(n_arms)  # Durchschnittliche Belohnung für jeden Arm


    def select_arm(self):
        if sum(self.arm_counts) == 0:
            return np.random.randint(0, self.n_arms)

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


# Lade die Daten
data_path = '../facebook-ad-campaign-data.csv'
ad_data = pd.read_csv(data_path)

# Gruppiere nach Anzeigen (campaign_id)
grouped_data = ad_data.groupby('campaign_id').agg({
    'clicks': 'sum',
    'impressions': 'sum'
}).reset_index()

grouped_data['click_rate'] = (grouped_data['clicks'] / grouped_data['impressions']) * 1000
#grouped_data['click_rate'] = (grouped_data['click_rate']) / grouped_data['click_rate'].max()

n_arms = len(grouped_data)  # Anpassen hier
n_rounds = 1000
delta = 1/np.pow(n_rounds,2)
reward_history = []
bandit = UCB_Bandit(n_arms, delta)
optimal_reward = np.zeros(n_rounds)
# Simuliere die Runden
for n in range(n_rounds):
    arm = bandit.select_arm()
    click_rate = grouped_data['click_rate'].iloc[arm]
    reward = np.random.binomial(1, click_rate)
    bandit.update(arm, reward)
    reward_history.append(reward)
    optimal_reward[n] = max(grouped_data['click_rate'])
# Kumulativer Reward und Regret
cumulative_reward = np.cumsum(reward_history)

diff = np.zeros(n_arms)
for i in range(n_arms):
    diff[i] = optimal_reward[i] - bandit.arm_reward_means[i]

count = 0
for i in range(n_arms):
    if diff[i] > 0:
        count = count + ((16 * np.log(n_rounds))/diff[i])

formular = 3 * np.sum(diff) + count

print('formular', formular)
cumulative_reward_opt = np.cumsum(optimal_reward)
regret = np.abs(cumulative_reward_opt - cumulative_reward)
print(np.max(regret))
# Plot kumulierter Reward
plt.figure(figsize=(16, 10))
plt.plot(cumulative_reward, label='Kumulativer Reward')
plt.plot(cumulative_reward_opt, label='Optimaler kumulativer Reward', linestyle='--')
plt.title("Kumulierter Reward im Vergleich zum optimalen Reward")
plt.xlabel("Runden")
plt.ylabel("Kumulierter Reward")
plt.legend()
plt.savefig('cumulative_reward.pdf')
plt.show()

average_reward = np.convolve(reward_history, np.ones(100)/100, mode='valid')
plt.figure(figsize=(16, 10))
plt.plot(average_reward)
plt.title("Durchschnittlicher Reward über die Zeit")
plt.xlabel("Runden")
plt.ylabel("Durchschnittlicher Reward")
plt.savefig('averrage_reward.pdf')   # Speichern als PDF
plt.show()

plt.figure(figsize=(16, 10))
plt.bar(range(n_arms), bandit.arm_counts)
plt.title("Häufigkeit der Arm-Auswahl")
plt.xlabel("Arm (Anzeige)")
plt.ylabel("Anzahl der Ziehungen")
plt.savefig('häufigkeit_arme.pdf')
plt.show()

plt.figure(figsize=(16, 10))
plt.plot(regret)
plt.title("Kumulative Regret über die Zeit")
plt.xlabel("Runden")
plt.ylabel("Regret")
plt.savefig('cumulative_regret.pdf')
plt.show()

window_size = 1000
rewards_smoothed = pd.Series(reward_history).rolling(window=window_size).mean()
plt.figure(figsize=(16, 10))
plt.plot(rewards_smoothed)
plt.xlabel("Number of Draws")
plt.ylabel("Reward per Draw")
plt.title("Reward per Draw over Time")
plt.show()

print(bandit.arm_reward_means)