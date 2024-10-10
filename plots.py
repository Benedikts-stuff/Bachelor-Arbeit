import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


epsilon_reward = np.load('./epsilon-greedy/cumulative_reward.npy')
ucb_reward = np.load('./UCB/cumulative_reward.npy')
optimal_reward = np.load('./epsilon-greedy/cumulative_reward_optimal.npy')


#cumulative reward
plt.figure(figsize=(16, 10))
plt.plot(epsilon_reward, label='Epsilon-Greedy')  # Füge eine Beschriftung hinzu
plt.plot(ucb_reward, label='UCB')  # Füge eine Beschriftung hinzu
plt.plot(optimal_reward, label='Optimal' )
plt.title("Kumulierter Reward über die Zeit")
plt.xlabel("Runden")
plt.ylabel("Reward")
plt.legend()  # Legende anzeigen

plt.savefig('reward.pdf')  # Speichern als PDF
plt.show()


#cumulative regret
cumulative_regret_UCB = optimal_reward - ucb_reward
cumulative_regret_epsilon = optimal_reward - epsilon_reward

plt.figure(figsize=(16, 10))
plt.plot(cumulative_regret_epsilon, label='Epsilon-Greedy')  # Füge eine Beschriftung hinzu
plt.plot(cumulative_regret_UCB, label='UCB')  # Füge eine Beschriftung hinzu
plt.title("Durchschnittlicher regret über die Zeit")
plt.xlabel("Runden")
plt.ylabel("Regret")
plt.legend()  # Legende anzeigen

plt.savefig('regret.pdf')  # Speichern als PDF
plt.show()


