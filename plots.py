import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


epsilon_reward = np.load('./epsilon-greedy/cumulative_reward.npy')
ucb_reward = np.load('./UCB/cumulative_reward.npy')
lin_ucb_reward = np.load('LinUCB/cumulative_reward_with_linear_model.npy')
lin_ucb_reward_non_linear = np.load('LinUCB/cumulative_reward.npy')
lin_ucb_reward2 = np.load('LinUCB/cumulative_reward2.npy')
thompson_reward = np.load('Thompson/cumulative_reward_not_learned.npy')
thompson_reward_learned = np.load('Thompson/cumulative_reward.npy')
optimal_ctr = 0.00024408887246319506
t_rounds= 100000
print(optimal_ctr)
x = [optimal_ctr for _ in range(t_rounds)]
cumulative_optimal_reward = np.cumsum(x)

#cumulative reward
plt.figure(figsize=(16, 10))
plt.plot(epsilon_reward, label='Epsilon-Greedy')
plt.plot(ucb_reward, label='UCB')
plt.plot(cumulative_optimal_reward, label='Erwartete optimale Belohnung')
plt.plot(thompson_reward, label='thompson' )
#plt.plot(lin_ucb_reward, label='LinUCB' )
#plt.plot(lin_ucb_reward2, label='LinUCB' )
plt.plot(thompson_reward_learned, label='Thompson_learned' )
plt.title("Kumulierter Reward über die Zeit")
plt.xlabel("Runden")
plt.ylabel("Reward")
plt.legend()  # Legende anzeigen

plt.savefig('reward.pdf')  # Speichern als PDF
plt.show()


#cumulative regret
cumulative_regret_UCB = cumulative_optimal_reward - ucb_reward
cumulative_regret_epsilon = cumulative_optimal_reward - epsilon_reward
cumulative_regret_thompson = cumulative_optimal_reward - thompson_reward
#cumulative_regret_lin_ucb = cumulative_optimal_reward - lin_ucb_reward
#cumulative_regret_lin_ucb_non_linear = cumulative_optimal_reward - lin_ucb_reward_non_linear
cumulative_regret_thompson_learned = cumulative_optimal_reward - thompson_reward_learned
plt.figure(figsize=(16, 10))
#plt.plot(cumulative_regret_epsilon, label='Epsilon-Greedy')  # Füge eine Beschriftung hinzu
plt.plot(cumulative_regret_UCB, label='UCB')  # Füge eine Beschriftung hinzu
#plt.plot(cumulative_regret_thompson, label='thompson' )
#plt.plot(cumulative_regret_lin_ucb, label='LinUCB' )
#plt.plot(cumulative_regret_lin_ucb_non_linear, label='nonlinearLinUCB' )
#plt.plot(cumulative_regret_thompson_learned, label='thompson_learned' )
plt.title("Durchschnittlicher regret über die Zeit")
plt.xlabel("Runden")
plt.ylabel("Regret")
plt.legend()

plt.savefig('regret.pdf')  # Speichern als PDF
plt.show()


plt.figure(figsize=(16, 10))
plt.plot(lin_ucb_reward2, label='LinUCBadsadad' )
plt.plot(ucb_reward, label='UCB')
plt.xlabel("Runden")
plt.ylabel("Reward")
plt.title("asdasdfasfaf")
plt.legend()
plt.show()

cumulative_optimal_reward2 = np.arange(1, 500000 + 1) * optimal_ctr
lin_ucb_regret2 = cumulative_optimal_reward2 - lin_ucb_reward2
plt.figure(figsize=(16, 10))
plt.plot(lin_ucb_regret2, label='LinUCBadsadad' )
plt.xlabel("Runden")
plt.ylabel("regret")
plt.title("asdasdfasfaf")
plt.legend()
plt.show()
