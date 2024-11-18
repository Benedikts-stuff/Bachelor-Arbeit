from adversary import Adversary
from ucb import UCB_Bandit
import matplotlib.pyplot as plt
from exp3 import EXP3
import numpy as np
from tqdm import tqdm

np.random.seed(42)

n_arms = 3
n_rounds = 1000
delta = 0.8
eta = 0.1
probs = [0.2, 0.3, 0.5]
repetitions = 30

rewards1 = np.zeros(n_rounds)
rewards2 = np.zeros(n_rounds)

for i in range(repetitions):
    adversary = Adversary(probs, seed=i)
    bandit_exp = EXP3(eta, n_rounds, n_arms, adversary,i)
    bandit_ucb = UCB_Bandit(n_arms, delta, n_rounds, adversary, i)
    bandit_exp.run()
    bandit_ucb.run()
    rewards1 = np.add(rewards1, bandit_exp.reward_history)
    rewards2 = np.add(rewards2, bandit_ucb.reward_history)


optimal_reward = np.full(n_rounds, np.max(probs))
regret_ucb= np.cumsum(optimal_reward) - np.cumsum(np.array(rewards1)/repetitions)
regret_exp3 =np.cumsum(optimal_reward) - np.cumsum(np.array(rewards2)/repetitions)


plt.plot(regret_ucb, label='ucb')
plt.plot(regret_exp3, label='exp3')
plt.title("Cumulative regret")
plt.legend()
plt.show()