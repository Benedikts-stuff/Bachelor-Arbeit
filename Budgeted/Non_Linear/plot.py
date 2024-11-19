from gaussian_process_ucb import GPUCB
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# Parameter
n_arms = 5
n_rounds = 10000
beta_t = 2 # Explorationsgewicht (β_t)
n_features = 8
noise_std = 0.1  #  knee finding method anpassen ( prüfen ob der algo die verteilung gelernt hat)
train= [10, 20, 50 ,100, 200, 300, 500,700, 800, 1000, 2500, 5000, 7500,11000, 18000]
num_points = 150
X_test = np.linspace(-3, 3, num_points).reshape(-1, 1)

repetitions = 1
regret = np.zeros(n_rounds)

for i in range(repetitions):
    gpucb_bandit = GPUCB(n_arms, n_features, n_rounds, beta_t, train, i)
    gpucb_bandit.run()
    regret = np.add(regret,np.array(gpucb_bandit.opt_reward) - np.array(gpucb_bandit.observed_rewards))

regret = regret / repetitions
plt.subplot(122)
plt.plot(regret.cumsum(), label='exponential reward')
plt.title("Cumulative regret")
plt.legend()
plt.show()