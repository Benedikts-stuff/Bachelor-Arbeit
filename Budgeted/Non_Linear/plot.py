from gaussian_process_ucb import GPUCB
from neuralTS import run
from gaussian_process_ts import GPTS
import numpy as np
import matplotlib.pyplot as plt
import torch


np.random.seed(42)
# Parameter
n_arms = 5
n_rounds = 2000
beta_t = [2,3,4] # Explorationsgewicht (β_t)
lambda_ = [1, 1.5, 2]
n_features = 3
noise_std = 0.1  #  knee finding method anpassen ( prüfen ob der algo die verteilung gelernt hat)
#exponentiell wachsen lassen? Dauert sehr lange nach einigen runden (ich sollte nur batches verweden und nicht alle daten)
# könnte bei regelmässigen updates könnte er auch im adverserial setting recht gut abschneiden
train= [10, 20, 50 ,100, 200, 300, 500,700, 800, 1000, 2500, 5000, 7500]
num_points = 150

gamma = [0.1]

repetitions = 1
regret_ucb = np.zeros(n_rounds)
regret_ts = np.zeros(n_rounds)
regret_gp_ts = np.zeros(n_rounds)

for i in range(repetitions):
    np.random.seed(i)
    torch.manual_seed(i)
    context = [np.random.uniform(-1, 1, n_features) for i in range(n_rounds)]
    #context_ts = [torch.tensor(context, dtype=torch.float32) for context in context]
    #gpucb_bandit = GPUCB(n_arms, n_features, n_rounds, np.random.choice(beta_t), train, i, context, np.random.choice(gamma))
    #gpucb_bandit.run()
    #regret_ucb = np.add(regret_ucb,np.array(gpucb_bandit.opt_reward) - np.array(gpucb_bandit.observed_rewards))

    #NeuralTS
    #regret_ts += run(i, context_ts)

    #Gaussian Process TS
    #mit lamda 10 und
    gp_ts = GPTS(n_arms, n_features, n_rounds, 10, train, i, context, np.random.choice(gamma))
    gp_ts.run()
    regret_gp_ts += (np.array(gp_ts.opt_reward) - np.array(gp_ts.observed_rewards))




#regret_ts = regret_ts / repetitions
#regret_ucb = regret_ucb / repetitions
regret_gp_ts = regret_gp_ts / repetitions
plt.subplot(122)
#plt.plot(regret_ts, label='Neural TS')
#plt.plot(regret_ucb.cumsum(), label='GP UCB')
plt.plot(regret_gp_ts.cumsum(), label='GP TS')
plt.title("Cumulative regret")
plt.legend()
plt.show()