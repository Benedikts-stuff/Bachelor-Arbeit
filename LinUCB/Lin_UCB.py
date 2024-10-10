import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from LinUCB.parameter_extraction import sampler, Sampler

class LinUCB:
    def __init__(self, d, K, alpha):
        self.d = d
        self.K = K
        self.alpha = alpha
        self.A = np.eye(d)
        self.b = np.zeros(d)
        self.sampler = Sampler('../facebook-ad-campaign-data.csv')
        self.action_counter = np.zeros(self.K)
        self.model1 = sampler.estimate_reward(0)
        self.model2 = sampler.estimate_reward(1)
        self.model3 = sampler.estimate_reward(2)
        self.models = [self.model1, self.model2, self.model3]
        self.reward_history = []


    def execute_UCB(self):
        n_rounds = 10000
        for n in tqdm(range(n_rounds), desc='Processing'):
            inverse =np.linalg.inv(self.A)
            theta = np.dot(inverse, self.b)
            ucb = np.zeros(self.K)
            rewards = np.zeros(self.K)
            contexts =[]
            for action in range(self.K):
                contexts.append(self.sampler.sample_context())
                ucb[action] = np.dot(theta.T, contexts[action].values[0]) + (self.alpha * np.sqrt(np.dot(contexts[action], np.dot(inverse, contexts[action].values[0]))))
            campaign = np.argmax(ucb)
            model = self.models[campaign]
            ctr = model.predict(contexts[campaign]).flatten()[0]
            rewards[campaign] = np.random.binomial(1, ctr)

            self.action_counter[campaign] += 1
            reward_a = rewards[campaign]
            self.reward_history.append(reward_a)
            self.A = self.A + np.dot(contexts[campaign], contexts[campaign].T)
            self.b = self.b + (contexts[campaign].values[0] * reward_a)


bandit = LinUCB(7, 3, 0.9)
bandit.execute_UCB()
print(bandit.action_counter)

cumulative_rerward = np.cumsum(bandit.reward_history)

plt.figure(figsize=(16, 10))
plt.plot(cumulative_rerward)
plt.title("Kumulative reward über die Zeit")
plt.xlabel("Runden")
plt.ylabel("reward")
plt.savefig('cumulative_regret.pdf')
plt.show()

plt.figure(figsize=(16, 10))
plt.bar(range(3), bandit.action_counter)
plt.title("action counter")
plt.xlabel("Actions")
plt.ylabel("Anzahl")
plt.savefig('action_counter.pdf')
plt.show()