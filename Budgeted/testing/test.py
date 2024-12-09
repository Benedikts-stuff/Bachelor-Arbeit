import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot as plt

class ContextualBanditClassifier:
    def __init__(self, num_arms, context_dim):
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.model = [GradientBoostingClassifier() for _ in range(num_arms)]  # Beliebiges Klassifikationsmodell
        self.data = {arm: [] for arm in range(num_arms)}
        self.labels = {arm: [] for arm in range(num_arms)}

    def choose_arm(self, context):
        if len(self.data) < self.num_arms * 10:  # Frühzeitige Exploration
            return np.random.choice(self.num_arms)
        else:
            probs = []
            for arm in range(self.num_arms):
                x = np.hstack((context, [arm]))
                probs.append(self.model[arm].predict_proba([x])[0, 1])  # Wahrsch. für Reward=1
            return np.argmax(probs)

    def update(self, context, chosen_arm, reward):
        # Daten speichern
        x = context
        self.data[chosen_arm].append(x)
        self.labels[chosen_arm].append(reward)
        # Modell trainieren
        if 1 in self.labels[chosen_arm] and 0 in self.labels[chosen_arm]:
            self.model[chosen_arm].fit(self.data[chosen_arm], self.labels[chosen_arm])

# Beispiel mit 3 Armen und 2-dimensionalem Kontext
num_arms = 3
context_dim = 2
contexts = np.random.rand(1000, context_dim)
regret = np.zeros(1000)
for i in range(20):
    obs =[]
    opt = []
    bandit = ContextualBanditClassifier(num_arms, context_dim)
    for t in range(1000):
        print(t)
        context = contexts[t]
        chosen_arm = bandit.choose_arm(context)
        true_rewards = [0.1 + 0.2 * context[0], 0.3 + 0.1 * context[1], 0.2 + 0.2 * context[0]]
        reward = np.random.binomial(1, true_rewards[chosen_arm])
        obs.append(reward)
        opt.append(np.random.binomial(1,np.max(true_rewards)))
        bandit.update(context, chosen_arm, reward)
    regret = np.add(regret, np.array(opt) - np.array(obs))

regret = regret / 20
plt.subplot(122)
plt.plot(regret.cumsum(), label='ts model')
plt.title("Cumulative regret")
plt.legend()
plt.show()