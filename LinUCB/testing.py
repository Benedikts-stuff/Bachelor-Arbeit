import numpy as np
import matplotlib.pyplot as plt


class UCB:
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.counts = np.zeros(k)
        self.values = np.zeros(k)

    def select_arm(self):
        total_counts = np.sum(self.counts)
        ucb_values = self.values + np.sqrt(2 * np.log(total_counts) / (self.counts + 1))
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]



class LinUCB:
    def __init__(self, d, k, alpha):
        self.d = d
        self.k = k
        self.alpha = alpha
        self.A = [np.identity(d) for _ in range(k)]
        self.b = [np.zeros(d) for _ in range(k)]
        self.pseudo_inverse = [np.zeros((d, d)) for _ in range(k)]

    def select_arm(self, x):
        p = np.zeros(self.k)
        for arm in range(self.k):
            self.pseudo_inverse[arm] = np.linalg.inv(self.A[arm])
            p[arm] = x.T @ self.pseudo_inverse[arm] @ self.b[arm] + self.alpha * np.sqrt(x.T @ self.pseudo_inverse[arm] @ x)
        return np.argmax(p)

    def update(self, chosen_arm, reward, x):
        self.A[chosen_arm] += np.outer(x, x)
        self.b[chosen_arm] += reward * x


def simulate_bandits(k, n, d):
    # Erzeuge k Einheitsvektoren auf der Einheitssphäre
    arms = np.random.randn(k, d)
    arms /= np.linalg.norm(arms, axis=1, keepdims=True)

    # Generiere unbekannte Parameter θ auf der Einheitssphäre
    theta = np.random.randn(d)
    theta /= np.linalg.norm(theta)

    # Simuliere die Belohnungen
    rewards = np.dot(arms, theta) + np.random.normal(0, 1, (n, k))  # Rauschen hinzufügen

    return arms, theta, rewards


def run_experiment(k_values, n, d):
    regrets_ucb = []
    regrets_linucb = []

    for k in k_values:
        arms, theta, rewards = simulate_bandits(k, n, d)

        # UCB
        ucb = UCB(k, n)
        total_reward_ucb = 0

        for t in range(n):
            chosen_arm = ucb.select_arm()
            reward = rewards[t, chosen_arm]
            total_reward_ucb += reward
            ucb.update(chosen_arm, reward)

        # LinUCB
        linucb = LinUCB(d, k, alpha=1)
        total_reward_linucb = 0

        for t in range(n):
            x = arms[t % k]  # Feature-Vektor des Arms
            chosen_arm = linucb.select_arm(x)
            reward = rewards[t, chosen_arm]
            total_reward_linucb += reward
            linucb.update(chosen_arm, reward, x)

        # Regret berechnen
        optimal_reward = np.max(rewards.sum(axis=0))  # Optimale Belohnung
        regret_ucb = optimal_reward - total_reward_ucb
        regret_linucb = optimal_reward - total_reward_linucb

        regrets_ucb.append(regret_ucb)
        regrets_linucb.append(regret_linucb)

    return regrets_ucb, regrets_linucb


k_values = np.arange(2, 1001)  # k-Werte von 2 bis 1000
n = 5000  # Anzahl der Proben
d = 5  # Dimension der Features

regrets_ucb, regrets_linucb = run_experiment(k_values, n, d)

plt.figure(figsize=(12, 6))
plt.plot(k_values, regrets_ucb, label='UCB Regret', marker='o')
plt.plot(k_values, regrets_linucb, label='LinUCB Regret', marker='x')
plt.title('Expected Regret of UCB vs LinUCB')
plt.xlabel('Number of Arms (k)')
plt.ylabel('Expected Regret')
plt.legend()
plt.grid()
plt.show()