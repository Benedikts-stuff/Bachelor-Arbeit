import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import inv
import matplotlib.pyplot as plt
import tqdm

class KernelizedUCB:
    def __init__(self, n_arms, kernel='rbf', length_scale=1.0, lambda_param=1.0, delta=0.1):
        self.n_arms = n_arms
        self.kernel = kernel
        self.length_scale = length_scale
        self.lambda_param = lambda_param
        self.delta = delta

        # Speicher für Kontext, Belohnungen und Kernels
        self.contexts = {arm: [] for arm in range(n_arms)}
        self.rewards = {arm: [] for arm in range(n_arms)}
        self.K_inverse = {arm: None for arm in range(n_arms)}  # Inverse des Kernel-Matrix

    def rbf_kernel(self, X, Y):
        """Berechnet den RBF-Kernel zwischen X und Y."""
        dist = cdist(X / self.length_scale, Y / self.length_scale, metric='sqeuclidean')
        return np.exp(-0.5 * dist)

    def update(self, arm, context, reward):
        """Aktualisiert die gespeicherten Daten und die Kernel-Inverse."""
        self.contexts[arm].append(context)
        self.rewards[arm].append(reward)

        X = np.array(self.contexts[arm])
        K = self.rbf_kernel(X, X) + self.lambda_param * np.eye(len(X))
        self.K_inverse[arm] = inv(K)

    def predict(self, arm, context):
        """Sagt den Erwartungswert und die Unsicherheit für einen Kontext voraus."""
        if len(self.contexts[arm]) == 0:
            return 0, float('inf')  # Maximale Unsicherheit für unerforschte Arme

        X = np.array(self.contexts[arm])
        K_inv = self.K_inverse[arm]
        k_star = self.rbf_kernel(X, context.reshape(1, -1))
        k_star_star = self.rbf_kernel(context.reshape(1, -1), context.reshape(1, -1))

        mu = k_star.T @ K_inv @ np.array(self.rewards[arm])
        sigma = k_star_star - k_star.T @ K_inv @ k_star
        return mu.item(), sigma.item()

    def select_arm(self, context, t):
        """Wählt den Arm basierend auf dem UCB-Kriterium."""
        ucb_values = []
        beta_t = np.sqrt(2 * np.log(1 / self.delta)) + np.sqrt(self.lambda_param)

        for arm in range(self.n_arms):
            mu, sigma = self.predict(arm, context)
            ucb = mu + beta_t * np.sqrt(max(sigma, 0))
            ucb_values.append(ucb)

        return np.argmax(ucb_values)

    def run(self, contexts, reward_generator):
        """Führt den Algorithmus mit gegebenen Kontexten aus."""
        reward_history = []
        optimal_reward_history = []

        for t, context in enumerate(contexts):
            arm = self.select_arm(context, t)
            rewards = reward_generator(context)
            reward_history.append(rewards[arm])
            optimal_reward_history.append(np.max(rewards))
            self.update(arm, context, rewards[arm])

        return reward_history, optimal_reward_history


# Beispiel: Linearer Reward-Generator
def linear_reward_generator(context, n_arms, true_weights, noise_std=0.1):
    rewards = []
    for arm in range(n_arms):
        reward = np.dot(context, true_weights[arm]) # + np.random.normal(0, noise_std)
        rewards.append(np.clip(reward, 0, 1))
    return rewards


# Beispielaufruf
if __name__ == "__main__":
    np.random.seed(42)

    n_arms = 3
    d = 5  # Dimension der Kontexte
    true_weights = [np.random.rand(d) for _ in range(n_arms)]
    contexts = [np.random.uniform(-1, 1, d) for _ in range(10000)]

    k_ucb = KernelizedUCB(n_arms, length_scale=1.0)
    reward_history, optimal_reward_history = k_ucb.run(
        contexts, lambda context: linear_reward_generator(context, n_arms, true_weights)
    )

    # Plot der Ergebnisse
    cumulative_reward = np.cumsum(reward_history)
    optimal_cumulative_reward = np.cumsum(optimal_reward_history)

    regret = optimal_cumulative_reward - cumulative_reward

    #plt.plot(cumulative_reward, label="Kernelized UCB")
   # plt.plot(optimal_cumulative_reward, label="Optimal")
    plt.plot(regret, label= "regret kernel UCB rbf")
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.show()
