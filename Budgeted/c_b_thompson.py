import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class ThompsonSamplingContextualBandit:
    def __init__(self, d, v, n_arms, n_rounds, contexts, true_weights):
        """
        d: Dimension der Kontextvektoren
        v: Varianzparameter für die Normalverteilung
        B: Kovarianz Matrix
        """
        self.n_features = d
        self.v = v
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.B =np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]

        self.contexts = contexts
        self.true_weights = true_weights
        self.obeserved_reward_history = []
        self.actual_reward_history = self.contexts.dot(self.true_weights.T)#[self.true_weights[i].T * self.contexts for i in range(self.n_arms)]

    def sample_mu(self):
        return np.array([np.random.multivariate_normal(self.mu_hat[i], self.v ** 2 * np.linalg.inv(self.B[i])) for i in range(self.n_arms)])

    def update(self, reward, chosen_arm, context):
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.mu_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])

    def run(self):
        for i in range(self.n_rounds):
            context = self.contexts[i]
            sampled_mu = self.sample_mu()
            expected_rewards = [np.dot(sampled_mu[i], context) for i in range(self.n_arms)]
            chosen_arm = np.argmax(expected_rewards)
            actual_reward = np.dot(self.true_weights[chosen_arm], context)

            self.update(actual_reward, chosen_arm, context)
            self.obeserved_reward_history.append(actual_reward)


#set parameters
num_arms = 3
num_features = 3
num_rounds = 2500
true_weights = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.2], [0.5, 0.2, 0.3]])
context = np.random.rand(num_rounds, num_features)
v = 0.6

#run the bandit
bandit = ThompsonSamplingContextualBandit(num_features, v, num_arms, num_rounds, context, true_weights)
bandit.run()

#plot regret
optimal_reward = bandit.actual_reward_history.max(axis=1)
regret = optimal_reward - bandit.obeserved_reward_history

# Plot der kumulierten Belohnung
plt.subplot(2, 1, 1)
plt.plot(np.cumsum(optimal_reward), label="Optimal Kumulative Belohnung", color='red')
plt.plot(np.cumsum(bandit.obeserved_reward_history), label="Kumulative Belohnung", color='blue')
plt.xlabel("Runden")
plt.ylabel("Kumulative Belohnung")
plt.title("Kumulative Belohnung über Zeit")
plt.legend()

# Plot des Regret
plt.subplot(2, 1, 2)
plt.plot(np.cumsum(regret), label="Regret", color='red')
plt.xlabel("Runden")
plt.ylabel("Regret")
plt.title("Regret über Zeit")
plt.legend()

plt.tight_layout()
plt.show()