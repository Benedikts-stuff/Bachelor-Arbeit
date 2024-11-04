import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

class EpsilonGreedyContextualBandit:
    def __init__(self, d, epsilon,num_rounds,  n_arms, contexts, true_weights, true_cost, budget):
        """
        d: Dimension der Kontextvektoren
        epsilon: Wahrscheinlichkeit für Exploration
        """
        self.n_features = d
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.B = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.budget = budget
        self.costs = true_cost
        self.i = 0
        self.num_rounds = num_rounds

        self.contexts = contexts
        self.true_weights = true_weights
        self.observed_reward_history = []
        self.actual_reward_history = self.contexts.dot(self.true_weights.T)

    def select_arm(self, context):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)  # Exploration
        else:
            expected_rewards = np.array([np.dot(self.mu_hat[i], context) for i in range(self.n_arms)])
            return np.argmax(expected_rewards)  # Exploitation

    def update(self, reward, chosen_arm, context):
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.mu_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])
        self.budget = self.budget - self.costs[chosen_arm]

    def run(self):
         for i in range(self.num_rounds):
            context = self.contexts[i]
            chosen_arm = self.select_arm(context)
            actual_reward = np.dot(self.true_weights[chosen_arm], context)
            self.update(actual_reward, chosen_arm, context)
            self.observed_reward_history.append(actual_reward)



# Set parameters
num_arms = 3
num_features = 3
num_rounds = 1000
true_weights = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.2], [0.5, 0.2, 0.3]])
context = np.random.rand(num_rounds, num_features)
epsilon = 0.1  # Exploration rate
true_cost= np.array([0.1, 0.4, 0.2])
budget = 1000
bandit_eg = EpsilonGreedyContextualBandit(num_features, epsilon, num_rounds, num_arms, context, true_weights, true_cost, budget)
bandit_eg.run()


def optim_reward(true_cost, reward_history):
    optimal_reward = []
    for i in range(len(reward_history)):
        r_i = reward_history[i]
        c_r = r_i / true_cost
        optimal_reward.append(np.max(c_r))
    return np.array(optimal_reward)


# Plot comparison
optimal_reward = bandit_eg.actual_reward_history.max(axis=1)
regret_eg = optimal_reward - np.array(bandit_eg.observed_reward_history)





# Kumulative Belohnung
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(np.cumsum(optimal_reward), label="Optimal Kumulative Belohnung", color='red')
plt.plot(np.cumsum(bandit_eg.observed_reward_history), label="Epsilon-Greedy Kumulative Belohnung", color='green')
plt.xlabel("Runden")
plt.ylabel("Kumulative Belohnung")
plt.title("Kumulative Belohnung über Zeit")
plt.legend()

# Regret
plt.subplot(2, 1, 2)
plt.plot(np.cumsum(regret_eg), label="Epsilon-Greedy Regret", color='green')
plt.xlabel("Runden")
plt.ylabel("Regret")
plt.title("Regret über Zeit")
plt.legend()

plt.tight_layout()
plt.show()
