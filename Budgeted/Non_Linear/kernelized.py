import numpy as np
from pyparsing import Empty
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from matplotlib import pyplot as plt

class KDEBandit:
    def __init__(self, n_arms, bandwidth=0.2):
        self.n_arms = n_arms
        self.bandwidth = bandwidth
        self.kde_models = [KernelDensity(bandwidth=self.bandwidth) for _ in range(n_arms)]
        self.data = {arm: [] for arm in range(n_arms)}
        self.rewards = {arm: [] for arm in range(n_arms)}
        self.reward_history = []
        self.optimal_reward_history = []

    def update(self, arm, context, reward):
        """Update the KDE model for the chosen arm."""
        self.data[arm].append(context)
        self.rewards[arm].append(reward)
        # Fit KDE model with current data
        data_array = np.array(self.data[arm])
        reward_array = np.array(self.rewards[arm]).reshape(-1, 1)
        combined_data = np.hstack((data_array, reward_array))
        self.kde_models[arm].fit(combined_data)

    def predict(self, context):
        """Predict rewards for all arms using KDE."""
        predictions = []
        for arm in range(self.n_arms):
            if len(self.data[arm]) > 0:
                kde = self.kde_models[arm]
                # Sample synthetic rewards for the given context
                synthetic_data = np.hstack((np.tile(context, (100, 1)), np.linspace(0, 1, 100).reshape(-1, 1)))
                log_probs = kde.score_samples(synthetic_data)
            else:
                log_probs = np.random.uniform(0, 1, 100)
            probs = np.exp(log_probs)
            mean_reward = np.dot(probs, np.linspace(0, 1, 100))  # Expected reward
            predictions.append(mean_reward)
        return predictions

    def select_arm(self, context):
        """Select the best arm based on predicted rewards."""
        predicted_rewards = self.predict(context)
        return np.argmax(predicted_rewards)

    def run(self, contexts, reward_generator):
        """Run the bandit algorithm with a list of pre-defined contexts."""
        for round_idx, context in enumerate(contexts):
            arm = self.select_arm(context)
            rewards = reward_generator(context)
            self.reward_history.append(rewards[arm])
            self.optimal_reward_history.append(np.max(rewards))
            self.update(arm, context, rewards[arm])




def linear_reward_generator(context, n_arm, true_weights, noise_std=0.1):
    # Beispiel: Linearer Zusammenhang zwischen Kontext und Reward
    reward = []
    for arm in range(n_arm):
        reward.append(np.dot(context, true_weights[arm]) + np.random.normal(0, noise_std))
    return np.clip(reward, 0,1)  # Begrenze Reward auf [0, 1]

def probabilistic_reward_generator(context, true_weights, num_arms):
    # Belohnung basierend auf einer Wahrscheinlichkeit
    reward = []
    for arm in range(num_arms):
        prob = sigmoid(np.dot(context, true_weights[arm]))
        reward.append(np.random.binomial(1, prob))
    return reward

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Initialisierung
n_arms = 3
n_features = 2
n_rounds = 1000
true_weights = np.random.rand(n_arms, n_features)  # Wahre Gewichte für jede Aktion
contexts = np.random.rand(n_rounds, n_features)  # 100 vordefinierte Kontexte

# Linearer Zusammenhang
bandit1 = KDEBandit(n_arms)
bandit1.run(contexts, lambda ctx: linear_reward_generator(ctx, n_arms, true_weights))
regret1 = np.cumsum(np.array(bandit1.optimal_reward_history) - np.array(bandit1.reward_history))
print('regret1', regret1)

# Wahrscheinlichkeitsbasiert
bandit2 = KDEBandit(n_arms)
bandit2.run(contexts, lambda ctx: probabilistic_reward_generator(ctx, true_weights, n_arms))
regret2 = np.cumsum(np.array(bandit2.optimal_reward_history) - np.array(bandit2.reward_history))
print('regret2', regret2)





plt.plot(regret1, label='linear model')
plt.plot(regret2, label='stochastic model')
plt.title("Cumulative regret")
plt.legend()
plt.show()