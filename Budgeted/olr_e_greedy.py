import numpy as np
import matplotlib.pyplot as plt


class ContextualEpsilonGreedyBandit:
    def __init__(self, num_arms, num_features, epsilon, arm_cost):
        self.num_arms = num_arms
        self.num_features = num_features
        self.epsilon = epsilon
        self.weights = np.random.rand(num_arms, num_features)  # Initialisiere zufällige Gewichte
        self.arm_counts = np.zeros(num_arms)  # Zähler für jeden Arm
        self.total_rewards = np.zeros(num_arms)  # Kumulierte Belohnungen
        self.contexts_collected = []  # Gesammelte Kontexte
        self.rewards_collected = []  # Gesammelte Belohnungen
        self.arm_choices = []  # Gewählte Arme
        self.arm_cost = arm_cost

    def select_arm(self, context):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_arms)  # Exploration
        else:
            # Berechnung der erwarteten Belohnungen
            predicted_rewards = [np.dot(self.weights[arm], context) for arm in range(self.num_arms)]
            reward_cost_ratio = [predicted_rewards[arm] / self.arm_cost[arm] for arm in range(self.num_arms)]
            return np.argmax(reward_cost_ratio)  # Exploitation

    def update(self, arm, context, reward):
        # Speichere Kontext, Belohnung und Arm-Wahl
        self.contexts_collected.append(context)
        self.rewards_collected.append(reward)
        self.arm_choices.append(arm)

        # Update der Zähler und kumulierten Belohnungen
        self.arm_counts[arm] += 1
        self.total_rewards[arm] += reward

        # Fitting alle 200 Schritte
        if len(self.contexts_collected) >= 200:
            self.fit_model()

    def fit_model(self):
        # Trainiere das Modell auf den gesammelten Daten
        contexts_batch = np.array(self.contexts_collected)
        rewards_batch = np.array(self.rewards_collected)
        arms_batch = np.array(self.arm_choices)

        # Update der Gewichte für jeden Arm
        for arm in range(self.num_arms):
            indices = np.where(arms_batch == arm)[0]  # Indizes der gesammelten Daten für diesen Arm
            if len(indices) > 0:
                contexts_for_arm = contexts_batch[indices]  # Nur Kontexte für diesen Arm
                rewards_for_arm = rewards_batch[indices]  # Nur Belohnungen für diesen Arm
                # Berechne die geschätzten Belohnungen
                predictions = contexts_for_arm.dot(self.weights[arm])
                errors = rewards_for_arm - predictions
                # Update der Gewichte
                self.weights[arm] += (contexts_for_arm.T.dot(errors)) / len(indices)

        # Leere die gesammelten Daten für den nächsten Batch
        self.contexts_collected.clear()
        self.rewards_collected.clear()
        self.arm_choices.clear()


class Simulation:
    def __init__(self, num_arms, num_features, num_rounds, epsilon, budget):
        self.arm_cost = np.array([3, 6, 4])
        self.bandit = ContextualEpsilonGreedyBandit(num_arms, num_features, epsilon, self.arm_cost)
        self.budget = budget
        self.num_rounds = num_rounds
        self.contexts = np.random.rand(num_rounds, num_features)  # Zufällige Kontexte
        self.true_weights = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.2], [0.5, 0.2, 0.3]])  # Wahre Gewichte

        self.optimal_rewards = []  # Um die optimalen Belohnungen zu speichern
        self.cumulative_rewards = []  # Um die kumulierten Belohnungen zu speichern

    def calculate_reward(self, arm, context):
        return np.dot(self.true_weights[arm], context)  # Belohnung basierend auf den wahren Gewichten

    def run(self):
        t = 0
        while self.budget >= max(self.arm_cost):
            context = self.contexts[t]
            chosen_arm = self.bandit.select_arm(context)
            self.budget -= self.arm_cost[chosen_arm]
            reward = self.calculate_reward(chosen_arm, context)
            self.bandit.update(chosen_arm, context, reward)

            # Kumulierte Belohnungen verfolgen
            cumulative_reward = self.bandit.total_rewards.sum()
            self.cumulative_rewards.append(cumulative_reward)

            # Optimalen Reward verfolgen für Regret-Berechnung
            rewards =[self.calculate_reward(arm, context) for arm in range(self.bandit.num_arms)]
            optimal_idx = np.argmax([(rewards[arm] / self.arm_cost[arm]) for arm in range(self.bandit.num_arms)])
            self.optimal_rewards.append(rewards[optimal_idx])
            t +=1

    def plot_results(self):
        cumulative_rewards_array = np.array(self.cumulative_rewards)
        optimal_rewards_array = np.array(self.optimal_rewards)

        # Regret berechnen
        regret = np.cumsum(optimal_rewards_array) - cumulative_rewards_array
        optimal_cumulative = np.cumsum(optimal_rewards_array)
        plt.figure(figsize=(12, 6))

        # Plot der kumulierten Belohnung
        plt.subplot(2, 1, 1)
        plt.plot(optimal_cumulative, label="Optimal Kumulative Belohnung", color='red')
        plt.plot(cumulative_rewards_array, label="Kumulative Belohnung", color='blue')
        plt.xlabel("Runden")
        plt.ylabel("Kumulative Belohnung")
        plt.title("Kumulative Belohnung über Zeit")
        plt.legend()

        # Plot des Regret
        plt.subplot(2, 1, 2)
        plt.plot(regret, label="Regret", color='red')
        plt.xlabel("Runden")
        plt.ylabel("Regret")
        plt.title("Regret über Zeit")
        plt.legend()

        plt.tight_layout()
        plt.show()


# Simulation ausführen
np.random.seed(42)
simulation = Simulation(num_arms=3, num_features=3, num_rounds=10000, epsilon=0.1, budget=10000)
simulation.run()
simulation.plot_results()
