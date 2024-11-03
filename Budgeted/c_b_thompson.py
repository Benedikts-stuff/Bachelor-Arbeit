import numpy as np
import matplotlib.pyplot as plt


class ThompsonSamplingContextualBandit:
    def __init__(self, d, v):
        """
        d: Dimension der Kontextvektoren
        v: Varianzparameter für die Normalverteilung
        """
        self.d = d
        self.v = v
        self.B = np.identity(d)  # B wird als Identitätsmatrix der Dimension d gesetzt
        self.f = np.zeros(d)  # f ist ein Nullvektor der Dimension d
        self.mu_hat = np.zeros(d)  # μ^ wird als Nullvektor der Dimension d gesetzt

    def sample_mu_tilde(self):
        """Sample μ~(t) from a normal distribution with mean μ^ and covariance v^2 * B^-1."""
        cov_matrix = self.v ** 2 * np.linalg.inv(self.B)
        return np.random.multivariate_normal(self.mu_hat, cov_matrix)

    def choose_arm(self, contexts):
        """
        Wählt den besten Arm basierend auf den Kontextvektoren.
        contexts: Liste oder Matrix der Kontextvektoren für alle Arme in der aktuellen Runde.
        """
        mu_tilde = self.sample_mu_tilde()
        scores = [np.dot(context, mu_tilde) for context in contexts]
        return np.argmax(scores)

    def update(self, chosen_context, reward):
        """
        Aktualisiert die Parameter B, f und μ^ basierend auf dem gewählten Kontext und dem erhaltenen Reward.
        chosen_context: Kontextvektor des ausgewählten Arms in der aktuellen Runde.
        reward: Erhaltener Reward für den ausgewählten Arm.
        """
        # Update B und f
        self.B += np.outer(chosen_context, chosen_context)
        self.f += chosen_context * reward
        # Update von μ^
        self.mu_hat = np.linalg.inv(self.B).dot(self.f)


# Parameter
d = 3
v = 1.0
n_rounds = 10000

# Erstelle den ThompsonSamplingContextualBandit
ts_bandit = ThompsonSamplingContextualBandit(d, v)
np.random.seed(42)
# Beispiel-Kontexte für drei Arme
contexts = [
    np.array([1.0, 0.5, -0.2]),
    np.array([0.3, -0.1, 0.8]),
    np.array([0.5, 0.4, 0.3])
]

# Wahre Belohnungen (für Berechnung des Regrets, nehmen wir an, dass Arm 0 den höchsten Reward hat)
true_rewards = [1.0, 0.5, 0.2]

# Tracking cumulative reward und regret
cumulative_reward = 0
cumulative_regret = 0
cumulative_rewards = []
cumulative_regrets = []

# Simulation der Runden
for t in range(n_rounds):
    # Wähle den besten Arm für die aktuellen Kontexte
    chosen_arm = ts_bandit.choose_arm(contexts)
    # Simuliere einen Reward basierend auf dem gewählten Arm
    reward = np.random.normal(true_rewards[chosen_arm], 0.5)
    # Aktualisiere das Modell basierend auf dem gewählten Kontext und Reward
    ts_bandit.update(contexts[chosen_arm], reward)

    # Kumulativen Reward und Regret berechnen
    cumulative_reward += reward
    optimal_reward = true_rewards[0]  # Optimaler Reward (von Arm 0, da dies der beste ist)
    regret = optimal_reward - true_rewards[chosen_arm]
    cumulative_regret += regret

    # Speichern der Werte für den Plot
    cumulative_rewards.append(cumulative_reward)
    cumulative_regrets.append(cumulative_regret)

# Plotten der kumulativen Rewards und Regrets
plt.figure(figsize=(14, 6))

# Kumulativer Reward Plot
plt.subplot(1, 2, 1)
plt.plot(cumulative_rewards, label="Kumulativer Reward")
plt.xlabel("Runden")
plt.ylabel("Kumulativer Reward")
plt.title("Kumulativer Reward über die Runden")
plt.legend()

# Kumulativer Regret Plot
plt.subplot(1, 2, 2)
plt.plot(cumulative_regrets, label="Kumulativer Regret", color="red")
plt.xlabel("Runden")
plt.ylabel("Kumulativer Regret")
plt.title("Kumulativer Regret über die Runden")
plt.legend()

plt.tight_layout()
plt.show()
