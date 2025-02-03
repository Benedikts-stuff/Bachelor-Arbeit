import numpy as np
import pandas as pd


class ThompsonSampling:
    """Class to implement a Thompson sampling policy
    for a multi-armed bandit problem."""

    def __init__(self, num_arms, top_arms=1):
        self.top_arms = top_arms
        self.arm_count = np.zeros(num_arms)
        # Beta parameters
        self.a = np.ones(num_arms)  # alpha parameter (successes + 1)
        self.b = np.ones(num_arms)  # beta parameter (failures + 1)
        self.num_arms = num_arms

    def select_arm(self, random_seed):
        """Selects an arm for each round."""
        np.random.seed(seed=random_seed)

        # Draw samples from Beta distribution
        samples = np.random.beta(self.a, self.b)

        # Return index of arm(s) with the highest sample
        n=  np.random.choice(np.argsort(samples)[-self.top_arms:])
        self.arm_count[n] += 1
        return n

    def update(self, chosen_arm, reward):
        """Updates the parameters of the chosen arm."""
        self.a[chosen_arm] += reward
        self.b[chosen_arm] += (1 - reward)


# Beispiel für die Verwendung der Klasse

# Lese die Werbedaten ein
data_path = '../Budgeted/Experiment/test_experiment/data/facebook-ad-campaign-data.csv'
ad_data = pd.read_csv(data_path)

# Aggregiere die Daten nach Kampagne
grouped_data = ad_data.groupby('campaign_id').agg({
    'clicks': 'sum',
    'spent': 'sum',
    'impressions': 'sum'
}).reset_index()

n_arms = len(grouped_data['campaign_id'].unique())
n_rounds = 10000
ts = ThompsonSampling(n_arms)

# Wahrscheinlichkeiten der Conversion basierend auf den bisherigen Daten
conversion_rates = grouped_data['clicks'].values / grouped_data['impressions'].values
conversion_rates[np.isnan(conversion_rates)] = 0  # Setze NaN auf 0
optimal_reward = np.max(conversion_rates)
cumulative_regret = np.zeros(n_rounds)
rewards = []

for n in range(n_rounds):
    chosen_arm = ts.select_arm(random_seed=n)

    # Simuliere die Umgebung (hier wird 1 als Klick und 0 als kein Klick verwendet)
    reward = np.random.binomial(1, conversion_rates[chosen_arm])

    ts.update(chosen_arm, reward)
    rewards.append(reward)
    cumulative_regret[n] = (optimal_reward - conversion_rates[chosen_arm]) + (cumulative_regret[n - 1] if n > 0 else 0)

np.savetxt('cumulative_regret_two_thompson', cumulative_regret)

# Ergebnisse visualisieren
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 10))
plt.bar(range(n_arms), ts.arm_count)
plt.title("Häufigkeit der Arm-Auswahl")
plt.xlabel("Arm (Anzeige)")
plt.ylabel("Anzahl der Ziehungen")
plt.savefig('häufigkeit_arme_thompson.pdf')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(cumulative_regret, label='Kumulativer regret')
plt.title('Kumulative Belohnung über Runden')
plt.xlabel('Runden')
plt.ylabel('Kumulative Belohnung')
plt.legend()
plt.show()
