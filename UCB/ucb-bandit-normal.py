import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# UCB-Bandit-Algorithmus-Klasse
class UCB_Bandit:
    def __init__(self, n_arms, delta):
        self.n_arms = n_arms  # Anzahl der Arme
        self.delta = delta  # Parameter für den Konfidenzbonus
        self.arm_counts = np.zeros(n_arms)  # Zählungen für jeden Arm
        self.arm_rewards = np.zeros(n_arms)  # Durchschnittliche Belohnung für jeden Arm
        self.total_pulls = 0

    def select_arm(self):
        total_pulls = np.sum(self.arm_counts)  # Gesamtzahl der Ziehungen
        if total_pulls == 0:
            return np.random.randint(0, self.n_arms)  # Zufälliger Arm, wenn noch kein Arm gezogen wurde

        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.arm_counts[i] == 0:
                return i  # Wähle einen Arm, der noch nie gezogen wurde
            # Berechne UCB für den Arm
            bonus = np.sqrt(2 * np.log(1 / self.delta) / self.arm_counts[i])
            ucb_values[i] = self.arm_rewards[i] + bonus

        if ucb_bandit.total_pulls in ucb_values_history:
            ucb_values_history[ucb_bandit.total_pulls].append(ucb_values)

        # Wähle den Arm mit dem höchsten UCB-Wert
        self.total_pulls += 1
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.arm_counts[arm] += 1  # Aktualisiere die Anzahl der Ziehungen für den Arm
        n = self.arm_counts[arm]
        # Aktualisiere den gleitenden Mittelwert der Belohnungen
        self.arm_rewards[arm] = ((n - 1) * self.arm_rewards[arm] + reward) / n

    def plot_results(self, cumulative_reward, ucb_bandit, n_arms):
        # Kumulative Belohnung plotten
        plt.figure(figsize=(15, 10))

        # Subplot 1: Kumulative Belohnung
        plt.subplot(3, 2, 1)
        plt.plot(cumulative_reward)
        plt.title("Kumulative Belohnung über die Zeit")
        plt.xlabel("Runden")
        plt.ylabel("Kumulative Belohnung")

        # Subplot 2: Häufigkeit der Arm-Auswahl
        plt.subplot(3, 2, 2)
        plt.bar(range(n_arms), ucb_bandit.arm_counts)
        plt.title("Häufigkeit der Arm-Auswahl")
        plt.xlabel("Arm (Anzeige)")
        plt.ylabel("Anzahl der Ziehungen")

        # Subplot 3: Durchschnittliche Belohnung pro Arm
        plt.subplot(3, 2, 3)
        plt.bar(range(n_arms), ucb_bandit.arm_rewards)
        plt.title("Durchschnittliche Belohnung pro Arm")
        plt.xlabel("Arm (Anzeige)")
        plt.ylabel("Durchschnittliche Belohnung")

        # Optional: Kumulativer Regret plotten
        optimal_reward = max(grouped_data['click_reward']) * np.arange(1, len(cumulative_reward) + 1)
        regret = optimal_reward - cumulative_reward
        plt.subplot(3, 2, 4)
        plt.plot(regret)
        plt.title("Kumulative Regret über die Zeit")
        plt.xlabel("Runden")
        plt.ylabel("Regret")

        # Layout anpassen und alle Plots anzeigen
        plt.tight_layout()
        plt.show()

# Lade den Datensatz
data_path = '../facebook-ad-campaign-data.csv'
ad_data = pd.read_csv(data_path)

# Erstelle binäre Belohnungen: 1 für Klicks/Conversions, 0 sonst
ad_data['click_reward'] = ad_data['clicks'].apply(lambda x: x if x > 0 else 0)
ad_data['conversion_reward'] = ad_data['total_conversion'].apply(lambda x: x if x > 0 else 0)

# Gruppiere nach Anzeigen (ad_id)
grouped_data = ad_data.groupby('ad_id').agg({
    'click_reward': 'mean',
    'conversion_reward': 'mean',
    'clicks': 'sum',
    'total_conversion': 'sum'
}).reset_index()

# Beispiel-Parameter für UCB
n_arms = len(ad_data['campaign_id'].unique())  # Anzahl der einzigartigen Anzeigen
delta = 0.05

# Initialisiere den UCB-Bandit
ucb_bandit = UCB_Bandit(n_arms, delta)

# Simuliere die Anwendung des Algorithmus (Beispiel für 1000 Runden)
n_rounds = 2000
rewards_history = []
ucb_values_history = {step: [] for step in [1, 500, 1000, 2000]}

for t in range(n_rounds):
    arm = ucb_bandit.select_arm()  # Wähle einen Arm (Anzeige)
    reward = grouped_data['click_reward'].iloc[arm]  # Nutze Klick-Belohnung
    ucb_bandit.update(arm, reward)  # Aktualisiere den Algorithmus mit der Belohnung

    # Speichere die Belohnung
    rewards_history.append(reward)

# Zeige das Ergebnis nach 1000 Runden an
cumulative_reward = np.cumsum(rewards_history)
print(f'Kumulative Belohnung: {cumulative_reward[-1]}')

#ucb_bandit.plot_results(cumulative_reward, ucb_bandit, n_arms)

print(ucb_bandit.arm_counts)

# Plotten der UCB-Werte
plt.figure(figsize=(10, 6))

for step in ucb_values_history:
    ucb_values = np.array(ucb_values_history[step])
    plt.plot(n_arms, ucb_values[0], label=f'UCB nach {step} Schritten')

plt.title('Upper Confidence Bounds nach verschiedenen Schritten')
plt.xlabel('Arm (ad_campaign)')
plt.ylabel('UCB-Werte')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()
