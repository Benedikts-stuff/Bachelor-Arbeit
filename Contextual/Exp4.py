import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Lade die Daten
data_path = '../facebook-ad-campaign-data.csv'
ad_data = pd.read_csv(data_path)

# Gruppiere nach Anzeigen (campaign_id)
grouped_data = ad_data.groupby('campaign_id').agg({
    'clicks': 'sum',
    'total_conversion': 'sum',
    'impressions': 'sum'
}).reset_index()
grouped_data['CTR'] = grouped_data['clicks'] / grouped_data['impressions']

# Parameter für den EXP4-Algorithmus
n_arms = len(grouped_data['campaign_id'].unique())  # Anzahl der Aktionen (Kampagnen)
M = 7  # Anzahl der Experten
eta = 0.1  # Lernrate
gamma = 0.9  # Diskontierungsfaktor
Q = np.full((M,), 1 / M)  # Initialverteilung über Experten
n_round = 10000

# Initialisierung der Experten
experts = [np.random.dirichlet(np.ones(n_arms)) for _ in range(M)]
rewards = np.zeros(n_arms)
arm_count = np.zeros(n_arms)

for n in range(n_round):
    # Wähle einen Experten basierend auf Q
    expert_index = np.random.choice(range(M), p=Q)
    expert = experts[expert_index]

    # Wähle den Arm basierend auf der Empfehlung des gewählten Experten
    arm = np.random.choice(n_arms, p=expert)
    arm_count[arm] += 1

    # Simuliere die Belohnung (z.B. basierend auf CTR)
    reward = np.random.binomial(1, grouped_data['CTR'].iloc[arm])  # Belohnung (0 oder 1)

    # Schätzung der Belohnungen für alle Arme
    estimate_rewards = np.zeros(n_arms)

    for i in range(n_arms):
        if arm_count[i] > 0:
            # Schätze die Belohnung basierend auf der Anzahl der gezogenen Arme
            estimate_rewards[i] = (arm_count[i] / (arm_count[i] + gamma)) * (1 - grouped_data['CTR'].iloc[i])
        else:
            estimate_rewards[i] = 0  # Vermeide Division durch Null

    # Propagiere die Belohnung zu den Experten
    rewards = np.zeros(M)  # Belohnungen für Experten zurücksetzen
    for j in range(M):
        rewards[j] = np.dot(experts[j], estimate_rewards)

    # Update der Verteilung über die Experten Q_t
    Q_new = np.zeros_like(Q)
    for j in range(M):
        Q_new[j] = (np.exp(eta * rewards[j]) * Q[j]) / np.sum(np.exp(eta * rewards) * Q)

    Q = Q_new

plt.figure(figsize=(16, 10))
plt.bar(range(n_arms), arm_count)
plt.title("Häufigkeit der Arm-Auswahl")
plt.xlabel("Arm (Anzeige)")
plt.ylabel("Anzahl der Ziehungen")
plt.savefig('häufigkeit_arme_thompson_normal.pdf')
plt.show()
