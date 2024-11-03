import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# LinUCB Algorithmus
class LinUCB:
    def __init__(self, alpha, d, K):
        self.alpha = alpha
        self.d = d  # Anzahl der Features
        self.K = K  # Anzahl der Arme (Anzeigen)
        self.A = [np.identity(self.d) for _ in range(self.K)]  # A als Identitätsmatrix initialisieren
        self.b = [np.zeros(self.d) for _ in range(self.K)]  # b als Nullvektor initialisieren

    def select_arm(self, x):
        p = []
        for a in range(self.K):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            p_t_a = theta_a.T @ x + self.alpha * np.sqrt(x.T @ A_inv @ x)
            p.append(p_t_a)
        return np.argmax(p)

    def update(self, chosen_arm, x, reward):
        self.A[chosen_arm] += np.outer(x, x)
        self.b[chosen_arm] += reward * x


# Funktion zur Vorverarbeitung der Daten (Kategorien in numerische Werte umwandeln)
def preprocess_features(df):
    # Geschlecht kodieren: M -> [1, 0], F -> [0, 1]
    df1 = pd.get_dummies(df, columns=['gender'])  # drop_first=True nimmt nur die Männlich-Spalte
    df1.rename(columns={'gender_1': 'gender_M', 'gender_0': 'gender_F'}, inplace=True)  # Umbenennen

    # Altersgruppen encodieren: Altersgruppen in numerische Werte umwandeln (z.B. '30-34' -> 0, '35-39' -> 1, etc.)
    age_map = {age: idx for idx, age in enumerate(df1['age'].unique())}
    df1['age'] = df1['age'].map(age_map)

    # Features: interest1, interest2, interest3, gender_M, gender_F, age
    grouped = df1.groupby(['gender_M', 'gender_F', 'age']).size().reset_index(
        name='context_count')
    features = grouped[['gender_M', 'gender_F', 'age']]

    # Features skalieren
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features


# Daten laden
df = pd.read_csv('../facebook-ad-campaign-data.csv')

# Features und Rewards vorbereiten
features = preprocess_features(df)

# Häufigkeit jedes Kontextes berechnen
grouped2 = df.groupby(['gender', 'age']).agg({
    'clicks': 'sum',
    'impressions': 'sum'
}).reset_index()
grouped = df.groupby(['gender', 'age'])
grouped2['ctr'] = grouped2['clicks'] / grouped2['impressions'].replace(0, 1)
ctr = grouped2['ctr'].to_numpy()
context_counts = grouped.size().reset_index(name='group_size')
context_probs = context_counts['group_size'] / len(df)

grouped_data = df.groupby('campaign_id').agg({
    'clicks': 'sum',
    'impressions': 'sum'
}).reset_index()
grouped_data['CTR'] = grouped_data['clicks'] / grouped_data['impressions']
ctr2 = grouped_data['CTR'].to_numpy()


probs = context_probs.to_numpy()



d = features.shape[1]
K = df['campaign_id'].nunique()



T = 500000
reward_history = np.zeros((10, 500000))
arm_count = np.zeros(3)
for i in tqdm(range(10), desc='Processing'):
    linucb = LinUCB(0.0000001, d, K)
    np.random.seed(i)
    for t in range(T):

        idx = np.random.choice(len(grouped2), p=probs)
        x_t = features[idx]
        chosen_arm = linucb.select_arm(x_t)


        reward_t = np.random.binomial(1, ctr2[chosen_arm])
        arm_count[chosen_arm] += 1
        reward_history[i][t] = reward_t


        linucb.update(chosen_arm, x_t, reward_t)


reward_all = np.cumsum(reward_history, axis=1)
cumulative_reward = np.mean(reward_all, axis=0)
np.save('cumulative_reward2', cumulative_reward)

plt.figure(figsize=(16, 10))
plt.bar(range(3), arm_count)
plt.title("Häufigkeit der Arm-Auswahl")
plt.xlabel("Arm (Anzeige)")
plt.ylabel("Anzahl der Ziehungen")
plt.savefig('häufigkeit_arme_thompson_normal.pdf')
plt.show()