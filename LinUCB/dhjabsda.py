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
            print(a, ':', theta_a)
        return np.argmax(p)

    def update(self, chosen_arm, x, reward):
        self.A[chosen_arm] += np.outer(x, x)
        self.b[chosen_arm] += reward * x


df = pd.read_csv('./data_linear.csv')
grouped = df.groupby(['age','gender' ,'interest' ])['ctr'].mean().reset_index()
grouped2= df.groupby(['age', 'gender', 'interest' ])
context_counts = grouped2.size().reset_index(name='group_size')
context_probs = context_counts['group_size'] / len(df)

grouped_data = df.groupby('campaign_id')
ctr2 = grouped_data['ctr'].mean().to_numpy()
max_ctr = np.max(ctr2)

probs = context_probs.to_numpy()

features = grouped[[ 'age', 'gender', 'interest' ]].to_numpy()

print(features)



d = features.shape[1]
K = df['campaign_id'].nunique()  # Anzahl der eindeutigen Kampagnen

# LinUCB-Algorithmus initialisieren


# Simulation über mehrere zufällige Schritte (T)
T = 30000  # Anzahl der Runden, in denen der Algorithmus laufen soll
optimal_reward = np.zeros((5, 30000))
reward_history = np.zeros((5, 30000))
arm_count = np.zeros(2)
for i in tqdm(range(5), desc='Processing'):
    linucb = LinUCB(0.1, d, K)
    np.random.seed(i)
    for t in range(T):
        # Zufälligen Kontext (x_t) basierend auf der Häufigkeit samplen
        idx = np.random.choice(len(grouped2), p=probs)  # Sample basierend auf den Wahrscheinlichkeiten
        x_t = features[idx]  # Kontextvektor
        chosen_arm = linucb.select_arm(x_t)

        # Belohnung ist 1, wenn clicks > 0, sonst 0
        reward_t = np.random.binomial(1, ctr2[chosen_arm])
        opt_reward = np.random.binomial(1, max_ctr)
        optimal_reward[i][t] = opt_reward
        arm_count[chosen_arm] += 1
        reward_history[i][t] = reward_t

        # Update des LinUCB Algorithmus
        linucb.update(chosen_arm, x_t, reward_t)

# Kumulierte Belohnung berechnen
reward_all = np.cumsum(reward_history, axis=1)
opti_all = np.cumsum(optimal_reward, axis=1)
cumu_opt_rew = np.mean(opti_all, axis=0)
cumulative_reward = np.mean(reward_all, axis=0)
np.save('cumulative_reward2', cumulative_reward)

plt.figure(figsize=(16, 10))
plt.plot(cumu_opt_rew, label = 'optimal reward')
plt.plot(cumulative_reward, label = 'cumulative')
plt.title("asdhasdbvas")
plt.xlabel("ashdbgasdas")
plt.ylabel("asjhdgajsbdga")
plt.legend()  # Legende anzeigen
plt.show()

regret = cumu_opt_rew - cumulative_reward
plt.figure(figsize=(16, 10))
plt.plot(regret , label = 'regret')
plt.title("asdhasdbvas")
plt.xlabel("ashdbgasdas")
plt.ylabel("asjhdgajsbdga")
plt.legend()  # Legende anzeigen
plt.show()