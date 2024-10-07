import matplotlib.pyplot as plt
import numpy as np

# Lade die kumulierten Regret-Daten
cumulative_regret_bandit_1 = np.loadtxt('/Users/benedikt/Documents/Bachelor-Arbeit/Thompson/cumulative_regret_thompson')
cumulative_regret_bandit_2 = np.loadtxt('/Users/benedikt/Documents/Bachelor-Arbeit/Thompson/cumulative_regret_two_thompson')

# Anzahl der Runden
n_rounds = len(cumulative_regret_bandit_1)

# Plotten
plt.figure(figsize=(10, 6))
plt.plot(range(n_rounds), cumulative_regret_bandit_1, label="thompson", color='blue')
plt.plot(range(n_rounds), cumulative_regret_bandit_2, label="two thompson", color='green')
plt.xlabel("Runde")
plt.ylabel("Kumulativer Regret")
plt.title("Kumulativer Regret von Bandit 1 vs Bandit 2")
plt.legend()
plt.grid(True)
plt.savefig('thompson_vs_two.pdf')
plt.show()
