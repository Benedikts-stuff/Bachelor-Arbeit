import numpy as np
import matplotlib.pyplot as plt

# Parameter
D = 5  # Dimension des ursprünglichen Kontextvektors
K = 3  # Dimension des reduzierten Featurespaces
T = 10000  # Anzahl der Zeitschritte
lambda_fine = 1.0  # Regularisierungsparameter für feine Ebene
lambda_coarse = 1.0  # Regularisierungsparameter für grobe Ebene

# Kontext-Matrix W und reduzierte Matrix U erstellen
N = 100  # Anzahl der möglichen Kontexte
W = np.random.uniform(low=0.0, high=1.0, size=(D, N))
U, _, _ = np.linalg.svd(W, full_matrices=False)
U_reduced = U[:, :K]   # projections matrix um x in neuen feature space zu Projezieren

contexts = np.random.uniform(-1, 1, size=(T, D))  # Kontextvektoren
true_theta = np.random.uniform(0, 1, size=(D,))  # Wahre Gewichtungen
rewards = (contexts @ true_theta) + np.random.normal(0, 0.1, size=T)

M_coarse = lambda_coarse * np.eye(K)
w_coarse = np.zeros(K)

M_fine = lambda_fine * np.eye(D)
w_fine = np.zeros(D)


# Hilfsfunktionen
def upper_confidence_bound(mean, cov, beta=1.0):
    """Berechne Upper Confidence Bound."""
    return mean + beta * np.sqrt(np.diagonal(cov))


chosen_contexts = []
observed_rewards = []
mean = []
# CoFineUCB-Algorithmus
for t in range(T):
    x_t = contexts[t]  # Wähle aktuellen Kontext
    X_t = np.array(chosen_contexts) if chosen_contexts else np.empty((0, D))
    Y_t = np.array(observed_rewards) if observed_rewards else np.empty((0,))

    if t > 0:
        # Grobe Ebene
        X_coarse = U_reduced.T @ X_t.T  # Grobe Kontexte
        M_coarse = lambda_coarse * np.eye(K) + X_coarse @ X_coarse.T
        w_coarse = np.linalg.solve(M_coarse, X_coarse @ Y_t)

        # Feine Ebene
        M_fine = lambda_fine * np.eye(D) + X_t.T @ X_t
        w_fine = np.linalg.solve(M_fine, X_t.T @ Y_t + lambda_fine * U_reduced @ w_coarse)

    # Erwartungswert berechnen
    mu_t = w_fine @ x_t
    mean.append(mu_t)

    # Auswahl mit Upper Confidence Bound
    confidence = upper_confidence_bound(mu_t, np.linalg.inv(M_fine))
    x_star = x_t  # Aktuelle Aktion (bei mehreren: max UCB wählen)

    # Belohnung beobachten
    y_t = rewards[t]

    # Update
    chosen_contexts.append(x_star)
    observed_rewards.append(y_t)


# Ergebnisse anzeigen
print("Gewichtungen (feine Ebene):", w_fine)
print("Gewichtungen (grobe Ebene):", w_coarse)
print("Anzahl gewählter Kontexte:", len(chosen_contexts))
#divergence
print('divergence', rewards - mean)

plt.plot(rewards - mean, label='divergence')
plt.title("Cumulative regret")
plt.legend()
plt.show()