import numpy as np
import matplotlib.pyplot as plt

# Generiere 2D-Kontext-Features (Gitter im Bereich [0, 1] x [0, 1])
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
contexts = np.c_[X.ravel(), Y.ravel()]

# Wahre Belohnungsfunktion (z. B. eine nichtlineare Funktion)
def true_reward_function(context):
    return 0.3 * context[0] + 0.5 * context[1]**5 #+ 0.2 * np.sin(5 * context[0])

# Berechne Wahrscheinlichkeiten p für jeden Kontext
p_values = np.array([true_reward_function(context) for context in contexts])
p_values = np.clip(p_values, 0, 1)  # Werte auf [0, 1] beschränken

# Binomialverteilung: Rewards basierend auf den Wahrscheinlichkeiten p
rewards = np.random.binomial(1, p_values)

# Punkte mit Klasse 0 und Klasse 1 extrahieren
class_1 = contexts[rewards == 1]
class_0 = contexts[rewards == 0]

# Heatmap der Wahrscheinlichkeiten p und Scatterplot der Klassen
plt.figure(figsize=(12, 6))

# Heatmap
plt.subplot(1, 2, 1)
plt.contourf(X, Y, p_values.reshape(X.shape), levels=50, cmap='viridis')
plt.colorbar(label='P(reward=1)')
plt.title('Heatmap der Wahrscheinlichkeiten')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Scatterplot
plt.subplot(1, 2, 2)
plt.scatter(class_0[:, 0], class_0[:, 1], c='red', label='Klasse 0', alpha=0.6)
plt.scatter(class_1[:, 0], class_1[:, 1], c='blue', label='Klasse 1', alpha=0.6)
plt.title('Scatterplot der Klassen')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()
