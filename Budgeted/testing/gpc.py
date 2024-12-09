import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Daten generieren: 2 Klassen, 2 Merkmale
np.random.seed(42)
X = np.random.rand(100, 2) * 2 - 1  # Zufällige Werte im Bereich [-1, 1]
y = (X[:, 0]**2 + X[:, 1]**2 < 0.5).astype(int)  # Punkte innerhalb eines Kreises = Klasse 1

# Aufteilen in Training und Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Kernel: Kombination aus RBF und einem konstanten Multiplikator
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# Gaussian Process Classifier
gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)

# Training
gpc.fit(X_train, y_train)

# Vorhersagen und Wahrscheinlichkeiten
y_pred = gpc.predict(X_test)
y_prob = gpc.predict_proba(X_test)[:, 1]  # Wahrscheinlichkeit für Klasse 1

# Gitter für die Visualisierung
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

# Vorhersagewahrscheinlichkeiten für das Gitter
Z = gpc.predict_proba(grid)[:, 1]
Z = Z.reshape(xx.shape)

# Plotten
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.7)
plt.colorbar(label="Probability of Class 1")
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k", label="Train")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k", marker="x", label="Test")
plt.title("Gaussian Process Classifier")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
