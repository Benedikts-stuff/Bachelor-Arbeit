import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Definiere die Funktion
def adaptive_epsilon(t):
    return np.minimum(1, 4 * 5 / (t + 1))  # Vermeide Division durch Null mit np.minimum

# Erstelle ein Gitter für x
x = np.linspace(0, 999, 999)  # Werte von 0 bis 999
Y = adaptive_epsilon(x)

# Erstelle ein DataFrame für seaborn
data = pd.DataFrame({
    'x': x,  # Beispielspalte
    'y': Y        # Beispielspalte
})

# Setze LaTeX-Schriftarten für Konsistenz (optional)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{libertine}\usepackage[libertine]{newtxmath}'

# Plot mit seaborn
sns.lineplot(x="x", y="y", data=data,
             color="blue", linestyle="-", linewidth=2)

# Achsentitel und Titel
plt.title(r"Adaptive $\epsilon$", fontsize=14)
plt.xlabel(r"$t$",  fontsize=14)
plt.ylabel(r"$\epsilon$", fontsize=14)

plt.savefig("adaptive_epsilon.pdf", dpi=300)
plt.show()