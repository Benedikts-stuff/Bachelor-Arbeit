import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# LaTeX-Einstellungen
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'''
    \usepackage{libertine}
    \usepackage[libertine]{newtxmath}
    \usepackage[scaled=.92]{sourcesanspro}
'''

# Definition der Parameter und Titel für jeden Subplot:
# Links oben: a=10, b=10
# Rechts oben: a=3,  b=3
# Links unten: a=1,   b=1
# Rechts unten: a=0.5, b=0.5
params = [(10, 10), (2, 2), (0.9, 0.9), (0.5, 0.5)]
titles = [r'$\mathcal{B}(10, 10)$',
          r'$\mathcal{B}(2, 2)$',
          r'$\mathcal{B}(0.9, 0.9)$',
          r'$\mathcal{B}(0.5, 0.5)$']

# Erzeuge einen x-Bereich
x = np.linspace(0.01, 1 - 0.01, 1000)

# Erstelle ein 2x2 Grid für die Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Iteriere über die Parameter und zeichne jeden Subplot
for ax, (a, b), title in zip(axes.flatten(), params, titles):
    # Berechne die Dichtefunktion der Beta-Verteilung
    pdf = beta.pdf(x, a, b)

    # Zeichne die Linie und fülle den Bereich darunter
    ax.plot(x, pdf, color='blue', lw=2, label=f'Beta({a}, {b})')
    ax.fill_between(x, pdf, color='blue', alpha=0.3)

    # Setze Titel und Achsenbeschriftungen
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$f(x)$', fontsize=16)
    ax.set_xlim(0, 1)

    # Wähle eine sinnvolle y-Achsenbegrenzung (etwas oberhalb des Maximums)
    y_max = np.max(pdf) * 1.1
    ax.set_ylim(0, y_max)

    # Gitterlinien
    ax.grid(True, linestyle='--', alpha=0.6)

    # Berechne statistische Werte (Mittelwert und Standardabweichung)
    mean = a / (a + b)
    variance = (a * b) / (((a + b) ** 2) * (a + b + 1))
    std = np.sqrt(variance)
    info_text = fr'$\mu = {mean:.2f}$' + '\n' + fr'$\sigma = {std:.2f}$'

    # Füge den Text in den Plot ein
    ax.text(0.65, 0.8 * y_max, info_text, fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8))

# Optimiere das Layout und speichere den Plot
plt.tight_layout()
plt.savefig('four_beta_distributions.pdf', dpi=300)
plt.show()
