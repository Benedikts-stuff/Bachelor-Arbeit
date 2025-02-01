import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, uniform

# LaTeX-Einstellungen
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'''
    \usepackage{libertine}
    \usepackage[libertine]{newtxmath}
    \usepackage[scaled=.92]{sourcesanspro}
'''

# Parameter für die Beta-Verteilung
a = 0.5  # alpha
b = 0.5  # beta

# Erzeuge einen Bereich von Werten für die x-Achse
x = np.linspace(0, 1, 1000)

# Berechne die Dichtefunktionen
beta_pdf = beta.pdf(x, a, b)  # Beta-Verteilung
uniform_pdf = uniform.pdf(x, 0, 1)  # Uniformverteilung im Intervall [0,1]

# Erstelle die Plots
plt.figure(figsize=(14, 6))

# Plot für die Uniformverteilung
plt.subplot(1, 2, 1)
plt.plot(x, uniform_pdf, color='green', label='Uniform [0,1]')
plt.fill_between(x, uniform_pdf, color='green', alpha=0.3)
plt.title(r'$\mathcal{U}([0,1))$', fontsize=28)
plt.xlabel(r'$x$', fontsize=24)  # LaTeX für die x-Achsenbeschriftung
plt.ylabel(r'$f(x)$', fontsize=24)  # LaTeX für die y-Achsenbeschriftung
plt.xlim(0, 1)
plt.ylim(0, 2.5)
plt.grid(True, linestyle='--', alpha=0.6)

# Füge statistische Informationen hinzu
mean_uniform = 0.5
variance_uniform = 1 / 12
std_dev_uniform = np.sqrt(variance_uniform)
info_text_uniform = fr'$\mu = {mean_uniform:.2f}$' + '\n' + fr'$\sigma = {std_dev_uniform:.4f}$' + '\n' + fr'$\sigma^2 = {variance_uniform:.4f}$'
plt.text(0.6, 2.0, info_text_uniform, fontsize=18, bbox=dict(facecolor='white', alpha=0.8))

# Plot für die Beta-Verteilung
plt.subplot(1, 2, 2)
plt.plot(x, beta_pdf, color='blue', label=f'Beta({a}, {b})')
plt.fill_between(x, beta_pdf, color='blue', alpha=0.3)
plt.title(r'$\mathcal{B}(0.5, 0.5)$', fontsize=28)  # LaTeX für den Titel
plt.xlabel(r'$x$', fontsize=24)  # LaTeX für die x-Achsenbeschriftung
plt.ylabel(r'$f(x)$', fontsize=24)  # LaTeX für die y-Achsenbeschriftung
plt.xlim(0, 1)
plt.ylim(0, 2.5)
plt.grid(True, linestyle='--', alpha=0.6)

# Füge statistische Informationen hinzu
mean_beta = a / (a + b)
variance_beta = (a * b) / ((a + b + 1) * (a + b)**2)
std_dev_beta = np.sqrt(variance_beta)
info_text_beta = fr'$\mu = {mean_beta:.2f}$' + '\n' + fr'$\sigma = {std_dev_beta:.4f}$' + '\n' + fr'$\sigma^2 = {variance_beta:.4f}$'
plt.text(0.6, 2.0, info_text_beta, fontsize=18, bbox=dict(facecolor='white', alpha=0.8))



# Zeige die Plots
plt.tight_layout()
plt.savefig('beta_vs_uniform.pdf', dpi=300)
plt.show()