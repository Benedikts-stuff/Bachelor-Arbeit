import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definiere das Polynom in 2D
def polynomial_2D(x, y):
    return -6 * x**2 * y**2 + 3 * x**2 * y + 3 * x * y**2 - 2 * x**3 - 2 * y**3

# Erstelle ein Gitter für x und y
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = polynomial_2D(X, Y)

# 3D-Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Achsentitel
ax.set_title("2D Polynom schwer zu approximieren")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
