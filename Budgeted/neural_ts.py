import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=3):
        super(NeuralNet, self).__init__()
        self.hidden_dim = hidden_dim

        # Erste Schicht
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Versteckte Schichten
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)
        ])

        # Letzte Schicht
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Gewichte initialisieren
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialisiere erste und versteckte Schichten mit N(0, sqrt(4 / m))
        for layer in [self.input_layer, *self.hidden_layers]:
            nn.init.normal_(layer.weight, mean=0, std=(4 / self.hidden_dim) ** 0.5)
            nn.init.constant_(layer.bias, 0)  # Setze Bias auf 0 (optional)

        # Initialisiere die letzte Schicht mit N(0, sqrt(2 / m))
        nn.init.normal_(self.output_layer.weight, mean=0, std=(2 / self.hidden_dim) ** 0.5)
        nn.init.constant_(self.output_layer.bias, 0)  # Setze Bias auf 0 (optional)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))

        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        x = self.output_layer(x)
        return x

class NeuralThompsonSampling:
    def __init__(self, lamda, n_arms, n_features, m, L):
        self.n_arms = n_arms
        self.n_features = n_features
        self.lamda = lamda
        self.m = m # width of the NN
        self.L = L #depth of the NN
        self.U = np.array([lamda * np.identity(n_features) for _ in range(n_arms)])

        self.W = [np.random.normal(0, np.sqrt(4 / m), (m, m)) if l < L - 1 else np.random.normal(0, np.sqrt(2 / m), (m, 1))
             for l in range(L)]
        self.theta = np.concatenate([W_l.flatten() for W_l in self.W])  # theta_0 in vektorisierter Form

