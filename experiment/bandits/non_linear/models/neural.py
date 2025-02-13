import torch.nn as nn
import torch

class SingleArmNetwork(nn.Module):
    def __init__(self, context_dim):
        super(SingleArmNetwork, self).__init__()
        self.fc1 = nn.Linear(context_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.out(x)


class SingleArmNetwork2(nn.Module):
    def __init__(self, context_dim):
        super(SingleArmNetwork2, self).__init__()
        self.fc1 = nn.Linear(context_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)  # Single reward prediction

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)