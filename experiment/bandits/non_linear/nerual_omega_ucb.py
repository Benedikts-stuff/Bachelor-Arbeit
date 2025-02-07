import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .models.neural import SingleArmNetwork as SingleArmNetwork


class NeuralOmegaUCB:
    def __init__(self, n_arms, context_dim, p):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.arm_counts = np.ones(self.n_arms, dtype=np.float64)
        self.gamma = 1e-8

        self.p = p
        self.z = 1

        # Initialize models and optimizers
        self.models = [SingleArmNetwork(self.context_dim).double() for _ in range(self.n_arms)]
        self.optimizer = [optim.Adam(model.parameters(), lr=0.01) for model in self.models]
        self.criterion = nn.MSELoss()

        # Models for cost
        self.models_c = [SingleArmNetwork(self.context_dim).double() for _ in range(self.n_arms)]
        self.optimizer_c = [optim.Adam(model.parameters(), lr=0.01) for model in self.models_c]
        self.criterion_c = nn.MSELoss()

        self.contexts_seen = [[] for _ in range(self.n_arms)]
        self.rewards_seen = [[] for _ in range(self.n_arms)]
        self.costs_seen = [[] for _ in range(self.n_arms)]

    def select_arm(self, context, t):
        if t < self.n_arms:
            self.contexts_seen[t].append(context)
            return t

        context_tensor = torch.tensor(context, dtype=torch.double).unsqueeze(0)
        context_tensor.requires_grad = True

        mu_r = np.array([self.models[t](context_tensor).detach().numpy().squeeze(0).item() for t in range(self.n_arms)])
        mu_c = np.array([self.models_c[t](context_tensor).detach().numpy().squeeze(0).item() for t in range(self.n_arms)])

        z = np.sqrt(2 * self.p * np.log(t))

        A = self.arm_counts + z**2

        B_r = 2 * self.arm_counts * mu_r + z**2
        B_c = 2 * self.arm_counts * mu_c + z**2

        C_r = self.arm_counts * mu_r**2
        C_c = self.arm_counts * mu_c**2

        x_r = np.sqrt(np.clip((B_r**2 / (4 * A**2)) - (C_r / A), 0, None))
        x_c = np.sqrt(np.clip((B_c**2 / (4 * A**2)) - (C_c / A), 0, None))

        omega_r = (B_r / (2 * A)) + x_r
        omega_c = (B_c / (2 * A)) - x_c

        arm = np.argmax(omega_r / (omega_c))

        self.contexts_seen[arm].append(context)
        return arm

    def update(self, actual_reward, actual_cost, chosen_arm, context):
        self.arm_counts[chosen_arm] += 1

        self.rewards_seen[chosen_arm].append(actual_reward)
        self.costs_seen[chosen_arm].append(actual_cost)

        if sum(len(v) for v in self.contexts_seen) < 3000:
            self.update_parameters_reward(chosen_arm, self.contexts_seen[chosen_arm], self.rewards_seen[chosen_arm])
            self.update_parameters_cost(chosen_arm, self.contexts_seen[chosen_arm], self.costs_seen[chosen_arm])

    def update_parameters_reward(self, action, contexts, rewards):
        scale_factor = 1
        scaled_rewards = [r * scale_factor for r in rewards]

        batch_size = min(len(contexts), 48)

        dataset = TensorDataset(torch.tensor(contexts, dtype=torch.double), torch.tensor(scaled_rewards, dtype=torch.double))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.models[action]
        optimizer = self.optimizer[action]

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        model.train()
        for batch, reward in dataloader:
            optimizer.zero_grad()
            predicted_reward = model(batch).squeeze()
            loss = self.criterion(predicted_reward, reward)
            loss.backward()
            optimizer.step()

        scheduler.step()

    def update_parameters_cost(self, action, contexts, costs):
        scale_factor = 1
        scaled_costs = [r * scale_factor for r in costs]

        batch_size = min(len(contexts), 48)

        dataset = TensorDataset(torch.tensor(contexts, dtype=torch.double), torch.tensor(scaled_costs, dtype=torch.double))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.models_c[action]
        optimizer = self.optimizer_c[action]

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        model.train()
        for batch, cost in dataloader:
            optimizer.zero_grad()
            predicted_cost = model(batch).squeeze()
            loss = self.criterion(predicted_cost, cost)
            loss.backward()
            optimizer.step()

        scheduler.step()
