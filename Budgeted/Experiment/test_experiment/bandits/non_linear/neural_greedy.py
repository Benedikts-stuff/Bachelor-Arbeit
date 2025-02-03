import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .models.neural import SingleArmNetwork


class NeuralGreedy:
    def __init__(self, n_arms, context_dim, alpha):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.arm_counts = np.zeros(self.n_arms)
        self.gamma = 1e-8

        self.alpha = alpha

        # Initialize variables
        self.models = [SingleArmNetwork(self.context_dim) for _ in range(self.n_arms)]
        self.optimizer = [optim.Adam(model.parameters(), lr=0.001) for model in self.models]
        self.criterion = nn.MSELoss()

        # models for cost
        self.models_c = [SingleArmNetwork(self.context_dim) for _ in range(self.n_arms)]
        self.optimizer_c = [optim.Adam(model.parameters(), lr=0.001) for model in self.models_c]
        self.criterion_c = nn.MSELoss()

        self.contexts_seen = [[] for _ in range(self.n_arms)]
        self.rewards_seen =[[] for _ in range(self.n_arms)]
        self.costs_seen = [[] for _ in range(self.n_arms)]

    def select_arm(self, context, t):
        epsilon = min(1, self.alpha * (self.n_arms / (t + 1)))
        arm: any
        if np.random.rand() < epsilon:
            arm = np.random.randint(self.n_arms)  # Explore
        else:
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
            context_tensor.requires_grad = True

            # Vorhersagen für Belohnungen und Kosten
            reward = np.array([np.clip(self.models[t](context_tensor).detach().numpy().squeeze(0), 0, 1) for t in range(self.n_arms)])
            cost = np.array([np.clip(self.models_c[t](context_tensor).detach().numpy().squeeze(0), self.gamma, 1) for t in range(self.n_arms)])

            # Arm mit dem besten Verhältnis von Belohnung zu Kosten auswählen
            arm = np.argmax(reward / cost)

        self.contexts_seen[arm].append(context)
        return arm

    def update(self, actual_reward, actual_cost, chosen_arm, context):
        self.arm_counts[chosen_arm] +=1

        self.rewards_seen[chosen_arm].append(actual_reward)
        self.costs_seen[chosen_arm].append(actual_cost)

        if sum(len(v) for v in self.contexts_seen) < 1000:
            self.update_parameters_reward(chosen_arm, self.contexts_seen[chosen_arm], self.rewards_seen[chosen_arm])

            self.update_parameters_cost(chosen_arm,  self.contexts_seen[chosen_arm], self.costs_seen[chosen_arm])

    def update_parameters_reward(self, action, contexts, rewards):
        batch_size = len(contexts)
        if len(contexts) >= 20:
            batch_size = 20

        dataset = TensorDataset(torch.FloatTensor(contexts),
                                torch.FloatTensor(rewards))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.models[action]
        optimizer = self.optimizer[action]

        model.train()
        for batch, reward in dataloader:
            optimizer.zero_grad()
            predicted_reward = model(batch).squeeze()
            loss = self.criterion(predicted_reward, reward)
            loss.backward()
            optimizer.step()

    def update_parameters_cost(self, action, contexts, cost):
        batch_size = len(contexts)
        if len(contexts) >= 20:
            batch_size = 20

        dataset = TensorDataset(torch.FloatTensor(contexts),
                                torch.FloatTensor(cost))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.models_c[action]
        optimizer = self.optimizer_c[action]

        model.train()
        for batch, cost in dataloader:
            optimizer.zero_grad()
            predicted_reward = model(batch).squeeze()
            loss = self.criterion_c(predicted_reward, cost)
            loss.backward()
            optimizer.step()


