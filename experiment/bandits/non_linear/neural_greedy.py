import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .models.neural import SingleArmNetwork as SingleArmNetwork


class NeuralGreedy:
    def __init__(self, n_arms, context_dim, alpha):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.arm_counts = np.zeros(self.n_arms)
        self.gamma = 1e-8

        self.alpha = alpha

        # Initialize variables
        self.models = [SingleArmNetwork(self.context_dim) for _ in range(self.n_arms)]
        self.optimizer = [optim.Adam(model.parameters(), lr=0.01) for model in self.models]
        self.criterion = nn.MSELoss()

        # models for cost
        self.models_c = [SingleArmNetwork(self.context_dim) for _ in range(self.n_arms)]
        self.optimizer_c = [optim.Adam(model.parameters(), lr=0.01) for model in self.models_c]
        self.criterion_c = nn.MSELoss()

        self.contexts_seen = [[] for _ in range(self.n_arms)]
        self.rewards_seen =[[] for _ in range(self.n_arms)]
        self.costs_seen = [[] for _ in range(self.n_arms)]

    def select_arm(self, context, t):
        if t< self.n_arms:
            self.contexts_seen[t].append(context)
            return t

        epsilon = min(1, self.alpha * (self.n_arms / (t + 1)))
        arm: any
        if np.random.rand() < epsilon:
            arm = np.random.randint(self.n_arms)  # Explore
        else:
            context_tensor = torch.DoubleTensor(context).unsqueeze(0)
            context_tensor.requires_grad = True

            # Vorhersagen für Belohnungen und Kosten
            reward = np.array([self.models[t](context_tensor).detach().numpy().squeeze(0)  for t in range(self.n_arms)])
            cost = np.array([self.models_c[t](context_tensor).detach().numpy().squeeze(0)  for t in range(self.n_arms)])
            print("guess: ", reward)
            arm = np.argmax(reward / (cost+self.gamma))

        self.contexts_seen[arm].append(context)
        return arm

    def update(self, actual_reward, actual_cost, chosen_arm, context):
        self.arm_counts[chosen_arm] +=1

        self.rewards_seen[chosen_arm].append(actual_reward)
        self.costs_seen[chosen_arm].append(actual_cost)

        if sum(len(v) for v in self.contexts_seen) < 3000:
            self.update_parameters_reward(chosen_arm, self.contexts_seen[chosen_arm], self.rewards_seen[chosen_arm])

            self.update_parameters_cost(chosen_arm,  self.contexts_seen[chosen_arm], self.costs_seen[chosen_arm])

    def update_parameters_reward(self, action, contexts, rewards):
        # Skalierungsfaktor für die Rewards
        scale_factor = 1

        # Skalierte Rewards für das Training
        scaled_rewards = [r * scale_factor for r in rewards]

        # Dynamische Batch-Größe (maximal 256)
        batch_size = min(len(contexts), 48)

        # Erstelle ein Dataset und DataLoader
        dataset = TensorDataset(torch.DoubleTensor(contexts), torch.DoubleTensor(scaled_rewards))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Hole das Modell und den Optimizer für die gegebene Aktion
        model = self.models[action]
        model = model.double()
        optimizer = self.optimizer[action]

        # Learning Rate Scheduler (optional)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Setze das Modell in den Trainingsmodus
        model.train()

        # Training über den gesamten Datensatz
        for batch, reward in dataloader:
            optimizer.zero_grad()

            # Vorhersage des Modells
            predicted_reward = model(batch).squeeze()

            # Berechnung des Verlusts mit den **hochskalierten** Rewards
            loss = self.criterion(predicted_reward, reward)

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Update der Learning Rate (falls ein Scheduler verwendet wird)
        scheduler.step()

    def update_parameters_cost(self, action, contexts, costs):
        scale_factor = 1

        # Skalierte Rewards für das Training
        scaled_costs = [r * scale_factor for r in costs]

        # Dynamische Batch-Größe (z. B. bis zu 256, aber nicht mehr als die Anzahl der Contexts)
        batch_size = min(len(contexts), 48)  # Erhöhe die Batch-Größe für stabilere Gradienten

        # Erstelle ein Dataset und DataLoader
        dataset = TensorDataset(torch.DoubleTensor(contexts), torch.DoubleTensor(scaled_costs))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Hole das Modell und den Optimizer für die gegebene Aktion
        model = self.models_c[action]
        model = model.double()
        optimizer = self.optimizer_c[action]

        # Learning Rate Scheduler (optional)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Setze das Modell in den Trainingsmodus
        model.train()

        # Training über den gesamten Datensatz
        for batch, cost in dataloader:
            # Nullsetzen der Gradienten
            optimizer.zero_grad()

            # Vorhersage des Modells
            predicted_cost = model(batch).squeeze()

            # Berechnung des Verlusts
            loss = self.criterion(predicted_cost, cost)

            # Backpropagation
            loss.backward()

            # Update der Parameter
            optimizer.step()


        # Update der Learning Rate (falls ein Scheduler verwendet wird)
        scheduler.step()


