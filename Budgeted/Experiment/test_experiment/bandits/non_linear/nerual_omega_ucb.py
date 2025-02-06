import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .models.neural import SingleArmNetwork as SingleArmNetwork


class NeuralOmegaUCB:
    def __init__(self, n_arms, context_dim, p):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.arm_counts = np.zeros(self.n_arms)
        self.gamma = 1e-8

        self.z = 1
        self.p = p

        # Initialize variables
        self.models = [SingleArmNetwork(self.context_dim).double() for _ in range(self.n_arms)]
        self.optimizer = [optim.Adam(model.parameters(), lr=0.01) for model in self.models]
        self.criterion = nn.MSELoss()

        # models for cost
        self.models_c = [SingleArmNetwork(self.context_dim).double() for _ in range(self.n_arms)]
        self.optimizer_c = [optim.Adam(model.parameters(), lr=0.01) for model in self.models_c]
        self.criterion_c = nn.MSELoss()

        self.contexts_seen = [[] for _ in range(self.n_arms)]
        self.rewards_seen =[[] for _ in range(self.n_arms)]
        self.costs_seen = [[] for _ in range(self.n_arms)]


    def calculate_upper_confidence_bound(self, context, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        context_tensor = torch.DoubleTensor(context).unsqueeze(0)
        context_tensor.requires_grad = True
        output = [self.models[t](context_tensor).detach().numpy().squeeze(0) for t in range(self.n_arms)]

        upper =[]
        for i, reward in enumerate(output):
            mu_r = reward.item()
            #print(f"NeuralOmnegaUCB mu_r in round {round} and arm {i}", mu_r)
            eta = 1
            arm_count = self.arm_counts[i]
            z = np.sqrt(2* self.p* np.log(round + 2))
            if mu_r != 0 and mu_r != 1:
                eta = 1 #

           # print('LOOOL', mu_r )
            A = arm_count + z**2 * eta
            B = 2*arm_count*mu_r + z**2 * eta # eig noch * (M-m) aber das ist hier gleich 1
            C = arm_count* mu_r**2
            x = np.sqrt(np.clip((B**2 / (4* A**2)) - (C/A), 0, None))
            omega_r = np.clip((B/(2*A)) + x, 0, 1)
            upper.append(omega_r)

        return upper

    def calculate_lower_confidence_bound(self, context,round):
        context_tensor = torch.DoubleTensor(context).unsqueeze(0)
        context_tensor.requires_grad = True
        output = [self.models_c[t](context_tensor).detach().numpy().squeeze(0) for t in range(self.n_arms)]
        lower = []
        for i, cost in enumerate(output):
            mu_c = cost.item()

            arm_count = self.arm_counts[i]
            eta = 1
            z = np.sqrt(2 * self.p * np.log(round + 2))

            A = arm_count + z**2 * eta
            B = 2 * arm_count * mu_c + z**2 * eta  # eig noch * (M-m) aber das ist hier gleich 1
            C = arm_count * mu_c**2

            omega_c = B / (2 * A) - np.sqrt(np.clip(    (B ** 2 / (4 * A ** 2)) - C / A, 0, None)    )
            lower.append(np.clip(omega_c, self.gamma, 1))

        return lower

    def select_arm(self, context, round):
        """
        Select the arm with the highest upper confidence bound, adjusted for cost.
        """
        upper = np.array(self.calculate_upper_confidence_bound(context, round))
        lower = np.array(self.calculate_lower_confidence_bound(context, round))
        ratio = upper/lower
        arm = np.argmax(ratio)
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
