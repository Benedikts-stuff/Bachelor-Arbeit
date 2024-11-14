import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


def initialize_neuralts_params(d, m, L, lambda_):
    # Initializing network parameters as described in the paper
    layers = [d] + [m] * (L - 2) + [1]
    model = NeuralNetwork(layers)

    U_0 = lambda_ * torch.eye(sum(param.numel() for param in model.parameters()))
    theta_0 = torch.cat([param.flatten() for param in model.parameters()])

    return [model, U_0, theta_0]


def sample_posterior_reward(model, x, theta, U_inv, nu, m, lamda):
    # Compute the gradient g(x; θ)
    x = x.clone().detach().requires_grad_(True)
    output = model(x)
    output.backward()
    grad = torch.cat([param.grad.flatten() for param in model.parameters()])

    # Calculate posterior variance and sample reward
    sigma_sq = lamda *(grad.T @ U_inv @ grad) / m
    mean_reward = output.item()
    sampled_reward = np.random.normal(mean_reward, max(nu * sigma_sq, 0.00000000001))

    return sampled_reward, grad.detach()


def update_posterior(U, grad, m):
    U += (grad.unsqueeze(1) @ grad.unsqueeze(0)) / m
    #U_inv = torch.linalg.inv(U)
    return U


def gradient_descent_step(model, x, reward, lambda_, theta_0, eta, J):
    # Define loss function for L2 regularized square loss
    optimizer = optim.SGD(model.parameters(), lr=eta)
    criterion = nn.MSELoss()

    # Gradient descent for J steps
    for _ in range(J):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, reward) + lambda_ * (
                    (torch.cat([p.flatten() for p in model.parameters()]) - theta_0) ** 2).sum()
        loss.backward()
        optimizer.step()

    # Update theta to new model parameters
    theta = torch.cat([param.flatten() for param in model.parameters()])
    return theta


def neural_thompson_sampling(T, d, m, L, lambda_, nu, eta, J, contexts, rewards, num_arms):
    # Initialize network and posterior parameters
    arms = [initialize_neuralts_params(d, m, L, lambda_) for i in range(num_arms)] # = [model, U_0, theta_0]
    observed_rewards = []
    optimal_rewards = []

    cumulative_reward = 0
    for t in tqdm(range(T)):
        #round_reward = 0
        sampled_rewards = []
        gradients = []
        for i in range(num_arms):
            # Sample posterior reward
            model = arms[i][0]
            U_inv = torch.linalg.inv(arms[i][1])
            theta = arms[i][2]
            sampled_reward, grad = sample_posterior_reward(model, contexts[t], theta, U_inv, nu, m, lambda_)
            sampled_rewards.append(sampled_reward)
            gradients.append(grad)

        chosen_action = np.argmax(sampled_rewards)
        # Pull arm at and observe actual reward
        actual_reward = torch.tensor(rewards[t][chosen_action]) # Simulated reward for chosen context
        observed_rewards.append(actual_reward.item())

        optimal_reward = np.max(rewards[t])
        optimal_rewards.append(optimal_reward)

        # Update posterior distribution
        #reward_diff = actual_reward - round_reward
        U = arms[chosen_action][1]
        grad = gradients[chosen_action]
        arms[chosen_action][1]= update_posterior(U, grad, m)

        # Gradient descent to update model parameters
        x_chosen = contexts[t]
        theta = gradient_descent_step(arms[chosen_action][0], x_chosen, actual_reward, lambda_, arms[chosen_action][2], eta, J)
        arms[chosen_action][2] = theta
        cumulative_reward += actual_reward

    return cumulative_reward, observed_rewards, optimal_rewards


# Hyperparameters (Example)
T = 5000  # Number of rounds
d = 3  # Context dimension
m =48  # Neural network width 20
L = 3  # Depth of network 3
lambda_ = 1  # Regularization parameter
nu = 0.4  # Exploration variance 1 0.2
eta = 0.1  # Learning rate for gradient descent 0.01
J = 10 # Number of gradient descent iterations 5
num_arms = 3
# True weights for each arm
torch.manual_seed(42)
np.random.seed(42)
true_weights = [torch.rand(d) for i in range(num_arms)]

# Simulated contexts
contexts = [torch.rand(d) for _ in range(T)]

# Generate rewards based on a sinusoidal function of the dot product
rewards = [[torch.dot(true_weights[j], contexts[i]) for j in range(num_arms)] for i in range(len(contexts))] # reward = exp(f(x)) wo bei f linear in context x

# Run Neural Thompson Sampling
reward = neural_thompson_sampling(T, d, m, L, lambda_, nu, eta, J, contexts, rewards, num_arms)
cumulative_reward = np.cumsum(reward[1])
opt_reward = np.cumsum(reward[2])
regret = opt_reward - cumulative_reward

print("Cumulative Reward:", cumulative_reward)
print("Optimal Cumulative Reward:", opt_reward)

plt.subplot(122)
#plt.plot(regret.cumsum(), label='linear model')
plt.plot(regret, label='Neural TS')
plt.title("Cumulative regret")
plt.legend()
plt.show()