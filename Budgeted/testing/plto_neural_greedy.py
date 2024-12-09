from nerual_greedy import run_2
from nerual_greedy_1 import run_1
from  nerual_greedy_1 import generate_data
from nerual_greedy_1 import sigmoid_reward
import numpy as np
import matplotlib.pyplot as plt

num_iterations = 1
regret_1 = np.zeros(10000)
regret_2 = np.zeros(10000)
num_samples = 200000
context_dim = 5
num_actions = 3
contexts, true_rewards = generate_data(num_samples, context_dim, num_actions, sigmoid_reward)
for i in range(num_iterations):
    #x = run_1(i, contexts, true_rewards)
    #regret_1 = np.add(regret_1, x)
    y = run_2(i, contexts, true_rewards)
    regret_2 = np.add(regret_2, y)
    print(i)

plt.figure(figsize=(10, 6))
#plt.plot(regret_1 / num_iterations, label="Cumulative Regret multiple 1 nn", color="red")
plt.plot(regret_2 / num_iterations, label="Cumulative Regret multiple nns", color="blue")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Regret")
plt.title("Epsilon-Greedy Contextual Bandit with Separate Networks per Arm")
plt.legend()
plt.grid()
plt.show()


