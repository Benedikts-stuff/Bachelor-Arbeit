from Adverserial.adversary import Friend
from adversary import Adversary
from ucb import UCB_Bandit
import matplotlib.pyplot as plt
from exp3 import EXP3
import numpy as np


np.random.seed(42)

n_arms = 3
n_rounds = 300
delta = [0.1, 0.2, 0.3, 0.4]
eta = [0.1, 0.2, 0.3, 0.4, 0.5]
probs = [0.2, 0.3, 0.5]
repetitions = 30

rewards1 = np.zeros(n_rounds)
rewards2 = np.zeros(n_rounds)
rewards3 = np.zeros(n_rounds)
rewards4 =np.zeros(n_rounds)

for i in range(repetitions):
    #in the adversary setting
    np.random.seed(i)
    adversary = Adversary(probs, seed=i)
    bandit_exp = EXP3(np.random.choice(eta), n_rounds, n_arms, adversary,i)
    bandit_exp.run()
    rewards1 = np.add(rewards1, bandit_exp.reward_history)

    bandit_ucb = UCB_Bandit(n_arms, np.random.choice(delta), n_rounds, adversary, i)
    bandit_ucb.run()
    rewards2 = np.add(rewards2, bandit_ucb.reward_history)

    #in the stochastic bandit setting
    reward_generator = Friend(probs, seed= i)

    bandit_exp_normal = EXP3(np.random.choice(eta), n_rounds, n_arms, reward_generator, i)
    bandit_exp_normal.run()
    rewards3 = np.add(rewards3, bandit_exp_normal.reward_history)

    bandit_ucb_normal = UCB_Bandit(n_arms, np.random.choice(delta), n_rounds, reward_generator, i)
    bandit_ucb_normal.run()
    rewards4 = np.add(rewards4, bandit_ucb_normal.reward_history)



optimal_reward = np.full(n_rounds, np.max(probs))
regret_ucb_adv= np.cumsum(optimal_reward) - np.cumsum(np.array(rewards1)/repetitions)
regret_exp3_adv =np.cumsum(optimal_reward) - np.cumsum(np.array(rewards2)/repetitions)
regret_ucb = np.cumsum(optimal_reward) - np.cumsum(np.array(rewards3)/repetitions)
regret_exp3 = np.cumsum(optimal_reward) - np.cumsum(np.array(rewards4)/repetitions)

plt.plot(regret_ucb_adv, label='UCB adversary setting')
plt.plot(regret_exp3, label='EXP3 classic setting')
plt.plot(regret_ucb, label='UCB classic setting')
plt.plot(regret_exp3_adv, label='EXP3 adversary setting')
plt.title("Cumulative regret")
plt.legend()
plt.savefig("adversarial_setting_example")
plt.show()