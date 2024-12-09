import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import scipy.optimize as opt
import pandas as pd


class MyGPR(GaussianProcessRegressor):
    def __init__(self, kernel, alpha, normalize_y, n_restarts_optimizer,  _max_iter=100):
        super().__init__(kernel, alpha=alpha, normalize_y= normalize_y, n_restarts_optimizer= n_restarts_optimizer)
        self._max_iter = _max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        def new_optimizer(obj_func, initial_theta, bounds):
            result = opt.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": self._max_iter}
            )
            return result.x, result.fun  # Optimierte Parameter und minimaler Funktionswert

        self.optimizer = new_optimizer
        return super()._constrained_optimization(obj_func, initial_theta, bounds)

# Wahre Belohnungsfunktion f*
def true_reward_function(context, action):
    if action == 0:
        #return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
        return 1/(1 + np.exp(-(np.tanh(0.5 * context[0] + 0.3 * context[1] + 0.1 * context[2]))))
    elif action == 1:
        return 1 / (1 + np.exp(-np.sin((0.1 * context[0] + 0.8 * context[1] + 0.1 * context[2]))))
        #return np.tanh(0.1 * context[0] + 0.8 * context[1] + 0.1 * context[2])
    elif action == 2:
        return 1 / (1 + np.exp(-np.cos((0.2 * context[0] + 0.2 * context[1] + 0.6 * context[2]))))
        #return np.tanh(0.3 * context[0] + 0.3 * context[1] + 0.6 * context[2])
    elif action == 3:
        return 1 / (1 + np.exp(-(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])))
        #return np.tanh(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])
    elif action == 4:
        return 1 / (1 + np.exp(-(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])))
        #return np.tanh(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])



# Gaussian Process Modelle für jeden Arm mit mu_0 = 0 und sigma_0 = 1
class GPUCB:
    def __init__(self, n_arms, n_features, n_rounds, cost, context, budget, p, seed):
        np.random.seed(seed)
        self.n_arms = n_arms
        self.n_features = n_features
        self.n_rounds = n_rounds
        self.budget = budget
        self.cost = cost
        self.p = p

        self.kernels = [RBF(length_scale=0.2, length_scale_bounds=(1e-60, 10)) for _ in range(n_arms)]
        self.gps = [
            MyGPR(kernel=kernel, alpha=1e-2, normalize_y=True, n_restarts_optimizer=10)
            for kernel in self.kernels
        ]

        self.arm_contexts = [[] for _ in range(n_arms)]
        self.arm_rewards = [[] for _ in range(n_arms)]
        self.opt_reward = []
        self.empirical_cost_means = np.random.rand(self.n_arms)
        self.cum = np.zeros(self.n_arms)

        self.selected_arms = []
        self.observed_rewards = []

        self.context = context
        self.arm_counts = np.ones(self.n_arms)
        self.sigma_t_1 = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
        self.B = 0.2 #max kernel norm
        self.R = 1#function range
        self.sigma_t_1 = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
        self.gamma = 0.1
        self.count = 0
        self.picked_arms = []



    def calculate_upper_confidence_bound(self, context, arm_id, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        mu_r, std = self.gps[arm_id].predict(context.reshape(1, -1), return_std=True)
        eta = 1
        arm_count = self.arm_counts[arm_id]
        z = np.sqrt(2* self.p* np.log(round + 2))
        if mu_r != 0 and mu_r != 1:
            eta = 1 #std / ((1 - mu_r) * mu_r)


       # print('LOOOL', mu_r )
        A = arm_count + z**2 * eta
        B = 2*arm_count*mu_r + z**2 * eta # eig noch * (M-m) aber das ist hier gleich 1
        C = arm_count* mu_r**2
        x = np.sqrt(np.clip((B**2 / (4* A**2)) - (C/A), 0, None))
        omega_r = (B/(2*A)) + x

        return omega_r


    def calculate_lower_confidence_bound(self, arm_id, round):
        mu_c = self.empirical_cost_means[arm_id]
        arm_count = self.arm_counts[arm_id]
        eta = 1
        z = np.sqrt(2 * self.p * np.log(round + 2))

        A = arm_count + z**2 * eta
        B = 2 * arm_count * mu_c + z**2 * eta  # eig noch * (M-m) aber das ist hier gleich 1
        C = arm_count * mu_c**2

        omega_c = B / (2 * A) + np.sqrt((B ** 2 / (4 * A ** 2)) - C / A)
        return np.clip(omega_c, 0.00000001, None)

    def run(self):
        t = 0
        while self.budget >= np.max(self.cost):
            current_context = self.context[t]
            ucb_values = []
            for arm_id in range(self.n_arms):
                if len(self.arm_contexts[arm_id]) > 0:
                    upper = self.calculate_upper_confidence_bound(current_context,arm_id, t)
                    lower = self.calculate_lower_confidence_bound(arm_id, t)
                    c_r_ratio = upper / lower
                else:
                    c_r_ratio = np.array([np.inf])
                ucb_values.append(c_r_ratio[0])

            selected_arm = np.argmax(ucb_values)
            self.arm_counts[selected_arm] += 1
            self.selected_arms.append(selected_arm)

            true_reward = true_reward_function(current_context, selected_arm)
            all_rewards = np.array([true_reward_function(current_context, arm) for arm in range(self.n_arms)])
            opt_arm = np.argmax(all_rewards/self.cost)
            if selected_arm != opt_arm:
                self.count = self.count + 1
                self.picked_arms.append(self.count)
            else:
                self.picked_arms.append(self.count)

            #print("opt_arm: ", opt_arm, "%n", "chosen: ", selected_arm )
            self.opt_reward.append(np.max(all_rewards/self.cost))
            observed_reward = true_reward
            self.observed_rewards.append(observed_reward/self.cost[selected_arm])

            self.arm_contexts[selected_arm].append(current_context)
            self.arm_rewards[selected_arm].append(observed_reward)

            if t <1000:
                self.gps[selected_arm].fit(
                    np.array(self.arm_contexts[selected_arm]),
                    np.array(self.arm_rewards[selected_arm])
                )

            if (len(self.arm_contexts[selected_arm]) + 1) % 1000 == 0:
                self.gps[selected_arm].fit(
                    np.array(self.arm_contexts[selected_arm][-999:]),
                    np.array(self.arm_rewards[selected_arm][-999:])
                )

            self.cum[selected_arm] += np.random.binomial(1, self.cost[selected_arm])
            self.empirical_cost_means[selected_arm] = self.cum[selected_arm] / (self.arm_counts[selected_arm] + 1)
            self.budget -= self.cost[selected_arm]
            print(t)
            t +=1


def generate_true_cost(num_arms, method='uniform'):
    """Erzeugt true_cost für die Banditen."""
    if method == 'uniform':
        return np.random.uniform(0.1, 1, num_arms)
    elif method == 'beta':
        return np.clip(np.random.beta(0.5, 0.5, num_arms), 0.01, 1)


np.random.seed(42)
# Parameter
n_arms = 3
n_rounds = 10000000
n_features = 3
num_points = 150

budget = 15000
regret_ucb = np.zeros(10000000)
p = [1]

context = [np.random.uniform(0, 1, n_features) for i in range(n_rounds)]

for i in range(1):
    print(i)
    cost = generate_true_cost(n_arms)
    gpucb_bandit = GPUCB(n_arms, n_features, n_rounds,cost, context,budget, np.random.choice(p), i)
    gpucb_bandit.run()
    regret_ucb = np.add(regret_ucb[:len(gpucb_bandit.observed_rewards)], np.array(gpucb_bandit.opt_reward) - np.array(gpucb_bandit.observed_rewards))


regret_ucb = np.array(regret_ucb)/10
plt.subplot(122)
plt.plot(regret_ucb.cumsum(), label='GP UCB')
plt.title("Cumulative regret")
plt.legend()
plt.show()

