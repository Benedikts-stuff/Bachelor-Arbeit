import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
import scipy.optimize as opt
from matplotlib import pyplot as plt


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
def true_reward_function(context, arm_id):
    if arm_id == 0:
        #return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
        return 1/(1 + np.exp(-(0.5 * context[0] + 0.3 * context[1] + 0.1 * context[2])))
    elif arm_id == 1:
        return 1 / (1 + np.exp(-(0.1 * context[0] + 0.8 * context[1] + 0.1 * context[2])))
        #return np.tanh(0.1 * context[0] + 0.8 * context[1] + 0.1 * context[2])
    elif arm_id == 2:
        return 1 / (1 + np.exp(-(0.2 * context[0] + 0.2 * context[1] + 0.6 * context[2])))
        #return np.tanh(0.3 * context[0] + 0.3 * context[1] + 0.6 * context[2])
    elif arm_id == 3:
        return 1 / (1 + np.exp(-(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])))
        #return np.tanh(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])
    elif arm_id == 4:
        return 1 / (1 + np.exp(-(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])))
        #return np.tanh(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])

def true_reward_function_2(context, arm_id):
    if arm_id == 0:
        #return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
        x = (0.5 * context[0] + 0.3 * context[1] + 0.1 * context[2])
        reward = (
                10 * (x ** 4)  # Leading term for 4th degree
                - 15 * (x ** 3)  # Add negative cubic term
                + 6 * (x ** 2)  # Add positive quadratic term
                + 0.5 * x  # Linear term
                + 0.1  # Constant offset
        )

        return reward/ 1.7

    elif arm_id == 1:
        #return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
        x = (0.3 * context[0] + 0.5 * context[1] + 0.1 * context[2])
        reward = (
                10 * (x ** 4)  # Leading term for 4th degree
                - 15 * (x ** 3)  # Add negative cubic term
                + 6 * (x ** 2)  # Add positive quadratic term
                + 0.5 * x  # Linear term
                + 0.1  # Constant offset
        )

        return reward/ 1.7
    elif arm_id == 2:
        #return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
        x = (0.1 * context[0] + 0.3 * context[1] + 0.5 * context[2])
        reward = (
                10 * (x ** 4)  # Leading term for 4th degree
                - 15 * (x ** 3)  # Add negative cubic term
                + 6 * (x ** 2)  # Add positive quadratic term
                + 0.5 * x  # Linear term
                + 0.1  # Constant offset
        )

        return reward/ 1.7
    elif arm_id == 3:
        return 1 / (1 + np.exp(-(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])))
        #return np.tanh(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])
    elif arm_id == 4:
        return 1 / (1 + np.exp(-(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])))
        #return np.tanh(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])



# Gaussian Process Modelle für jeden Arm mit mu_0 = 0 und sigma_0 = 1
class GPUCB:
    def __init__(self, n_arms, n_features, n_rounds, beta_t, train_rounds, seed, context, gamma):
        np.random.seed(seed)
        self.n_arms = n_arms
        self.gamma = gamma
        self.n_features = n_features
        self.n_rounds = n_rounds
        self.beta_t = beta_t
        self.train_rounds = train_rounds

        self.kernels = [RBF(length_scale=0.2, length_scale_bounds=(1e-60, 10)) for _ in range(n_arms)]
        self.gps = [
            MyGPR(kernel=kernel, alpha=1e-2, normalize_y=True, n_restarts_optimizer=10)
            for kernel in self.kernels
        ]

        self.arm_contexts = [[] for _ in range(n_arms)]
        self.arm_rewards = [[] for _ in range(n_arms)]
        self.opt_reward = []

        self.selected_arms = []
        self.observed_rewards = []

        self.context = context
        self.arm_counts = np.ones(self.n_arms)
        self.sigma_t_1 = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
        self.B = 0.2 #max kernel norm
        self.R = 1#function range

        self.cost = [1, 1, 1]
        self.plot_rew = []


    def compute_beta_t(self, gain):
        beta_t = self.B + self.R * np.sqrt(2*(gain + 1+ np.log(1/self.gamma))) #gain is gamma
        return beta_t

    def run(self):
        for t in tqdm(range(self.n_rounds)):
            current_context = self.context[t]
            ucb_values = []
            for arm_id in range(self.n_arms):
                if len(self.arm_contexts[arm_id]) > 0:
                    mu, sigma = self.gps[arm_id].predict(current_context.reshape(1, -1), return_std=True)
                    beta = 2 * np.log(self.arm_counts[arm_id] * (t**2) * np.pi**2 / (6 * self.gamma))
                    #gain =  self.sigma_t_1[arm_id] - sigma
                    #self.sigma_t_1[arm_id] = sigma
                    #beta_t = self.compute_beta_t(gain)
                    ucb = mu + np.sqrt(beta) * sigma
                else:
                    ucb = np.array([np.inf])
                ucb_values.append(ucb[0])

            selected_arm = np.argmax(ucb_values/np.array(self.cost))
            self.arm_counts[selected_arm] += 1
            self.selected_arms.append(selected_arm)

            true_reward = true_reward_function(current_context, selected_arm)
            self.plot_rew.append(true_reward/self.cost[selected_arm])

            all_rewards = np.array([true_reward_function(current_context, arm) for arm in range(self.n_arms)])
            self.opt_reward.append(np.max(all_rewards/self.cost))    #kosten nicht vergessen
            observed_reward = true_reward
            self.observed_rewards.append(observed_reward)

            self.arm_contexts[selected_arm].append(current_context)
            self.arm_rewards[selected_arm].append(observed_reward)
            if t < 1000:
                self.gps[selected_arm].fit(
                    np.array(self.arm_contexts[selected_arm]),
                    np.array(self.arm_rewards[selected_arm])
                )
                #adaptive beta
        print("alpha", [self.gps[i].alpha for i in range(self.n_arms)])

context = [np.random.uniform(0, 1, 3) for i in range(2000)]

gpucb_bandit = GPUCB(n_arms = 3, n_features= 3, n_rounds =2000, beta_t=1, train_rounds=1, seed=0, context=context, gamma=0.1)
gpucb_bandit.run()
regret_ucb =np.array(gpucb_bandit.opt_reward) - np.array(gpucb_bandit.plot_rew)

plt.subplot(122)
plt.plot(regret_ucb.cumsum(), label='linear model')
plt.title("Cumulative regret")
plt.legend()
plt.show()