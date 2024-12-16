import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
import scipy.optimize as opt
from matplotlib import pyplot as plt
from tqdm import tqdm
import time



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


# Gaussian Process Modelle für jeden Arm mit mu_0 = 0 und sigma_0 = 1
class GPWUCB:
    def __init__(self, n_arms, n_features, context, true_theta, cost, budget, p, repetition, logger, seed):
        np.random.seed(seed)
        self.n_arms = n_arms
        self.n_features = n_features
        self.beta_t = 1
        self.logger = logger
        self.true_theta = true_theta
        self.cost = cost
        self.budget = budget
        self.og_budget = budget
        self.repetition = repetition
        self.empirical_cost_means = np.random.rand(self.n_arms)
        self.cum = np.zeros(self.n_arms)

        self.kernels = [RBF(length_scale=0.2, length_scale_bounds=(1e-60, 10)) for _ in range(n_arms)]
        self.gps = [
            MyGPR(kernel=kernel, alpha=1e-2, normalize_y=True, n_restarts_optimizer=10)
            for kernel in self.kernels
        ]

        self.arm_contexts = [[] for _ in range(n_arms)]
        self.arm_rewards = [[] for _ in range(n_arms)]
        self.opt_reward = []

        self.selected_arms = []

        self.context = context
        self.arm_counts = np.ones(self.n_arms)

        #self.cost = [1, 1, 1]
        self.summed_regret = 0
        self.p =p


    def calculate_upper_confidence_bound(self, context, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        upper =[]
        for i in range(self.n_arms):
            if len(self.arm_contexts[i]) > 0:
                mu_r,sigma= self.gps[i].predict(context.reshape(1, -1), return_std=True)
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
                omega_r = np.clip((B/(2*A)) + x, 0, None)
                upper.append(omega_r[0])
            else:
                ucb = np.array([np.inf])
                upper.append(ucb[0])
        # Adjust for cost and return estimated reward per cost ratio
        return upper

    def calculate_lower_confidence_bound(self, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        lower = []
        for i in range(self.n_arms):
            mu_c = self.empirical_cost_means[i]

            arm_count = self.arm_counts[i]
            eta = 1
            z = np.sqrt(2 * self.p * np.log(round + 2))

            A = arm_count + z**2 * eta
            B = 2 * arm_count * mu_c + z**2 * eta  # eig noch * (M-m) aber das ist hier gleich 1
            C = arm_count * mu_c**2

            omega_c = B / (2 * A) - np.sqrt((B ** 2 / (4 * A ** 2)) - C / A)
            lower.append(np.clip(omega_c, 0.000001, None))
        # Adjust for cost and return estimated reward per cost ratio
        return lower


    def run(self):
        true_rewards = self.context.dot(self.true_theta.T)
        t = 0
        progress = tqdm(total=100000, desc="Processing gp_ucb", unit="step",ncols=100, position=None)  # Fortschrittsbalken ohne Total
        while self.budget > np.max(self.cost):
            current_context = self.context[t]

            upper = np.array(self.calculate_upper_confidence_bound(current_context, t))
            lower = np.array(self.calculate_lower_confidence_bound(t))

            selected_arm = np.argmax(upper/(lower))

            self.budget -= self.cost[selected_arm]
            self.cum[selected_arm] += np.random.binomial(1, self.cost[selected_arm])
            self.empirical_cost_means[selected_arm] = self.cum[selected_arm] / (self.arm_counts[selected_arm] + 1)

            self.arm_counts[selected_arm] += 1
            self.selected_arms.append(selected_arm)

            true_reward = true_rewards[t, selected_arm]

            all_rewards = np.array([true_rewards[t, arm] for arm in range(self.n_arms)])
            self.opt_reward.append(np.max(all_rewards/self.cost))    #kosten nicht vergessen
            observed_reward = true_reward

            self.arm_contexts[selected_arm].append(current_context)
            self.arm_rewards[selected_arm].append(observed_reward)
            if t < 1000:
                self.gps[selected_arm].fit(
                    np.array(self.arm_contexts[selected_arm]),
                    np.array(self.arm_rewards[selected_arm])
                )
                #adaptive beta

            self.summed_regret += self.opt_reward[t] - (true_reward / self.cost[selected_arm])

            self.logger.track_rep(self.repetition)
            self.logger.track_approach(0)
            self.logger.track_round(t)
            self.logger.track_regret(self.summed_regret)
            self.logger.track_normalized_budget((self.og_budget - self.budget)/ self.og_budget)
            self.logger.track_spent_budget(self.og_budget - self.budget)
            self.logger.finalize_round()



            t+=1
            progress.update(1)  # Fortschrittsbalken aktualisieren
        progress.close()
        print('gpUCB finish')

#context = [np.random.uniform(0, 1, 3) for i in range(2000)]

#gpucb_bandit = GPUCB(n_arms = 3, n_features= 3, n_rounds =2000, beta_t=1, train_rounds=1, seed=0, context=context, gamma=0.1)
#gpucb_bandit.run()
#regret_ucb =np.array(gpucb_bandit.opt_reward) - np.array(gpucb_bandit.plot_rew)

#plt.subplot(122)
#plt.plot(regret_ucb.cumsum(), label='linear model')
#plt.title("Cumulative regret")
#plt.legend()
#plt.show()