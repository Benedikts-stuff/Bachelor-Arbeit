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


# Gaussian Process Modelle für jeden Arm mit mu_0 = 0 und sigma_0 = 1
class GPTS:
    def __init__(self, n_arms, n_features, context, trueweights, cost, budget, logger,  repetition,  seed, delta= 0.1):
        np.random.seed(seed)
        self.n_arms = n_arms
       # self.lambda_ = lambda_
        self.n_features = n_features


        self.kernels = [RBF(length_scale=0.2, length_scale_bounds=(1e-60, 10)) for _ in range(n_arms)]
        self.gps = [
            MyGPR(kernel=kernel, alpha=1e-2, normalize_y=True, n_restarts_optimizer=10)
            for kernel in self.kernels
        ]

        self.arm_contexts = [[] for _ in range(n_arms)]
        self.arm_rewards = [[] for _ in range(n_arms)]
        self.opt_reward = []
        self.true_weights = trueweights
        self.cost = cost
        self.budget = budget
        self.logger = logger
        self.repetition = repetition
        self.og_budget = budget
        self.summed_regret = 0
        self.cum = np.zeros(self.n_arms)
        self.alpha =np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

        self.empirical_cost_means = np.random.rand(self.n_arms)

        self.selected_arms = []
        self.observed_rewards = []

        self.context = context

        self.B = 0.2 #max kernel norm
        self.R = 1#max reward
        self.gamma = delta # initialisierung

        self.sigma_t_1 = np.array([2.5, 2.5 ,2.5, 2.5, 2.5])

        self.arm_counts = np.ones(n_arms)

    def compute_beta_t(self, gain):
        beta_t = self.B + self.R * np.sqrt(2*(gain + 1+ np.log(2/self.gamma))) #gain is gamma
        return beta_t

    def run(self):
       t= 0
       while self.budget > np.max(self.cost):
            context = self.context[t]
            sampled_values = []
            cost = np.zeros(self.n_arms)
            for arm_id in range(self.n_arms):
                if len(self.arm_contexts[arm_id]) > 0:
                    mu, sigma = self.gps[arm_id].predict(context.reshape(1, -1), return_std=True)
                    # Kovarianzmatrix: Quadratischer Wert von sigma, da wir nur 1 Punkt haben
                    # Samplen aus N(mu, K)
                    #beta = self.compute_beta_t(t)
                    #if t > 6000: beta = 0.000000001
                    #vorher np.sqrt(self.lambda_) * sigma
                    #This works
                    gain = self.sigma_t_1[arm_id] - sigma
                    self.sigma_t_1[arm_id] = sigma
                    beta_t = self.compute_beta_t(gain)
                    #print("true rewards", np.dot(self.true_weights[arm_id], context), "guess: ", mu)
                    sampled_value = np.random.normal(mu, beta_t * sigma)

                    cost = np.array([np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)])
                else:
                    sampled_value = np.array([np.inf])  # Für untrainierte Arme
                sampled_values.append(sampled_value[0])

            selected_arm = np.argmax(sampled_values/(cost + 0.000001))
            self.arm_counts[selected_arm] += 1
            self.selected_arms.append(selected_arm)

            true_reward = np.dot(self.true_weights[selected_arm], context)
            all_rewards = [np.dot(self.true_weights[arm], context) for arm in range(self.n_arms)]
            self.opt_reward.append(np.max(all_rewards/ (np.array(self.cost) + 0.000001)))
            observed_reward = true_reward
            self.observed_rewards.append(observed_reward)

            self.arm_contexts[selected_arm].append(context)
            self.arm_rewards[selected_arm].append(observed_reward)

            cost_t = np.random.binomial(1, self.cost[selected_arm])
            self.alpha[selected_arm] += cost_t
            self.beta[selected_arm] += 1 - cost_t
            self.arm_counts[selected_arm] += 1

            if t < 1000:
                self.gps[selected_arm].fit(
                    np.array(self.arm_contexts[selected_arm]),
                    np.array(self.arm_rewards[selected_arm])
                )
                #adaptive beta
                #self.beta_t = 1 / (np.log(t))

            self.summed_regret +=self.opt_reward[t] - (all_rewards[selected_arm]/ (self.cost[selected_arm] +0.000001))
            self.budget -= self.cost[selected_arm]

            self.logger.track_rep(self.repetition)
            self.logger.track_approach(1)
            self.logger.track_round(t)
            self.logger.track_regret(self.summed_regret)
            self.logger.track_normalized_budget((self.og_budget - self.budget)/ self.og_budget)
            self.logger.track_spent_budget(self.og_budget - self.budget)
            self.logger.finalize_round()
            t+=1


#plt.figure(figsize=(10, 6))
#plt.plot(X_test, [true_reward_function(x, 0) for x in X_test], 'r:', label='Wahre Funktion')
#plt.plot(X_test, np.array([gps[0].predict(x.reshape(-1,1))[0] for x in X_test]), 'b-', label='Gelernte Funktion (GP Vorhersage)')
#plt.fill_between(X_test.ravel(), mu - 1.96 * sigma, mu + 1.96 * sigma, alpha=0.2, color='blue', label='95% Unsicherheit')
#plt.scatter(X_train, y_train, color='black', zorder=10, label='Trainingsdaten')
#plt.legend(loc='upper left')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('Gaussian Process Regression')
#plt.show()


#np.random.seed(42)
#train= [10, 20, 50 ,100, 200, 300, 500,700, 800, 1000, 2500, 5000, 7500,11000, 18000]
#n_rounds = 1000
#n_features = 3
#context = [np.random.uniform(-1, 1, n_features) for i in range(n_rounds)]

#bandit = GPTS(5, n_features, n_rounds, train, 42,  context, 1)
#bandit.run()
#regret = np.array(bandit.opt_reward) - np.array(bandit.observed_rewards)

#plt.subplot(122)
#plt.plot(regret.cumsum(), label='linear model')
#plt.title("Cumulative regret")
#plt.legend()
#plt.show()