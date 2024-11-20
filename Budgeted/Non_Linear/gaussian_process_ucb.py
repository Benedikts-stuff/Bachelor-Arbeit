import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
import scipy.optimize as opt

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
        return np.exp(0.5 * context[0] + 0.3 * context[1] + 0.6*context[2])
    elif arm_id == 1:
        return np.exp(0.1 * context[0] + 0.8 * context[1] + 0.1 * context[2])
    elif arm_id == 2:
        return np.exp(0.3 * context[0] + 0.3 * context[1] + 0.6 * context[2])
    elif arm_id == 3:
        return np.exp(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])
    elif arm_id == 4:
        return np.exp(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])



# Gaussian Process Modelle für jeden Arm mit mu_0 = 0 und sigma_0 = 1
class GPUCB:
    def __init__(self, n_arms, n_features, n_rounds, beta_t, train_rounds, seed, context):
        np.random.seed(seed)
        self.n_arms = n_arms
        self.n_features = n_features
        self.n_rounds = n_rounds
        self.beta_t = beta_t
        self.train_rounds = train_rounds

        self.kernels = [RBF(length_scale=1.0, length_scale_bounds=(1e-10, 10)) for _ in range(n_arms)]
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

    def run(self):
        for t in tqdm(range(self.n_rounds)):
            current_context = self.context[t]
            #adaptive beta
            #self.beta_t = 1/(np.log(t+1.00001)**2)
            ucb_values = []
            for arm_id in range(self.n_arms):
                if len(self.arm_contexts[arm_id]) > 0:
                    mu, sigma = self.gps[arm_id].predict(current_context.reshape(1, -1), return_std=True)
                    ucb = mu + np.sqrt(self.beta_t) * sigma
                else:
                    ucb = np.array([np.inf])
                ucb_values.append(ucb[0])

            selected_arm = np.argmax(ucb_values)
            self.selected_arms.append(selected_arm)

            true_reward = true_reward_function(current_context, selected_arm)
            all_rewards = [true_reward_function(current_context, arm) for arm in range(self.n_arms)]
            self.opt_reward.append(np.max(all_rewards))
            observed_reward = true_reward
            self.observed_rewards.append(observed_reward)

            self.arm_contexts[selected_arm].append(current_context)
            self.arm_rewards[selected_arm].append(observed_reward)

            if t in self.train_rounds:
                self.gps[selected_arm].fit(
                    np.array(self.arm_contexts[selected_arm]),
                    np.array(self.arm_rewards[selected_arm])
                )
                #adaptive beta
                #self.beta_t = 1 / (np.log(t))
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

#regret = np.array(opt_reward) - np.array(observed_rewards)

#plt.subplot(122)
#plt.plot(regret.cumsum(), label='linear model')
#plt.title("Cumulative regret")
#plt.legend()
#plt.show()