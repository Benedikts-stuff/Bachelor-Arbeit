import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt
from sklearn.utils.optimize import _check_optimize_result
from tqdm import tqdm
import scipy
import scipy.optimize as opt
from scipy.optimize import minimize

np.random.seed(42)
# Parameter
n_arms = 5  # Anzahl der Arme
n_rounds = 10000  # Anzahl der Runden
beta_t = 2  # Explorationsgewicht (β_t)
n_features = 8  # Anzahl der Kontextfeatures
noise_std = 0.1  # Standardabweichung des Rauschens
# müsste man mit einer art knee finding method anpassen ( prüfen ob der algo die verteilung gelernt hat)
train= [10, 20, 50 ,100, 500, 1000, 2000, 5000, 9000]
num_points = 150
X_test = np.linspace(-3, 3, num_points).reshape(-1, 1)

class MyGPR(GaussianProcessRegressor):
    def __init__(self, kernel, alpha, normalize_y, n_restarts_optimizer,  _max_iter=100):
        super().__init__(kernel, alpha=alpha, normalize_y= normalize_y, n_restarts_optimizer= n_restarts_optimizer)
        self._max_iter = _max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        def new_optimizer(obj_func, initial_theta, bounds):
            # scipy.optimize.minimize aufrufen
            result = opt.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": self._max_iter}
            )
            # Rückgabe an das erwartete Format anpassen
            return result.x, result.fun  # Optimierte Parameter und minimaler Funktionswert

        self.optimizer = new_optimizer  # Optimierer zuweisen
        return super()._constrained_optimization(obj_func, initial_theta, bounds)

# Wahre Belohnungsfunktion f*
def true_reward_function(context, arm_id):
    if arm_id == 0:
        return np.sin(0.5 * context[0] + 0.3)
    elif arm_id == 1:
        return np.exp(0.1 * context[0] + 0.8 * context[1] + 0.1 * context[2])
    elif arm_id == 2:
        return np.exp(0.3 * context[0] + 0.3 * context[1] + 0.6 * context[2])
    elif arm_id == 3:
        return np.exp(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])
    elif arm_id == 4:
        return np.exp(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])



# Gaussian Process Modelle für jeden Arm mit mu_0 = 0 und sigma_0 = 1
kernels = [
     RBF(length_scale=1.0, length_scale_bounds=(1e-10, 10))
    for _ in range(n_arms)
]
gps = [
    MyGPR(kernel=kernel, alpha=1e-2, normalize_y=True, n_restarts_optimizer=10)
    for kernel in kernels
]

# Speichern von Kontexten und Belohnungen für jeden Arm
arm_contexts = [[] for _ in range(n_arms)]
arm_rewards = [[] for _ in range(n_arms)]
opt_reward  =[]

# Ergebnisse speichern
selected_arms = []
observed_rewards = []

# GP-UCB Algorithmus

for t in tqdm(range(n_rounds)):
    # Generiere einen zufälligen Kontext
    current_context = np.random.uniform(-1, 1, n_features)

    # Berechne UCB für jeden Arm
    ucb_values = []
    for arm_id in range(n_arms):
        if len(arm_contexts[arm_id]) > 0:
            mu, sigma = gps[arm_id].predict(current_context.reshape(1, -1), return_std=True)
            ucb = mu + beta_t * sigma
        else:
            ucb = np.array([np.inf])  # Arme ohne Daten bekommen maximalen UCB
        ucb_values.append(ucb[0])

    # Wähle den Arm mit dem höchsten UCB
    selected_arm = np.argmax(ucb_values)
    selected_arms.append(selected_arm)

    # Berechne den echten Reward und füge Rauschen hinzu
    true_reward = true_reward_function(current_context, selected_arm)
    rew = [true_reward_function(current_context, arm) for arm in range(n_arms)]
    opt_reward.append(np.max(rew))
    observed_reward = true_reward
    observed_rewards.append(observed_reward)

    # Speichere den Kontext und Reward für den gewählten Arm
    arm_contexts[selected_arm].append(current_context)
    arm_rewards[selected_arm].append(observed_reward)

    # Update des GP-Modells für den gewählten Arm
    if t in train:
        gps[selected_arm].fit(np.array(arm_contexts[selected_arm]), np.array(arm_rewards[selected_arm]))


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

regret = np.array(opt_reward) - np.array(observed_rewards)

plt.subplot(122)
plt.plot(regret.cumsum(), label='linear model')
plt.title("Cumulative regret")
plt.legend()
plt.show()