import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt
from tqdm import tqdm

np.random.seed(42)
# Parameter
n_arms = 5  # Anzahl der Arme
n_rounds = 2000  # Anzahl der Runden
beta_t = 2  # Explorationsgewicht (β_t)
n_features = 3  # Anzahl der Kontextfeatures
noise_std = 0.1  # Standardabweichung des Rauschens


# Wahre Belohnungsfunktion f*
def true_reward_function(context, arm_id):
    return 0.5 * context[0] + 0.3 * (arm_id + 1) * context[1] **2 - 0.2 * context[2]**3


# Gaussian Process Modelle für jeden Arm
kernels = [
    C(1.0, (1e-3, 100)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10))
    for _ in range(n_arms)
]
gps = [
    GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True, n_restarts_optimizer=10)
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
    opt_reward.append(np.max([true_reward_function(current_context, arm) for arm in range(n_arms)]))
    observed_reward = true_reward #+ np.random.normal(0, noise_std)
    observed_rewards.append(observed_reward)

    # Speichere den Kontext und Reward für den gewählten Arm
    arm_contexts[selected_arm].append(current_context)
    arm_rewards[selected_arm].append(observed_reward)

    # Update des GP-Modells für den gewählten Arm
    if t<650:
        gps[selected_arm].fit(np.array(arm_contexts[selected_arm]), np.array(arm_rewards[selected_arm]))

# Ergebnisse anzeigen
print("Ausgewählte Arme (erste 10 Runden):", selected_arms[:10])
print("Beobachtete Rewards (erste 10 Runden):", observed_rewards[:10])
regret = np.array(opt_reward) - np.array(observed_rewards)

plt.subplot(122)
plt.plot(regret.cumsum(), label='linear model')
plt.title("Cumulative regret")
plt.legend()
plt.show()