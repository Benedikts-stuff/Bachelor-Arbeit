import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import os


class Logger:
    def __init__(self, filename="experiment_logs.csv"):
        self.filename = filename
        self.logs = []

    def log(self, algo_name, round_num, reward, regret, normalized_budget, run_index, seed):
        self.logs.append({
            "algorithm": algo_name,
            "round": round_num,
            "reward": reward,
            "regret": regret,
            "normalized_budget": normalized_budget,
            "run_index": run_index,
            "seed": seed
        })

    def save_to_csv(self):
        df = pd.DataFrame(self.logs)
        file_exists = os.path.isfile(self.filename)

        # Speichere Daten im Append-Modus ('a'), falls Datei existiert, sonst erstelle mit Header
        df.to_csv(self.filename, index=False, mode='a', header=not file_exists)
        self.logs = []  # Speicher freigeben


class Runner:
    def __init__(self, algo_name, bandit_class, bandit_params, reward_function, cost_function, generator, n_features,
                 n_runs=10, b=1000, filename="experiment_logs.csv"):
        self.algo_name = algo_name
        self.bandit_class = bandit_class
        self.bandit_params = bandit_params
        self.n_runs = n_runs
        self.generator = generator(n_features)
        self.get_reward = reward_function
        self.cost_function = cost_function
        self.B = b
        self.results = []
        self.filename = filename

    def _run_bandit(self, args):
        seed, run_index = args
        np.random.seed(seed)
        bandit = self.bandit_class(**self.bandit_params)
        rewards = []
        budget = self.B
        round_num = 0
        local_logger = Logger(self.filename)

        while budget > 0:
            context = self.generator.sample_uniform()
            action = bandit.select_action(context, round_num)
            reward = self.get_reward(context)
            cost = self.cost_function(context)
            bandit.update(reward[action], cost[action], action, context)

            optimal_reward =  np.max(np.array(reward)/np.array(cost))
            regret = optimal_reward - reward[action]/cost[action]
            normalized_budget = budget / self.B

            local_logger.log(self.algo_name, round_num, reward, regret, normalized_budget, run_index, seed)

            rewards.append(reward)
            budget -= cost[action]
            round_num += 1

        local_logger.save_to_csv()  # Speichert alle gesammelten Logs nach dem Run
        return np.cumsum(rewards)

    def run_experiment(self):
        with mp.Pool(mp.cpu_count()) as pool:
            self.results = pool.map(self._run_bandit, [(i, i) for i in range(self.n_runs)])

        return np.mean(self.results, axis=0)


class Executor:
    def __init__(self, algorithms, reward_function, cost_function, generator, n_features, n_runs, b, filename):
        self.algorithms = algorithms
        self.reward_function = reward_function
        self.cost_function = cost_function
        self.generator = generator
        self.n_features = n_features
        self.n_runs = n_runs
        self.budget = b
        self.filename = filename

    def run_all(self):
        # Falls Datei schon existiert, vorher löschen, um frische Daten zu speichern
        if os.path.exists(self.filename):
            os.remove(self.filename)

        for algo_name, bandit_class, bandit_params in self.algorithms:
            print(f"Running experiment for {algo_name}...")

            runner = Runner(algo_name, bandit_class, bandit_params, self.reward_function, self.cost_function,
                            self.generator, self.n_features, self.n_runs, self.budget, self.filename)
            runner.run_experiment()



# --- Beispielhafte Klassen ---
class LinGreedy:
    def __init__(self, context_dim, n_arms, epsilon, seed):
        np.random.seed(seed)
        self.n_features = context_dim
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.B = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.f_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.arm_counts = np.zeros(self.n_arms)
        self.gamma = 1e-8

    def select_action(self, context, i):
        epsilon = min(1, self.epsilon * (self.n_arms / (i + 1)))
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_arms)
        else:
            expected_rewards = np.array([np.dot(self.mu_hat[a], context) for a in range(self.n_arms)])
            expected_cost = np.array([np.dot(self.mu_hat_c[a], context) for a in range(self.n_arms)])

            return np.argmax(expected_rewards / (expected_cost + self.gamma))

    def update(self, reward, cost, chosen_arm, context):
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.f_c[chosen_arm] += cost * context
        self.mu_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])
        self.mu_hat_c[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f_c[chosen_arm])

        self.arm_counts[chosen_arm] += 1


class LinearReward:
    def __init__(self, n_arms, n_features):
        self.weights = np.random.rand(n_arms, n_features)
        self.weights /= self.weights.sum(axis=1, keepdims=True)

    def __call__(self, context):
        return np.dot(self.weights, context)

class LinearCost:
    def __init__(self, n_arms, n_features):
        self.weights = np.random.rand(n_arms, n_features)
        self.weights /= self.weights.sum(axis=1, keepdims=True)

    def __call__(self, context):
        return np.dot(self.weights, context)



class StochasticCost:
    def __init__(self, n_arms, n_features):
        self.weights = np.random.rand(n_arms, n_features)
        self.weights /= self.weights.sum(axis=1, keepdims=True)

    def __call__(self, context, action):
        probability = np.dot(self.weights[action], context)
        return np.random.binomial(1, probability)


class SyntheticContext:
    def __init__(self, num_features):
        self.num_features = num_features

    def sample_uniform(self):
        return np.random.rand(self.num_features)


def plot_cumulative_regret(csv_path):
    """
    Liest eine CSV-Datei ein, berechnet die kumulative Summe der Spalte "regret"
    und plottet diese.

    :param csv_path: Pfad zur CSV-Datei
    """
    # CSV-Datei einlesen
    df = pd.read_csv(csv_path)

    # Überprüfen, ob die Spalte "regret" existiert
    if 'regret' not in df.columns:
        raise ValueError("Die CSV-Datei enthält keine Spalte 'regret'.")

    # Kumulative Summe der Spalte "regret" berechnen
    df['cumulative_regret'] = df['regret'].cumsum()

    # Plot erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(df['cumulative_regret'], label='Kumulatives Regret')
    plt.title('Kumulatives Regret über die Zeit')
    plt.xlabel('Runde')
    plt.ylabel('Kumulatives Regret')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    algorithms = [
        ("LinGreedy_1", LinGreedy, {"context_dim": 5, "n_arms": 3, "epsilon": 0.1, "seed": 42}),
        #("LinGreedy_2", LinGreedy, {"context_dim": 5, "n_arms": 3, "epsilon": 0.2, "seed": 42}),
    ]

    reward_function = LinearReward(3, 5)
    cost_function = LinearCost(3, 5)
    filename = "experiment_logs.csv"

    executor = Executor(algorithms, reward_function, cost_function, SyntheticContext,
                        n_features=5, n_runs=1, b=1000, filename=filename)
    executor.run_all()
    # Beispielaufruf
    plot_cumulative_regret('experiment_logs.csv')