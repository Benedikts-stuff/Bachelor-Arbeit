import os
from runner import Runner

class Executor:
    def __init__(self, algorithms, reward_function, cost_function, generator, n_features, n_runs, b, filename,bernoulli):
        self.algorithms = algorithms
        self.reward_function = reward_function
        self.cost_function = cost_function
        self.generator = generator
        self.n_features = n_features
        self.n_runs = n_runs
        self.budget = b
        self.filename = filename
        self.bernoulli = bernoulli

    def run_all(self):
        # Falls Datei schon existiert, vorher löschen, um frische Daten zu speichern
        if os.path.exists(self.filename):
            os.remove(self.filename)

        for algo_name, bandit_class, bandit_params in self.algorithms:
            print(f"Running experiment for {algo_name}...")

            runner = Runner(algo_name, bandit_class, bandit_params, self.reward_function, self.cost_function,
                            self.generator, self.n_features, self.n_runs, self.budget, self.filename, self.bernoulli)
            runner.run_experiment()