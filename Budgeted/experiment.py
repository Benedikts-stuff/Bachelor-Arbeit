import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from tqdm import tqdm
import time  # Beispielweise zum Simulieren von Berechnungszeit

from Budgeted.c_b_thompson import ThompsonSamplingContextualBandit
#from c_b_thompson import ThompsonSamplingContextualBandit
from lin_ucb import LinUCB
from olr_e_greedy import EpsilonGreedyContextualBandit
from w_ucb import OmegaUCB

from multiprocessing import Pool
from time import sleep

REP = "rep"
ROUND = r"$t$"
BEST_ARM = "best-arm"
APPROACH = "Approach"
TIME = "time"
K = r"$K$"
CURRENT_ARM = r"$I_t$"
OPTIMAL_TOTAL_REWARD = r"$r_1$"
#OPTIMAL_COST = r"$c_1$"
SPENT_BUDGET = r"spent-budget"
EXPECTED_SPENT_BUDGET = r"expected-spent-budget"
ACTUAL_TOTAL_REWARD = "reward"
AVG_COST_OF_CURRENT_ARM = r"$\mu_i^c$"
AVG_REWARD_OF_CURRENT_ARM = r"$\mu_i^r$"
COST_OF_CURRENT_ARM = r"$c_{i,t}$"
REWARD_OF_CURRENT_ARM = r"$r_{i,t}$"
MINIMUM_AVERAGE_COST = r"$c_{min}$"
REGRET = "Regret"
NORMALIZED_SPENT_BUDGET = "Normalized Budget"
RHO = r"$\rho$"
IS_OUR_APPROACH = "our_approach"
APPROACH_ORDER = "order"
NORMALIZED_REGRET = "Normalized Regret"


all_ids = [
    REP,
    ROUND,
    APPROACH,
#    BEST_ARM,
#    CURRENT_ARM,
#    OPTIMAL_TOTAL_REWARD,
#    OPTIMAL_COST,
    SPENT_BUDGET,
#    EXPECTED_SPENT_BUDGET,
    ACTUAL_TOTAL_REWARD,
#    AVG_COST_OF_CURRENT_ARM,
#    AVG_REWARD_OF_CURRENT_ARM,
#    COST_OF_CURRENT_ARM,
#    REWARD_OF_CURRENT_ARM,
#    MINIMUM_AVERAGE_COST,
    REGRET,
#    TIME,
    NORMALIZED_SPENT_BUDGET,
#   APPROACH_ORDER
]


class Experiment:
    def __init__(self, functions, arguments, jobs):
       self.functions = functions
       self.arguments = arguments
       self.njobs = jobs

    # Funktion, die einen bestimmten Zeitraum wartet und dann den Wert zurückgibt
    def _wait(i):
        sleep(i)  # Warte für 'i' Sekunden
        return i  # Gib den Wert 'i' zurück

    # Funktion zur asynchronen Ausführung einer beliebigen Funktion in mehreren Prozessen
    def run_async(function, args_list, njobs, sleep_time_s=0.01):
        # Erstelle einen Pool von 'njobs' parallelen Prozessen
        pool = Pool(njobs)

        # Starte asynchrone Ausführung der Funktion mit verschiedenen Argumenten
        # `pool.apply_async` führt `function` asynchron mit den jeweiligen Argumenten aus
        results = [pool.apply_async(function, args=args) for args in args_list]

        # Überprüfe wiederholt, ob alle asynchronen Aufgaben abgeschlossen sind
        while not all(future.ready() for future in results):
            sleep(sleep_time_s)  # Warte eine kurze Zeit bevor erneut geprüft wird

        # Hole die Ergebnisse aller asynchronen Aufgaben ab, nachdem sie abgeschlossen sind
        results = [result.get() for result in results]

        # Schließe den Pool, sodass keine weiteren Aufgaben hinzugefügt werden können
        pool.close()

        return results  # Gib die Ergebnisse der asynchronen Aufgaben zurück

    def execute_parallel(self):
        # Hauptprogramm
        if __name__ == '__main__':
            #njobs = 3  # Anzahl der parallelen Prozesse, die genutzt werden sollen
            # Liste von Argumenten (hier: Wartezeiten), die an `_wait` übergeben werden
            delay = [[i] for i in range(4)]
            # Führe die Funktion `_wait` asynchron mit den Argumenten aus und speichere das Ergebnis
            result = self.run_async(self._wait, delay, self.njobs)
            print(result)  # Ausgabe: [0, 1, 2] (da `_wait(i)` jeweils den Wert `i` zurückgibt)


class BanditLogger:
    """
    Logger class to track the statistics of the bandits during execution
    """

    def __init__(self):
        self._columns = all_ids
        self._data = []
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._column_indices = {key: i for (i, key) in enumerate(self._columns)}

    def track_approach(self, value: str):
        self._track_value(value, APPROACH)

    def track_round(self, value: int):
        self._track_value(value, ROUND)

    def track_regret(self, value: float):
        self._track_value(value, REGRET)

    def track_normalized_budget(self, value: float):
        self._track_value(value, NORMALIZED_SPENT_BUDGET)

    def track_arm(self, value: int):
        self._track_value(value, CURRENT_ARM)

    def track_best_arm(self, value: int):
        self._track_value(value, BEST_ARM)

    def track_total_reward(self, value):
        self._track_value(value, ACTUAL_TOTAL_REWARD)

    def track_rep(self, value: int):
        self._track_value(value, REP)

    def track_optimal_reward(self, value: float):
        self._track_value(value, OPTIMAL_TOTAL_REWARD)

    def track_reward_sample(self, value: float):
        self._track_value(value, REWARD_OF_CURRENT_ARM)

    def track_spent_budget(self, value: float):
        self._track_value(value, SPENT_BUDGET)

    def finalize_round(self):
        """
        Add the current row to the data that will later become a data frame
        """
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        """
        Creates and returns the data frame based on the currently tracked results and the provided columns
        """
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

    def _track_value(self, newval, id):
        # Sets the value 'newval' for 'id' in the current row of the data frame
        self._current_row[self._index_of(id)] = newval

    def _index_of(self, id):
        return self._column_indices[id]

    def _reset(self):
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._data = []


class Runner:
    def __init__(self, n_rounds):
        self.iterations = n_rounds
        self.num_arms = 3
        self.num_features = 3
        self.num_rounds = 100000
        self.context = np.random.rand(self.num_rounds, self.num_features)
        self.budget = 4000
        self.df = pd.DataFrame(all_ids)
        self.normalized_budget_points = np.linspace(0, 1, 100)
        self.all_data = []
        self.logger =BanditLogger()
        self.df1 = pd.DataFrame(all_ids)
        self.df2 = pd.DataFrame(all_ids)

    def run_experiment(self):
        all_data1 = []  # Liste für die Daten von EpsilonGreedy
        all_data2 = []  # Liste für die Daten von Thompson Sampling

        for i in tqdm(range(self.iterations)):
            np.random.seed(i)  # Verwende den Iterationsindex, um die Zufälligkeit zu variieren

            true_weights = np.random.rand(self.num_arms, self.num_features)
            true_cost = np.random.uniform(0.1, 1, self.num_arms)
            #epsilon = [0.05, 0.1, 0.075, 0.025]
            #alpha = [0.1, 0.2, 0.3]

            # Bandits initialisieren und Experimente durchführen
            logger1 = BanditLogger()
            self.greedy_bandit = EpsilonGreedyContextualBandit(
                self.num_features, 1, self.num_arms, self.context, true_weights, true_cost, self.budget, logger1, i, i
            )
            self.greedy_bandit.run()
            df1 = logger1.get_dataframe()
            all_data1.append(df1)

            logger2 = BanditLogger()
            self.thompson_bandit = ThompsonSamplingContextualBandit(
                self.num_features, 0.1, self.num_arms, self.context, true_weights, true_cost, self.budget, logger2, i, i
            )
            self.thompson_bandit.run()
            print('muHat TS', self.thompson_bandit.mu_hat)
            print('mu true', true_weights)

            df2 = logger2.get_dataframe()
            all_data2.append(df2)

        # Alle Daten nach den Iterationen zusammenführen
        self.df1 = pd.concat(all_data1, ignore_index=True)
        self.df2 = pd.concat(all_data2, ignore_index=True)

    def plot_budget_normalised_regret(self):
        # Angenommen, df ist dein DataFrame
        #eg_data = self.df[self.df[APPROACH] == 0]
        #ts_data = self.df[self.df[APPROACH] == 1]
        sns.lineplot(x=NORMALIZED_SPENT_BUDGET, y=REGRET, data=self.df1, hue=REP)

        # Plot konfigurieren
        plt.xlabel('Normalized Budget')
        plt.ylabel('Cumulative Regret')
        plt.title('Regret Plot e-greedy')

        plt.show()

        sns.lineplot(x=NORMALIZED_SPENT_BUDGET, y=REGRET, data=self.df2, hue=REP)

        # Plot konfigurieren
        plt.xlabel('Normalized Budget')
        plt.ylabel('Cumulative Regret')
        plt.title('Regret Plot ts')

        # Zeige den Plot an
        plt.show()

runner = Runner(10)
runner.run_experiment()
runner.plot_budget_normalised_regret()

