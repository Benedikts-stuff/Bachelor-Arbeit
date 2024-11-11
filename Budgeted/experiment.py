import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import pylab as pl
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

BGREEDY = "B-Greedy"
LINUCB = "Linear BUCB"
OMEGAUCB = "Omega BUCB"
THOMPSON = "Linear TS"

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

algorithms= [
BGREEDY,
LINUCB,
OMEGAUCB,
THOMPSON
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
        self.epsilon = np.array([0.05, 0.025, 0.1, 0.15, 0.125, 0.075, 0.09, 0.175])
        self.alpha = np.array([0.1, 0.2, 0.3])

    def run_experiment(self):
        all_data1 = []  # Liste für die Daten von EpsilonGreedy
        all_data2 = []  # Liste für die Daten von Thompson Sampling
        all_data3 = []  # Liste für die Daten von LinUCB
        all_data4 = []  # Liste für die Daten von OmegaUCB

        for i in tqdm(range(self.iterations)):
            np.random.seed(i)  # Verwende den Iterationsindex, um die Zufälligkeit zu variieren

            true_weights = np.random.rand(self.num_arms, self.num_features)
            true_cost = np.random.uniform(0.1, 1, self.num_arms)

            # Bandits initialisieren und Experimente durchführen
            #Epsilon Greedy
            logger1 = BanditLogger()
            self.greedy_bandit = EpsilonGreedyContextualBandit(
                self.num_features, np.random.choice(self.epsilon), self.num_arms, self.context, true_weights, true_cost, self.budget, logger1, i, i
            )
            self.greedy_bandit.run()
            df1 = logger1.get_dataframe()
            all_data1.append(df1)

            #Thompson sampling
            logger2 = BanditLogger()
            self.thompson_bandit = ThompsonSamplingContextualBandit(
                self.num_features, 1, self.num_arms, self.context, true_weights, true_cost, self.budget, logger2, i, i
            )
            self.thompson_bandit.run()

            df2 = logger2.get_dataframe()
            all_data2.append(df2)

            #LinUCB
            logger3 = BanditLogger()
            self.lin_ucb= LinUCB(
                self.num_arms, self.num_features, self.context, true_weights, true_cost, np.random.choice(self.alpha), self.budget, logger3, i, i
            )
            self.lin_ucb.run()

            df3 = logger3.get_dataframe()
            all_data3.append(df3)


            #OmegaLinUCB
            logger4 = BanditLogger()
            self.omega_lin_ucb= OmegaUCB(
                self.num_arms, self.num_features, self.context, true_weights, true_cost, np.random.choice(self.alpha), self.budget, logger4, i, i
            )
            self.omega_lin_ucb.run()

            df4 = logger4.get_dataframe()
            all_data4.append(df4)

        # Alle Daten nach den Iterationen zusammenführen
        #self.df1 = pd.concat(all_data1, ignore_index=True)
        #self.df2 = pd.concat(all_data2, ignore_index=True)
        plot_data = np.array([self.interp_plot(all_data1), self.interp_plot(all_data2), self.interp_plot(all_data3), self.interp_plot(all_data4)])
        self.plot_budget_normalised_regret(plot_data)

    def plot_budget_normalised_regret(self, plot_data):
        # Angenommen, df ist dein DataFrame
        #eg_data = self.df[self.df[APPROACH] == 0]
        #ts_data = self.df[self.df[APPROACH] == 1]

        pl.ylim(0,500)
        i = 0
        for line in plot_data:
            plt.plot(line[0], line[1], '--', color='red', label=algorithms[i])
            i +=1
        plt.xlabel(NORMALIZED_SPENT_BUDGET)
        plt.ylabel(REGRET)
        plt.legend()
        plt.show()

        #sns.lineplot(x=NORMALIZED_SPENT_BUDGET, y=REGRET, data=self.df1, hue=REP)

        # Plot konfigurieren
        #plt.xlabel('Normalized Budget')
        #plt.ylabel('Cumulative Regret')
        #plt.ylim(0,3000)
        #plt.title('Regret Plot e-greedy')

        #plt.show()

        #sns.lineplot(x=NORMALIZED_SPENT_BUDGET, y=REGRET, data=self.df2, hue=REP)

        # Plot konfigurieren
        #plt.xlabel('Normalized Budget')
        #plt.ylabel('Cumulative Regret')
        #plt.title('Regret Plot ts')

        # Zeige den Plot an
        #plt.show()

    def interp_plot(self, dfs, x_col=NORMALIZED_SPENT_BUDGET, y_col=REGRET):
        # Liste der Achsen-Daten erstellen
        axis_list = []
        for df in dfs:
            data = df[[x_col, y_col]].sort_values(by=x_col).drop_duplicates(x_col).to_numpy()
            axis_list.append(data)

        # Minimum und Maximum für jeden Datensatz finden
        min_max_xs = [(min(axis[:, 0]), max(axis[:, 0])) for axis in axis_list]
        new_axis_xs = [np.linspace(0, 1, 100) for min_x, max_x in min_max_xs]

        # Interpolierte Y-Werte berechnen
        new_axis_ys = [np.interp(new_x_axis, axis[:, 0], axis[:, 1]) for axis, new_x_axis in
                       zip(axis_list, new_axis_xs)]

        # Mittelwert der X- und Y-Werte berechnen
        midx = [np.mean([new_axis_xs[axis_idx][i] for axis_idx in range(len(axis_list))]) for i in range(100)]
        midy = [np.mean([new_axis_ys[axis_idx][i] for axis_idx in range(len(axis_list))]) for i in range(100)]

        # Plot für jeden Durchlauf und den Durchschnitt erstellen
        #for axis in axis_list:
           # plt.plot(axis[:, 0], axis[:, 1], color='black', alpha=0.3)
        return np.array([midx, midy])
        #pl.ylim(0,500)
        #plt.plot(midx, midy, '--', color='red', label='Interpolierter Mittelwert')
        #plt.xlabel(x_col)
        #plt.ylabel(y_col)
        #plt.legend()
        #plt.show()

runner = Runner(4)
runner.run_experiment()

