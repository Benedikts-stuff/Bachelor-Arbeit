import pandas as pd
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from concurrent.futures import as_completed
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

multiprocessing.set_start_method("spawn", force=True)
from tqdm import tqdm
from utils import*
import time
from bandit_factory import BanditFactory

REP = "rep"
ROUND = r"$t$"
APPROACH = "Approach"
CURRENT_ARM = r"$I_t$"
SPENT_BUDGET = r"spent-budget"
ACTUAL_TOTAL_REWARD = "reward"
REGRET = "Regret"
NORMALIZED_SPENT_BUDGET = "Normalized Budget"
BGREEDY = "B-Greedy"
LINUCB = "Linear BUCB"
OMEGAUCB = "Omega BUCB"
THOMPSON = "Linear TS"

all_ids = [
    REP,
    ROUND,
    APPROACH,
    SPENT_BUDGET,
    ACTUAL_TOTAL_REWARD,
    REGRET,
    NORMALIZED_SPENT_BUDGET,
]

algorithms= [
    BGREEDY,
    LINUCB,
    OMEGAUCB,
    THOMPSON
]



# Runner-Klasse
class Runner:
    def __init__(self, n_rounds, reward_typ,c,cost_type,cost_index,  num_arms=3, num_features=3, budget=1000, adversary= False):
        self.iterations = n_rounds
        self.adversary = adversary
        self.cdc = c
        self.cost_type = cost_type
        self.num_arms = num_arms
        self.num_features = num_features
        self.num_rounds = 10000000
        self.context = np.random.rand(self.num_rounds, self.num_features)
        self.budget = budget
        self.normalized_budget_points = np.linspace(0, 1, 100)
        self.epsilon = np.array([4]) #np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.p =np.array([0.25]) #oder 1 wählen
        self.gamma= np.array([0.1])
        self.cost_kind = ['continuous', 'bernoulli']
        self.cost_index =cost_index
        self.reward_type = reward_typ

        self.algorithms = ['Budget_CB']


    def run_experiment(self):

        all_data = {name: [] for name in self.algorithms} #['C-UCB', 'C-ThompsonSampling', 'OmegaUCB','NeuralOmegaUCB', 'LinUCB', 'EpsilonGreedy', 'ThompsonSampling', 'GPUCB', 'GPTS']}   #{name: [] for name in ['EpsilonGreedy', 'ThompsonSampling', 'LinUCB', 'OmegaUCB']}
        #['LinOmegaUCB','NeuralOmegaUCB', 'LinUCB', 'GPUCB', 'GPTS']
        #['GPWUCB','C-LinUCB', 'LinOmegaUCB','NeuralOmegaUCB', 'LinUCB', 'GPUCB']
        #[ 'C-LinearThompsonSampling', 'EpsilonGreedy', 'LinearThompsonSampling', 'GPTS']
        #['C-LinearThompsonSampling', 'LinearEpsilonGreedy', 'LinearThompsonSampling', 'LinOmegaUCB', 'C-LinUCB', 'LinUCB' ]
        # ['GPWUCB', 'NeuralOmegaUCB', 'GPUCB', 'GPTS']
        # ['LinUCB_CDC', 'NeuralOmegaUCB_CDC',  'LinearEpsilonGreedy_CDC', 'LinOmegaUCB_CDC' ]
        #['C-LinearThompsonSampling', 'LinearEpsilonGreedy', 'LinearThompsonSampling', 'LinOmegaUCB', 'C-LinUCB', 'LinUCB', 'RandomBandit']
        #'Budget_CB', 'LinOmegaUCB_CDC', 'LinearEpsilonGreedy_CDC'
        with ProcessPoolExecutor(max_workers=7) as executor:
            futures = []

            if __name__ == '__main__':
                for i in tqdm(range(self.iterations), desc="Running Bandit Experiments"):
                    true_weights = generate_true_weights(self.num_arms, self.num_features, seed=i, method="reward")
                    true_cost_weights = generate_true_weights(self.num_arms, self.num_features,seed=i +42, method="reward")
                    true_cost = generate_true_cost(self.num_arms, self.cdc)

                    for bandit_type in all_data.keys():
                        futures.append(
                            executor.submit(self.run_bandit, bandit_type, true_weights, true_cost, i, true_cost_weights, self.cost_kind[self.cost_index], self.reward_type, self.adversary)
                        )

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Futures"):
                bandit_type, result = future.result()
                all_data[bandit_type].append(result)

        if __name__ == '__main__':
            plot_data = {name: self.interp_plot(data) for name, data in all_data.items()}
            mapping = create_global_color_mapping(plot_data)

            self.plot_budget_normalised_regret(plot_data, mapping)
            self.plot_regret_distribution_violin(plot_data, mapping)

    def run_bandit(self, bandit_type, true_weights, true_cost, seed, true_cost_weights, cost_kind, reward_type, adversary):
        np.random.seed(seed)
        bandit = None
        if self.cdc:
            if reward_type == 'linear' and self.cost_type == 'linear':
                bandit = BanditFactory.create(bandit_type, self, true_weights, true_cost, seed, true_cost_weights, cost_kind,adversary, 0, linear_reward, linear_cost)
            elif reward_type == 'polynomial' and self.cost_type == 'linear':
                bandit = BanditFactory.create(bandit_type, self, true_weights, true_cost, seed, true_cost_weights,
                                              cost_kind, polynomial_reward,adversary, linear_cost, adversary)
            elif reward_type == 'linear' and self.cost_type == 'polynomial':
                bandit = BanditFactory.create(bandit_type, self, true_weights, true_cost, seed, true_cost_weights,
                                              cost_kind,adversary, linear_reward, polynomial_cost, adversary)
            elif reward_type == 'polynomial' and self.cost_type == 'polynomial':
                bandit = BanditFactory.create(bandit_type, self, true_weights, true_cost, seed, true_cost_weights,
                                              cost_kind,adversary, polynomial_reward, polynomial_cost)

        else:
            bandit = BanditFactory.create(bandit_type, self, true_weights, true_cost, seed, true_cost_weights,
                                          cost_kind, adversary, 0, linear_reward, linear_cost)


        bandit.run()
        return bandit_type, bandit.logger.get_dataframe()

    def interp_plot(self, dfs, x_col=NORMALIZED_SPENT_BUDGET, y_col=REGRET):

        axis_list = [
            df[[x_col, y_col]].sort_values(by=x_col).drop_duplicates(x_col).to_numpy() for df in dfs
        ]
        new_axis_xs = np.linspace(0, 1, 100)  # Generate evenly spaced x-axis values
        new_axis_ys = [np.interp(new_axis_xs, axis[:, 0], axis[:, 1]) for axis in axis_list]

        # Compute statistical metrics
        midy = np.mean(new_axis_ys, axis=0)
        q25 = np.percentile(new_axis_ys, 25, axis=0)
        q75 = np.percentile(new_axis_ys, 75, axis=0)
        median = np.median(new_axis_ys)
        std = np.std(new_axis_ys)
        summed = [np.max(regret) for regret in new_axis_ys]

        # Return as a DataFrame
        return pd.DataFrame({
            x_col: new_axis_xs,
            y_col: midy,
            "summed": np.pad(summed, (0, len(new_axis_xs) - len(summed)), mode='constant', constant_values=np.nan),
            "q25": q25,
            "q75": q75,
        })

    def plot_budget_normalised_regret(self, plot_data, color_mapping):
        plt.figure(figsize=(10, 6))

        # Initialisiere ein Mapping von Methoden zu Farben
        styles = {
            'LinearEpsilonGreedy': ('-', 0),
            'LinearEpsilonGreedy2': ('-', 1),
            'LinearThompsonSampling': ('-', 2),
            'LinUCB': ('-', 3),
            'LinOmegaUCB': ('-', 4),
            'NeuralOmegaUCB': ('-', 5),
            'GPUCB': ('-', 6),
            'GPWUCB': ('-', 7),
            'GPTS': ('-', 8),
            'C-LinUCB': ('-', 9),
            'C-LinearThompsonSampling': ('-', 10),
            'LinOmegaUCB_CDC': ('-', 11),
            'LinearEpsilonGreedy_CDC': ('-', 12),
            'RandomBandit': ('-', 13),
            'LinUCB_CDC': ('-', 14),
            'NeuralOmegaUCB_CDC': ('-', 15),
            'Budget_CB': ('-', 16),
            'NeuralGreedy': ('-', 17),
            'EpsilonFirst': ('-', 18),
            'C_LinUCB_CDC':  ('-', 18),

        }

        y_lim = 0
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'sourcesanspro'
        plt.rcParams[
            'text.latex.preamble'] = r'\usepackage{libertine}\usepackage[libertine]{newtxmath}\RequirePackage[scaled=.92]{sourcesanspro}'
        # Zeichne den Violin-Plot mit schönerem Design

        for index, (name, df) in enumerate(plot_data.items()):
            # Hole Farbe und Linienstil aus der Palette
            style = '-'
            color = color_mapping[name]

            sns.lineplot(x=NORMALIZED_SPENT_BUDGET, y=REGRET, data=df, label=name, color=color, linestyle=style, linewidth=2)
            y_lim = max(y_lim, df[REGRET].max())

        plt.xlabel("Normalized Spent Budget", fontsize=14)
        plt.ylabel("Cumulative Regret", fontsize=14)
        plt.ylim(0, y_lim)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig("./data/cumulative_regret_linear_bern.pdf", dpi=300)  # Höhere Auflösung für bessere Qualität
        plt.show()

    def plot_regret_distribution_violin(self, plot_data, color_mapping):
        plt.figure(figsize=(12, 8))

        # Initialisiere eine Liste, um die kombinierten DataFrame-Daten zu speichern
        combined_data = []

        for name, df in plot_data.items():
            # Hole die 'summed' Werte und den Algorithmus-Namen
            summed = df["summed"].head(self.iterations).to_numpy()

            # Erstelle ein DataFrame für diesen Algorithmus
            temp_df = pd.DataFrame({
                'Algorithm': [name] * len(summed),  # Setze den Algorithmus-Namen
                'summed': summed  # Die 'summed' Werte
            })

            # Füge dieses DataFrame zu den kombinierten Daten hinzu
            combined_data.append(temp_df)

        # Kombiniere alle DataFrames zu einem einzigen DataFrame
        combined_df = pd.concat(combined_data)

        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'sourcesanspro'
        plt.rcParams[
            'text.latex.preamble'] = r'\usepackage{libertine}\usepackage[libertine]{newtxmath}\RequirePackage[scaled=.92]{sourcesanspro}'
        # Zeichne den Violin-Plot mit schönerem Design
        sns.violinplot(x="Algorithm", y="summed", data=combined_df, inner="quart", palette=color_mapping, scale="area", cut=0)

        # Setze Titel und Labels
        plt.ylabel(r"Cumulative Regret", fontsize=14, labelpad=15)

        # Anpassungen für das Layout
        plt.xticks(rotation=45, ha="right", fontsize=12)  # Drehe die x-Achsen-Beschriftung für bessere Lesbarkeit
        plt.yticks(fontsize=12)
        plt.tight_layout()

        # Weitere visuelle Verbesserungen
        # Füge ein Grid hinzu, um den Plot klarer zu machen
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Speichere den Plot und zeige ihn an
        plt.savefig("./data/linear_violin_bern.pdf", dpi=300)  # Höhere Auflösung für bessere Qualität
        plt.show()

    def plot_function(self):
        # Kontext-Vektoren generieren (1000 Samples, 3 Dimensionen)
        num_samples = 1000


        # Generiere die Gewichte für die lineare Belohnungsfunktion
        true_cost_weights = generate_true_weights(self.num_arms, self.num_features, seed=42, method="reward")


        # Kontext-Vektoren generieren (1000 Samples, 2 Dimensionen für Heatmap)
        x_vals = np.linspace(0, 1, num_samples)
        y_vals = np.linspace(0, 1, num_samples)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Berechne die Belohnungswerte für jedes (x, y)-Paar
        reward_values = np.zeros((num_samples, num_samples))  # Die Form (num_samples, num_samples)

        for i in range(num_samples):
            for j in range(num_samples):
                context = np.array([X[i, j], Y[i, j], 0.5])  # Setze die dritte Dimension auf 0.5
                reward_values[i, j] = polynomial_reward2(context, true_cost_weights[0])[0]  # Verwende den ersten Arm

        # Erstelle die Heatmap
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(reward_values, cmap="viridis")
        ax.invert_yaxis()

        # Setze die X- und Y-Tick-Beschriftungen explizit
        ticks = [0, 0.25, 0.5, 0.75, 1]
        ax.set_xticks(np.linspace(0, num_samples - 1, len(ticks)))  # Position der Ticks
        ax.set_yticks(np.linspace(0, num_samples - 1, len(ticks)))  # Position der Ticks
        ax.set_xticklabels(ticks)  # Setze die Beschriftungen
        ax.set_yticklabels(ticks)  # Setze die Beschriftungen

        plt.title("Heatmap der Funktion mit heißeren Bereichen für hohe Ausgaben")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


# Starte dasd Experiment
reward_type = ['linear', 'nonlinear']
cdc = True #context dependent cost
adversary = False
runner = Runner(20,reward_type[0], cdc, reward_type[0], 1, adversary=adversary)
start_time = time.time()  # Startzeitpunkt
runner.run_experiment()
#runner.plot_function()

#runner.run_experiment()
end_time = time.time()  # Endzeitpunkt
execution_time = end_time - start_time  # Zeitdifferenz in Sekunden
if __name__ == '__main__':
    print(f"Die Methode dauerte {execution_time:.4f} Sekunden.")
