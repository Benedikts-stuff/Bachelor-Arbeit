import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from concurrent.futures import as_completed
import warnings
warnings.filterwarnings("ignore")
import colormaps as cmaps
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
    def __init__(self, n_rounds, reward_typ,c,cost_type,cost_index, alpha,  num_arms=3, num_features=3, budget=10000):
        self.iterations = n_rounds
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
        self.alpha = alpha

    def run_experiment(self):
        all_data = {name: [] for name in ['LinearEpsilonGreedy', 'LinearEpsilonGreedy2']} #['C-UCB', 'C-ThompsonSampling', 'OmegaUCB','NeuralOmegaUCB', 'LinUCB', 'EpsilonGreedy', 'ThompsonSampling', 'GPUCB', 'GPTS']}   #{name: [] for name in ['EpsilonGreedy', 'ThompsonSampling', 'LinUCB', 'OmegaUCB']}
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
                            executor.submit(self.run_bandit, bandit_type, true_weights, true_cost, i, true_cost_weights, self.cost_kind[self.cost_index], self.reward_type, self.alpha)
                        )

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Futures"):
                bandit_type, result = future.result()
                all_data[bandit_type].append(result)

        if __name__ == '__main__':
            plot_data = {name: self.interp_plot(data) for name, data in all_data.items()}
            norm_reg = 0
            budg_reg = 0
            for key, value in plot_data.items():
                if key == 'LinearEpsilonGreedy':
                    norm_reg = value[REGRET].max()
                else:
                    budg_reg = value[REGRET].max()
            return norm_reg, budg_reg


    def run_bandit(self, bandit_type, true_weights, true_cost, seed, true_cost_weights, cost_kind, reward_type, alpha):
        np.random.seed(seed)
        bandit = None
        if self.cdc:
            if reward_type == 'linear' and self.cost_type == 'linear':
                bandit = BanditFactory.create(bandit_type, self, true_weights, true_cost, seed, true_cost_weights, cost_kind, linear_reward, linear_cost)
            elif reward_type == 'polynomial' and self.cost_type == 'linear':
                bandit = BanditFactory.create(bandit_type, self, true_weights, true_cost, seed, true_cost_weights,
                                              cost_kind, polynomial_reward, linear_cost)
            elif reward_type == 'linear' and self.cost_type == 'polynomial':
                bandit = BanditFactory.create(bandit_type, self, true_weights, true_cost, seed, true_cost_weights,
                                              cost_kind, linear_reward, polynomial_cost)
            elif reward_type == 'polynomial' and self.cost_type == 'polynomial':
                bandit = BanditFactory.create(bandit_type, self, true_weights, true_cost, seed, true_cost_weights,
                                              cost_kind, polynomial_reward, polynomial_cost)

        else:
            bandit = BanditFactory.create(bandit_type, self, true_weights, true_cost, seed, true_cost_weights,
                                          cost_kind, alpha)

        bandit.run()
        return bandit_type, bandit.logger.get_dataframe()

    def interp_plot(self, dfs, x_col=NORMALIZED_SPENT_BUDGET, y_col=REGRET):
        axis_list = [
            df[[x_col, y_col]].sort_values(by=x_col).drop_duplicates(x_col).to_numpy() for df in dfs
        ]
        new_axis_xs = np.linspace(0, 1, 100)
        new_axis_ys = [np.interp(new_axis_xs, axis[:, 0], axis[:, 1]) for axis in axis_list]
        midy = np.mean(new_axis_ys, axis=0)
        return pd.DataFrame({x_col: new_axis_xs, y_col: midy})

    def plot_budget_normalised_regret(self, plot_data):
        plt.figure(figsize=(10, 6))
        styles = {
            'LinearEpsilonGreedy': ('#1f77b4', '-'),  # Blau (durchgehend)
            'LinearEpsilonGreedy2': ('#d62728', '-'),  # Blau (durchgehend)
            'LinearThompsonSampling': ('#ff7f0e', '--'),  # Orange (gestrichelt)
            'LinUCB': ('#2ca02c', '-.'),  # Grün (gepunktet-gestrichelt)
            'LinOmegaUCB': ('#9467bd', '-'),  # Rot (gepunktet)
            'NeuralOmegaUCB': ('#d62728', ':'),  # Lila (durchgehend)
            'GPUCB': ('#8c564b', '--'),  # Braun (gestrichelt)
            'GPWUCB': ('#000000', '--'),  # schwarz (gestrichelt)
            'GPTS': ('#e377c2', '-.'),  # Rosa (gepunktet-gestrichelt)
            'C-LinUCB': ('#7f7f7f', ':'),  # Grau (gepunktet)
            'C-LinearThompsonSampling': ('#bcbd22', '--'),  # Gelbgrün (gestrichelt)
            'LinOmegaUCB_CDC': ('#bcbd22', '--'),  # Gelbgrün (gestrichelt)
            'LinearEpsilonGreedy_CDC': ('#1f77b4', '-'),  # Blau (durchgehend)
            'RandomBandit': ('#2ca02c', '-.'),  # Grün (gepunktet-gestrichelt)
            'LinUCB_CDC': ('#8c564b', '--'),  # Braun (gestrichelt)
            'NeuralOmegaUCB_CDC': ('#bcbd22', ':'),
            'Budget_CB' :('#000000', '--'),
            'NeuralGreedy': ('#9467bd', '--'),  # Gelbgrün (gestrichelt)
        }

        y_lim = 0
        for name, df in plot_data.items():
            color, style = styles[name]
            sns.lineplot(x=NORMALIZED_SPENT_BUDGET, y=REGRET, data=df, label=name, color=color, linestyle=style)
            y_lim = max(y_lim, df[REGRET].max())

        plt.xlabel("Normalized Spent Budget", fontsize=14)
        plt.ylabel("Cumulative Regret", fontsize=14)
        plt.ylim(0, y_lim)
        plt.legend(fontsize=12)
        plt.title("Regret Comparison of Different Combined Bandit Approaches", fontsize=14)
        plt.savefig("nonLinear_with_beta_Cost")
        plt.show()

        #plot_data.to_csv('plot_data.csv', index=False)


if __name__ == '__main__':
    # Starte dasd Experiment
    reward_type = ['linear', 'nonlinear']
    cdc = False #context dependent cost
    alphas = list(range(1,20)) #+ list(range(80, 100))
    regret_norm = []
    regret_budg = []
    count  = 0
    for alpha in alphas:
        print("alpha count", count)
        runner = Runner(10,reward_type[0], cdc, reward_type[0], 1, alpha)
        norm_reg, budg_reg = runner.run_experiment()
        regret_norm.append(norm_reg)
        regret_budg.append(budg_reg)
        count+= 1

    # Erstelle den Plot
    plt.figure(figsize=(8, 6))
    # Plotten der beiden Linien mit 'viridis' Farbschema
    plt.plot(alphas, regret_norm, label="Normal", color=plt.cm.viridis(0.2))  # Erste Linie
    plt.plot(alphas, regret_budg, label="Budgeted", color=plt.cm.viridis(0.8))  # Zweite Linie

    # Achsen und Titel
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Regret')
    plt.title('Comparison: round-based vs budget-based exploration')

    # Legende
    plt.legend()
    #plt.savefig('./data/budgeted_vs_normal_expl.pdf')
    # Anzeige des Plots
    plt.show()