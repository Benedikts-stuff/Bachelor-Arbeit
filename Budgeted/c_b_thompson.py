import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

#Mit Beta verteilung wie im budgeted Thompson ansatz

class ThompsonSamplingContextualBandit:
    def __init__(self, d, v, n_arms, contexts, true_weights, cost, budget, logger, repetition, seed, cost_kind):
        """
        d: Dimension der Kontextvektoren
        v: Varianzparameter für die Normalverteilung
        n_arms: Anzahl der Arme
        n_rounds: Anzahl der Runden
        contexts: Kontextdaten für jede Runde
        true_weights: Wahre Gewichtungen für jeden Arm
        cost: Kosten pro Arm
        budget: Gesamtbudget
        """
        np.random.seed(seed)
        self.logger = logger
        self.repetition = repetition
        self.n_features = d
        self.variance = v
        self.n_arms = n_arms
        self.contexts = contexts
        self.true_weights = true_weights
        self.cost = np.array(cost)
        self.budget = budget
        self.og_budget = budget
        self.max_cost = np.max(self.cost)
        self.arm_counts = np.zeros(self.n_arms)
        self.gamma = 0.00000001
        self.cum = np.zeros(self.n_arms)
        self.cost_kind = cost_kind

        self.empirical_cost_means = np.random.rand(self.n_arms)
        self.summed_regret = 0
        self.alpha =np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

        # Initialisiere die Parameter für jeden Arm
        self.B = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]

        # Historie der beobachteten Belohnungen und optimalen Belohnungen
        self.observed_reward_history = []
        self.optimal_reward = []

        # Berechne die tatsächlichen Belohnungen für jede Runde und jeden Arm
        self.actual_reward_history = self.compute_actual_rewards()

    def compute_actual_rewards(self):
        """
        Berechnet die tatsächlichen Belohnungen basierend auf den wahren Gewichtungen und den Kontexten.
        """
        return self.contexts.dot(self.true_weights.T)

    def sample_mu(self, round):
        """
        Samplet Schätzungen der Gewichte (mu) aus einer Multinormalverteilung.
        """
        return np.array([
            np.random.multivariate_normal(self.mu_hat[arm], 0.2**2 * np.linalg.inv(self.B[arm]))
            for arm in range(self.n_arms)
        ])


    def select_arm(self, sampled_mu, context):
        """
        Wählt den Arm mit dem höchsten erwarteten Belohnungs-Kosten-Verhältnis.
        """
        sampled_cost = np.array([np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)])
        expected_rewards = np.array([np.dot(sampled_mu[arm], context) for arm in range(self.n_arms)])
        return np.argmax(expected_rewards / (sampled_cost + self.gamma))

    def calculate_optimal_reward(self, context):
        """
        Berechnet die optimale Belohnung für den gegebenen Kontext.
        """
        rewards = np.array([np.dot(self.true_weights[arm], context) for arm in range(self.n_arms)])
        return [np.max(np.array(rewards / self.cost)),np.argmax(np.array(rewards / self.cost))]

    def update_beliefs(self, reward, chosen_arm, context):
        """
        Aktualisiert die Schätzungen (mu_hat) und die Kovarianzmatrix (B) für den gewählten Arm.
        """
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.mu_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])
        self.budget -= self.cost[chosen_arm]

        cost_t = np.random.binomial(1, self.cost[chosen_arm])
        #self.empirical_cost_means[chosen_arm] = self.cum[chosen_arm] / (self.arm_counts[chosen_arm] + 1)
        #update beta dist. params
        self.alpha[chosen_arm] += cost_t
        self.beta[chosen_arm] += 1 - cost_t
        self.arm_counts[chosen_arm] += 1

    def run(self):
        """
        Führt den Thompson-Sampling-Algorithmus aus, bis das Budget aufgebraucht ist.
        """
        i = 0
        while self.budget > self.max_cost:
            context = self.contexts[i]
            sampled_mu = self.sample_mu(i)
            chosen_arm = self.select_arm(sampled_mu, context)
            actual_reward = np.dot(self.true_weights[chosen_arm], context)

            self.update_beliefs(actual_reward, chosen_arm, context)
            observed_reward = actual_reward / self.cost[chosen_arm]
            self.observed_reward_history.append(observed_reward)
            x = self.calculate_optimal_reward(context)
            self.optimal_reward.append(x[0])

            self.summed_regret += x[0] - observed_reward

            self.logger.track_rep(self.repetition)
            self.logger.track_approach(1)
            self.logger.track_round(i)
            self.logger.track_regret(self.summed_regret)
            self.logger.track_normalized_budget((self.og_budget - self.budget)/ self.og_budget)
            self.logger.track_spent_budget(self.og_budget - self.budget)
            self.logger.finalize_round()
            i += 1

        print('muHat TS', self.mu_hat)


