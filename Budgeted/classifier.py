import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

class ContextualBandit:
    def __init__(self, num_arms, context_dim, context, train, seed):
        np.random.seed(seed)
        # Anzahl der Arme und Dimension der Kontexte
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.context = context

        # Klassifikator für jeden Arm (logistische Regression)
        self.classifiers = [LogisticRegression() for _ in range(num_arms)]

        # Speichert die gesammelten Daten: (Kontext, Belohnung) für jedes Arm
        self.contexts = {arm: [] for arm in range(num_arms)}
        self.rewards = {arm: [] for arm in range(num_arms)}
        self.optimal = []
        self.observed = []

    # Wahre Belohnungsfunktion f*
    def true_reward_function(self, context, arm_id):
        if arm_id == 0:
            # return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
            return 0.4 * context[0] + 0.2 * context[1] + 0.4 * context[2]
        elif arm_id == 1:
            return 0.1 * context[0] + 0.5 * context[1] + 0.4 * context[2]
            # return np.tanh(0.1 * context[0] + 0.8 * context[1] + 0.1 * context[2])
        elif arm_id == 2:
            return 0.2 * context[0] + 0.3 * context[1] + 0.5 * context[2]
            # return np.tanh(0.3 * context[0] + 0.3 * context[1] + 0.6 * context[2])


    def choose_arm(self, context):
        """
        Wählt einen Arm basierend auf der höchsten Wahrscheinlichkeit des Klassifikators.
        Hier könnte man auch ein Exploration-Exploitation-Verfahren einbauen.
        """
        epsilon = 0.1
        if np.random.rand() < epsilon:
            return np.random.choice(self.num_arms)

        probs = []
        for arm in range(self.num_arms):
            # Vorhersage für den Kontext des aktuellen Arms
            if 0 in self.rewards[arm] and 1 in self.rewards[arm]:
                prob = self.classifiers[arm].predict_proba([context])[0, 1]  # Wahrscheinlichkeit für Belohnung=1
            else:

                prob = np.random.rand()
            probs.append(prob)

        print("predicted probs: ", probs)

        # Wähle den Arm mit der höchsten Wahrscheinlichkeit
        chosen_arm = np.argmax(probs)
        return chosen_arm

    def update(self, chosen_arm, context, reward):
        """
        Aktualisiert den Klassifikator des gewählten Arms mit dem Kontext und der Belohnung.
        """
        # Speichere den Kontext und die Belohnung
        self.contexts[chosen_arm].append(context)
        self.rewards[chosen_arm].append(reward)

        # Trainiere den Klassifikator des gewählten Arms mit den gesammelten Daten
        if 0 in self.rewards[chosen_arm] and 1 in self.rewards[chosen_arm]:
            print("test")
            self.classifiers[chosen_arm].fit(self.contexts[chosen_arm], self.rewards[chosen_arm])

    def train(self, contexts, rewards):
        """
        Trainiert den Banditen mit den gegebenen Kontexten und Belohnungen für eine festgelegte Anzahl von Episoden.
        """
        for context, reward in zip(contexts, rewards):
            chosen_arm = self.choose_arm(context)
            self.update(chosen_arm, context, reward)

    def run(self):
        for t in range(25000):
            context = self.context[t]
            chose_arm = self.choose_arm(context)
            p = [self.true_reward_function(context, arm) for arm in range(self.num_arms)]
            print("actual probs: ", p)
            reward = np.random.binomial(1, p[chose_arm])
            self.update(chose_arm, context, reward)
            self.optimal.append( np.random.binomial(1, np.max(p)))
            self.observed.append(reward)
            print(t)



# Beispiel-Daten: 3 Arme, 2 Merkmale im Kontext
num_arms = 3
context_dim = 3
regret = np.zeros(25000)
for i in range(10):
    np.random.seed(i)
    # Zufällige Kontextdaten und Belohnungen für das Training
    contexts = np.random.rand(25000, context_dim)  # 1000 Kontexte, 2 Merkmale
    train = np.random.rand(500, context_dim)
    bandit = ContextualBandit(num_arms, context_dim, contexts, train, i)

    bandit.run()
    regret = np.add(regret, np.array(bandit.optimal) - np.array(bandit.observed))

regret = regret / 20
plt.subplot(122)
plt.plot(regret.cumsum(), label='ts model')
plt.title("Cumulative regret")
plt.legend()
plt.show()
