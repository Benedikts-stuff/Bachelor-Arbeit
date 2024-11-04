import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)

class LinUCB:
    def __init__(self, models, alpha, n_rounds, context):
        self.n_a = 3
        self.k = 3
        self.n = n_rounds
        self.th = models
        self.features = context
        self.d = self.features.shape[1]
        self.D = context

        self.choices = np.zeros(self.n, dtype=int)
        self.rewards = np.zeros(self.n)
        self.explore = np.zeros(self.n)
        self.norms = np.zeros(self.n)
        self.arm_count = np.zeros(self.n_a)
        self.b = np.zeros((self.n_a, self.k))
        self.A = np.zeros((self.n_a, self.k, self.k))
        for a in range(0, self.n_a):
            self.A[a] = np.identity(self.k)

        self.lamda = 1
        self.m_2 = np.linalg.norm(self.th, axis=1)
        self.th_hat = np.zeros((self.n_a, self.k))
        self.p = np.zeros(self.n_a)
        self.alpha = alpha
        self.P = self.D.dot(self.th.T)
    def run_LinUCB(self):
        for i in range(0, self.n):
            x_i = self.D[i]

            for a in range(0, self.n_a):
                A_inv = np.linalg.inv(self.A[a])
                self.th_hat[a] = A_inv.dot(self.b[a])
                ta = x_i.dot(A_inv).dot(x_i)
                a_upper_ci = self.alpha * np.sqrt(ta)
                a_mean = self.th_hat[a].dot(x_i)
                self.p[a] = a_mean + a_upper_ci
            self.norms[i] = np.linalg.norm(self.th_hat - self.th, 'fro')

            self.p = self.p + (np.random.random(len(self.p)) * 0.000001)
            self.choices[i] = self.p.argmax()
            self.arm_count[self.choices[i]] += 1
            self.rewards[i] = self.th[self.choices[i]].dot(x_i)

            self.A[self.choices[i]] += np.outer(x_i, x_i)
            self.b[self.choices[i]] += self.rewards[i] * x_i


# Set parameters
num_features = 3
num_rounds = 2500
true_weights = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.2], [0.5, 0.2, 0.3]])
context = np.random.rand(num_rounds, num_features)
bandit_lin_ucb = LinUCB(true_weights, 0.5, num_rounds, context)
bandit_lin_ucb.run_LinUCB()


#LinUCB
plt.figure(1, figsize=(10, 5))
plt.subplot(121)
#plt.plot(bandit_weak.norms, label='Weak')
#plt.plot(bandit_medium.norms, label='Medium')
plt.plot(bandit_lin_ucb.norms, label='Strong')
#plt.plot(bandit_no.norms, label='no')
plt.title("Frobeninus norm of estimated theta vs actual")
plt.legend()
plt.show()

regret_strong = (bandit_lin_ucb.P.max(axis=1) - bandit_lin_ucb.rewards)
plt.subplot(122)
plt.plot(regret_strong.cumsum(), label='strong')
plt.title("Cumulative regret")
plt.legend()
plt.show()