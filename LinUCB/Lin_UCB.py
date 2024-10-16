from array import array

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tempp import Get_linear_model

sampler = Get_linear_model(1)
th2 = sampler.get_model()

df = sampler.test

grouped2= df.groupby(['age', 'gender', 'interest2', 'interest3' ])
context_counts = grouped2.size().reset_index(name='group_size')
context_probs = context_counts['group_size'] / len(df)

probs = context_probs.to_numpy()



# wenn man model.predcit verwenden will um die güte zu zeigen muss man noch model.intercept_ mit in den feature vektor
# einbinden


np.random.seed(0)
straight = np.array([0.1801471369890952,
0.16501501497368395,
0.12853849875116444
])
n_a = df['campaign_id'].nunique()  # Anzahl der eindeutigen Kampagnen
k= 5 # number of features
n =100000

#D = np.random.random((n, k)) - 0.5

x = sampler.get_model()
th = np.array([np.concatenate(([model.intercept_], model.coef_)) for model in x])

features =sampler.sample_contexts()
features = np.array([np.concatenate(([1], arr)) for arr in features])
d = features.shape[1]
D = np.zeros((n, k))
for i in range(n):
    D[i] = features[i]

#D = D - 0.5
#th = np.random.random((n_a, k)) - 0.5
eps = 0.2
choices = np.zeros(n, dtype=int)
rewards = np.zeros(n)
rewards2 = np.zeros(n)
explore = np.zeros(n)
norms = np.zeros(n)
#weights = [model.coef_ for model in th]
arm_count = np.zeros(n_a)
b = np.zeros((n_a,k))
A = np.zeros((n_a, k, k))
for a in range(0, n_a):
    A[a] = np.identity(k)
th_hat = np.zeros((n_a, k))  # our temporary feature vectors, our best current guesses
p = np.zeros(n_a)
alph = 0.2
#P = np.zeros((n, n_a))
P = D.dot(th.T)
# LINUCB, usign disjoint model
# This is all from Algorithm 1, p 664, "A contextual bandit appraoch..." Li, Langford
for i in range(0, n):
    x_i = D[i] #features[idxs[i]] - 0.5 # the current context vector
    #P[i] = [model.predict(x_i.reshape(1, -1))[0] for model in th]
    for a in range(0, n_a):
        A_inv = np.linalg.inv(A[a])  # we use it twice so cache it.
        th_hat[a] = A_inv.dot(b[a])  # Line 5
        #print(a, ': ', th_hat[a])
        ta = x_i.dot(A_inv).dot(x_i)  # how informative is this?
        a_upper_ci = alph * np.sqrt(ta)  # upper part of variance interval

        a_mean = th_hat[a].dot(x_i)  # current estimate of mean
        print(a, ': ', a_mean)
        p[a] = a_mean + a_upper_ci
    norms[i] = np.linalg.norm(th_hat - th, 'fro')  # diagnostic, are we converging?
    # Let's hnot be biased with tiebraks, but add in some random noise
    #p = p + (np.random.random(len(p)) * 0.000001)
    choices[i] = p.argmax()  # choose the highest, line 11
    arm_count[choices[i]] += 1

    rewards[i] = th[choices[i]].dot(x_i)
    # See what kind of result we get
    #rewards[i] = straight[choices[i]]
    #model_i = th2[choices[i]]
    #rewards[i] = model_i.predict(x_i.reshape(1, -1))[0]
    # update the input vector
    A[choices[i]] += np.outer(x_i, x_i)
    b[choices[i]] += rewards[i] * x_i

plt.figure(1, figsize=(10, 5))
plt.subplot(121)
plt.plot(norms)
plt.title("Frobeninus norm of estimated theta vs actual")
plt.show()

regret = (P.max(axis=1) - rewards)
plt.subplot(122)
plt.plot(regret.cumsum(), label='linear model')
plt.title("Cumulative regret")
plt.legend()
plt.show()

asdsd = rewards.cumsum()
print('cumu reward',asdsd[n-1])

plt.subplot(122)
plt.plot(rewards.cumsum(), label= 'linear_model')
plt.title("Cumulative reward")
plt.legend()
plt.show()