import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
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
n_a = df['campaign_id'].nunique()
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
th_hat = np.zeros((n_a, k))
p = np.zeros(n_a)
alph = 0.2
#P = np.zeros((n, n_a))
P = D.dot(th.T)
# LINUCB, usign disjoint model
# This is all from Algorithm 1, p 664, "A contextual bandit appraoch..." Li, Langford
for i in range(0, n):
    x_i = D[i]

    for a in range(0, n_a):
        A_inv = np.linalg.inv(A[a])
        th_hat[a] = A_inv.dot(b[a])
        #print(a, ': ', th_hat[a])
        ta = x_i.dot(A_inv).dot(x_i)
        a_upper_ci = alph * np.sqrt(ta)

        a_mean = th_hat[a].dot(x_i)
        print(a, ': ', a_mean)
        p[a] = a_mean + a_upper_ci
    norms[i] = np.linalg.norm(th_hat - th, 'fro')

    #p = p + (np.random.random(len(p)) * 0.000001)
    choices[i] = p.argmax()
    arm_count[choices[i]] += 1

    rewards[i] = th[choices[i]].dot(x_i)
    #rewards[i] = straight[choices[i]]
    #model_i = th2[choices[i]]
    #rewards[i] = model_i.predict(x_i.reshape(1, -1))[0]
    A[choices[i]] += np.outer(x_i, x_i)
    b[choices[i]] += rewards[i] * x_i




# Plot Frobenius Norm
sns.set(style="whitegrid", context="talk")
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sourcesanspro'
plt.rcParams['text.latex.preamble'] = r'\usepackage{libertine}\usepackage[libertine]{newtxmath}\RequirePackage[scaled=.92]{sourcesanspro}'

plt.figure(figsize=(12, 6))
sns.lineplot(x=range(len(norms)), y=norms, color="b", label=r'Frobenius Norm')

plt.xlabel(r'Iterations')
plt.ylabel(r'Frobenius Norm')
plt.legend()
plt.savefig('Frobenius_plot.pdf', format='pdf')  # Vektorgrafik im SVG-Format
plt.show()