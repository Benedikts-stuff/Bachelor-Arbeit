from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pdb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler



class Test_corr():
    def __init__(self):
        self.scaler = StandardScaler()
        self.data = pd.DataFrame()
        self.test = pd.DataFrame()

        self.scaler2 = MinMaxScaler(feature_range=(0, 1))
        self.means = np.zeros(3)


    def generate_data(self, correlation_level, num_samples=10000):
        np.random.seed(42)
        age = np.random.randint(18, 65, size=num_samples)

        gender = np.random.randint(0, 2, size=num_samples)
        campaign_id = np.random.randint(0, 3, size=num_samples)

        interest = np.random.randint(0,10, size=num_samples)
        self.data = pd.DataFrame({
            'campaign_id': campaign_id,
            'age': age,
            'gender': gender,
            'interest': interest
        })

        #if correlation_level == 'weak':
            #self.data['ctr'] = 0.2 * self.data['age'] + 0.2 * self.data['gender'] + 0.2 * self.data['interest'] + np.random.normal(0, 0.1, num_samples)
        #if correlation_level == 'medium':
            #self.data['ctr'] = 0.5 * self.data['age'] + 0.5 * self.data['gender'] + 0.5 * self.data['interest'] + np.random.normal(0, 0.1, num_samples)
        #if correlation_level == 'strong':
            #self.data['ctr'] = 0.8 * self.data['age'] + 0.8 * self.data['gender'] + 0.8 * self.data['interest'] + np.random.normal(0, 0.1, num_samples)


        data_916 = self.data[self.data['campaign_id'] == 0]
        data_936 = self.data[self.data['campaign_id'] == 1]
        data_1178 = self.data[self.data['campaign_id'] == 2]

        data_916 = pd.DataFrame(self.scaler.fit_transform(data_916), columns = self.data.columns)
        data_936 = pd.DataFrame(self.scaler.fit_transform(data_936), columns = self.data.columns)
        data_1178 = pd.DataFrame(self.scaler.fit_transform(data_1178), columns = self.data.columns)

        if correlation_level == 'weak':
            data_916['ctr'] = 0.6 * data_916['age'] + 0.3 * data_916['gender'] + 0.3 * data_916['interest'] #+ np.random.normal(0.0, 0.5, len(data_916))
            data_936['ctr'] = 0.3 * data_936['age'] + 0.6 * data_936['gender'] + 0.3 * data_936['interest'] #+ np.random.normal(0.0, 0.5, len(data_936))
            data_1178['ctr'] = 0.3 * data_1178['age'] + 0.3 * data_1178['gender'] + 0.6 * data_1178['interest'] #+ np.random.normal(0.0, 0.5, len(data_1178))

        if correlation_level == 'medium':
            data_916['ctr'] = 0.6 * data_916['age'] + 0.3 * data_916['gender'] + 0.3 * data_916['interest'] #+ np.random.normal(0.0, 0.2, len(data_916))
            data_936['ctr'] = 0.3 * data_936['age'] + 0.6 * data_936['gender'] + 0.3 * data_936['interest'] #+ np.random.normal(0.0, 0.2, len(data_936))
            data_1178['ctr'] = 0.3 * data_1178['age'] + 0.3 * data_1178['gender'] + 0.6 * data_1178['interest'] #+np.random.normal(0.0, 0.2, len(data_1178))


        if correlation_level == 'strong':
            data_916['ctr'] = 0.6 * data_916['age'] + 0.3 * data_916['gender'] + 0.3 * data_916['interest']# + np.random.normal(0, 0.05, len(data_916))
            data_936['ctr'] = (0.3 * data_936['age'] + 0.6* data_936['gender'] + 0.3 * data_936['interest'])  #+ np.random.normal(0, 0.05, len(data_936))
            data_1178['ctr'] = 0.3 * data_1178['age'] + 0.3 * data_1178['gender'] + 0.6 * data_1178['interest'] #+ np.random.normal(0, 0.05, len(data_1178))

        if correlation_level == 'no':
            data_916['ctr'] = np.random.normal(0.5, 0.1, len(data_916))  + np.random.normal(0, 0.1, len(data_916))
            data_936['ctr'] = np.random.normal(0.5, 0.1, len(data_936))  + np.random.normal(0, 0.1, len(data_936))
            data_1178['ctr'] = np.random.normal(0.5, 0.1, len(data_1178))  + np.random.normal(0, 0.1, len(data_1178))


        data_916['ctr'] = self.scaler2.fit_transform(data_916[['ctr']])
        data_936['ctr'] = self.scaler2.fit_transform(data_936[['ctr']])
        data_1178['ctr'] = self.scaler2.fit_transform(data_1178[['ctr']])

        #data_916['ctr'] = self.scaler2.fit_transform(data_916[['ctr']])
        #data_936['ctr'] = self.scaler2.fit_transform(data_936[['ctr']])
        #data_1178['ctr'] = self.scaler2.fit_transform(data_1178[['ctr']])



        #self.data.loc[self.data['campaign_id'] == 1, 'ctr'] *= 1.1

        #data_936.loc[:, 'ctr'] = data_936['ctr'] * 1.2

        self.means = np.array([data_916['ctr'].mean(), data_936['ctr'].mean(), data_1178['ctr'].mean()] )


        data_916 = data_916.groupby(['age', 'gender', 'interest'])['ctr'].mean().reset_index()

        data_936 = data_936.groupby(['age', 'gender', 'interest'])['ctr'].mean().reset_index()

        data_1178 = data_1178.groupby(['age', 'gender', 'interest'])['ctr'].mean().reset_index()

        # data['ctr'] = data['clicks'] / data['impressions']


        data_916 = data_916.replace([None, float('inf'), float('-inf')], 0)
        data_936 = data_936.replace([None, float('inf'), float('-inf')], 0)
        data_1178 = data_1178.replace([None, float('inf'), float('-inf')], 0)

        data_shuffled_916 = data_916.sample(frac=1, random_state=42).reset_index()
        data_shuffled_936 = data_936.sample(frac=1, random_state=42).reset_index()
        data_shuffled_1178 = data_1178.sample(frac=1, random_state=42).reset_index()

        split_index_916 = int(0.8 * len(data_shuffled_916))
        split_index_936 = int(0.8 * len(data_shuffled_936))
        split_index_1178 = int(0.8 * len(data_shuffled_1178))

        train_data_916 = data_shuffled_916[:split_index_916]
        test_data_916 = data_shuffled_916[split_index_916:]

        train_data_936 = data_shuffled_936[:split_index_936]
        test_data_936 = data_shuffled_936[split_index_936:]

        train_data_1178 = data_shuffled_1178[:split_index_1178]
        test_data_1178 = data_shuffled_1178[split_index_1178:]


        X_train_916 = train_data_916[['age', 'gender', 'interest']]
        y_train_916 = train_data_916['ctr']

        X_test_916 = test_data_916[['age', 'gender', 'interest']]
        y_test_916 = test_data_916['ctr']

        X_train_scaled_916 = X_train_916 #self.scaler.transform(X_train_916)
        X_test_scaled_916 =X_test_916 # self.scaler.transform(X_test_916)


        model_916 = LinearRegression()
        model_916.fit(X_train_scaled_916, y_train_916)


        y_pred_916 = model_916.predict(X_test_916)


        X_train_936 = train_data_936[['age', 'gender', 'interest']]
        y_train_936 = train_data_936['ctr']

        X_test_936 = test_data_936[['age', 'gender', 'interest']]
        y_test_936 = test_data_936['ctr']

        X_train_scaled_936 = X_train_936 # self.scaler.transform(X_train_936)
        X_test_scaled_936 = X_test_936 #self.scaler.transform(X_test_936)


        model_936 = LinearRegression()
        model_936.fit(X_train_scaled_936, y_train_936)


        y_pred_936 = model_936.predict(X_test_936)


        X_train_1178 = train_data_1178[['age', 'gender', 'interest']]
        y_train_1178 = train_data_1178['ctr']

        X_test_1178 = test_data_1178[['age', 'gender', 'interest']]
        y_test_1178 = test_data_1178['ctr']

        X_train_scaled_1178 = X_train_1178#self.scaler.transform(X_train_1178)
        X_test_scaled_1178 = X_test_1178 #self.scaler.transform(X_test_1178)


        model_1178 = LinearRegression()
        model_1178.fit(X_train_scaled_1178, y_train_1178)

        y_pred_1178 = model_1178.predict(X_test_1178)

        self.test = pd.concat([test_data_916, test_data_936, test_data_1178], ignore_index=True)

        mse_916 = mean_squared_error(y_test_916, y_pred_916)
        mse_936 = mean_squared_error(y_test_936, y_pred_936)
        mse_1178 = mean_squared_error(y_test_1178, y_pred_1178)
        # r2 = r2_score(y_test, y_pred)
        r2_916 = model_916.score(X_test_scaled_916, y_test_916)
        r2_936 = model_936.score(X_test_scaled_936, y_test_936)
        r2_1178 = model_1178.score(X_test_scaled_1178, y_test_1178)

        # Ausgabe der Ergebnisse
        print(correlation_level, "Mean Squared Error (MSE):", mse_916)
        print(correlation_level,"Mean Squared Error (MSE):", mse_936)
        print(correlation_level, "Mean Squared Error (MSE):", mse_1178)
        #print("R²-Wert:", r2_916)
        #print("R²-Wert:", r2_936)
        #print("R²-Wert:", r2_1178)


        return [model_916, model_936, model_1178]

    def sample_contexts(self):


        grouped_context = self.test.groupby(['age', 'gender', 'interest'])['ctr'].mean().reset_index()


        grouped_context['group_size'] = self.test.groupby(['age', 'gender', 'interest']).size().values


        grouped_context = grouped_context.replace([None, float('inf'), float('-inf')], 0)


        context_probs_df = grouped_context['group_size'] / len(self.test)

        #contexts = self.scaler.transform(grouped_context[['age', 'gender', 'interest']])
        contexts = grouped_context[['age', 'gender', 'interest']]
        probs = context_probs_df.to_numpy()


        chosen_indices = np.random.choice(len(contexts), size=1000000, p=probs)
        #chosen_contexts = contexts[chosen_indices]
        chosen_contexts = contexts.iloc[chosen_indices].values

        return chosen_contexts


class LinUCB:
    def __init__(self, models, samples, alpha, correlation_strength):
        self.n_a = 3
        self.correlation_strength = correlation_strength
        self.k = 4
        self.n = 1000000
        #self.noise = np.array([np.random.normal(0.1, noise, self.n)])
        self.models = [model for sublist in models for model in sublist]
        self.th = np.array([np.concatenate(([model.intercept_], model.coef_)) for model in self.models])
        self.features = np.array([np.concatenate(([1], arr)) for arr in samples])
        self.d = self.features.shape[1]
        self.D = np.zeros((self.n, self.k))
        for i in range(self.n):
            self.D[i] = self.features[i]

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
        self.alph = alpha
        self.P = self.D.dot(self.th.T)
        self.noise = np.random.normal(0.0, 0.1, self.P.shape)
        self.P = correlation_strength * self.P + (1 - correlation_strength) * self.noise
    def run_LinUCB(self):
        for i in range(0, self.n):
            x_i = self.D[i]

            for a in range(0, self.n_a):
                A_inv = np.linalg.inv(self.A[a])
                self.th_hat[a] = A_inv.dot(self.b[a])
                # print(a, ': ', th_hat[a])
                ta = x_i.dot(A_inv).dot(x_i)
                #a_upper_ci = (1 + np.sqrt(np.log(2 * (i+ 1)) / 2)) * np.sqrt(ta)
                L = np.linalg.norm(self.th_hat[a])
                s = (np.sqrt(self.lamda) * self.m_2[a]) #sqrt(lamda) * m2
                b = (2 * np.log(i+1)) # 2 log(n)
                c = (self.d * self.lamda + (i+ 1) * (L**2)) # d* log( (d * lamda + n * Lˆ2))
                d = (self.d * self.lamda) # d * lamda
                beta = (1 + (s + np.sqrt(b + ( self.d * np.log(c /d)))))
                a_upper_ci =   beta * np.sqrt(ta)
                #pdb.set_trace()
                a_mean = self.th_hat[a].dot(x_i)
                #print(a, ': ' , a_mean)
                self.p[a] = a_mean + a_upper_ci
            self.norms[i] = np.linalg.norm(self.th_hat - self.th, 'fro')

            # p = p + (np.random.random(len(p)) * 0.000001)
            self.choices[i] = self.p.argmax()
            self.arm_count[self.choices[i]] += 1
            self.rewards[i] = self.th[self.choices[i]].dot(x_i) * self.correlation_strength  + (1 - self.correlation_strength) * self.noise[i][self.choices[i]]
            # See what kind of result we get
            # rewards[i] = straight[choices[i]]
            # model_i = th2[choices[i]]
            # rewards[i] = model_i.predict(x_i.reshape(1, -1))[0]

            self.A[self.choices[i]] += np.outer(x_i, x_i)
            self.b[self.choices[i]] += self.rewards[i] * x_i

class UCB_Bandit:
    def __init__(self, n_arms, delta, seed, means, t_rounds):
        self.seed = seed
        np.random.seed(seed)
        self.n_arms = n_arms  # Anzahl der Arme
        self.delta = delta  # Parameter für den Konfidenzbonus
        self.arm_counts = np.zeros(n_arms)  # Zählungen für jeden Arm
        self.arm_reward_means = np.zeros(n_arms)  # Durchschnittliche Belohnung für jeden Arm
        self.actual_means = means
        self.t_rounds = t_rounds
        self.reward_history = np.zeros(self.t_rounds)
        self.played_round  = 0
        self.correlation_strength = 1


    def select_arm(self):
        ucb_of_arms = np.full(self.n_arms, np.inf)
        for i in range(self.n_arms):
            if self.arm_counts[i] == 0:
                continue
            else:
                ucb_of_arms[i] = self.arm_reward_means[i] + np.sqrt((2*np.log(np.pow(self.played_round,1))) / self.arm_counts[i]) #np.sqrt((2 * np.log(1 / self.delta)) / self.arm_counts[i])

        return np.argmax(ucb_of_arms)

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        self.arm_reward_means[arm] = ((n - 1) * self.arm_reward_means[arm] + reward) / n

    def execute(self):
        for n in range(self.t_rounds):
            self.played_round += 1
            arm = self.select_arm()
            #print(arm)
            ctr = self.actual_means[arm]
            # reward  = np.random.binomial(1, ctr)
            reward = ctr #ctr #ctr  * self.correlation_strength  + (1 - self.correlation_strength) * np.random.normal(0.0, 0.1)
            self.update(arm, reward)
            self.reward_history[n] = reward


# Erstelle Daten für verschiedene Korrelationsstufen
corr = Test_corr()
model_weak = corr.generate_data('strong')
means_weak = corr.means
#means_weak[1] = means_weak[1] * 1.5
max_mean = means_weak.max()
ucb_bandit = UCB_Bandit(n_arms=3, delta=0.5, seed=42, means=means_weak, t_rounds=1000000)
ucb_bandit.execute()


#LinUCB
data_weak = corr.generate_data('weak')
data_medium = corr.generate_data('medium')
data_strong = corr.generate_data('strong')
data_no = corr.generate_data('no')
context = corr.sample_contexts()
#bandit_weak = LinUCB(models=[data_weak], samples=context, alpha=0.2, correlation_strength=0.1)
#bandit_medium = LinUCB(models=[data_medium], samples=context, alpha=0.5, correlation_strength=0.4)
bandit_strong = LinUCB(models=[data_strong], samples=context, alpha=0.5, correlation_strength=1)
#bandit_no = LinUCB(models=[data_no], samples=context, alpha=0.2,  correlation_strength = 0)
#bandit_weak.run_LinUCB()
#bandit_medium.run_LinUCB()
bandit_strong.run_LinUCB()
#bandit_no.run_LinUCB()
print('finish')
#print("Sum", bandit_weak.choices.sum())


#UCB

reward_all = np.cumsum(ucb_bandit.reward_history)
ucb_cumulative_reward = reward_all #np.mean(reward_all, axis=0)
opt = [max_mean for _ in range(ucb_bandit.t_rounds)]
cumulative_optimal_reward = np.cumsum(opt)
cumulative_regret_UCB = cumulative_optimal_reward - ucb_cumulative_reward

#LinUCB
plt.figure(1, figsize=(10, 5))
plt.subplot(121)
#plt.plot(bandit_weak.norms, label='Weak')
#plt.plot(bandit_medium.norms, label='Medium')
plt.plot(bandit_strong.norms, label='Strong')
#plt.plot(bandit_no.norms, label='no')
plt.title("Frobeninus norm of estimated theta vs actual")
plt.legend()
plt.show()

#regret_weak = (bandit_weak.P.max(axis=1) - bandit_weak.rewards)
#regret_medium = (bandit_medium.P.max(axis=1) - bandit_medium.rewards)
regret_strong = (bandit_strong.P.max(axis=1) - bandit_strong.rewards)
#regret_no = (bandit_no.P.max(axis=1) - bandit_no.rewards)
plt.subplot(122)
#plt.plot(regret_weak.cumsum(), label='weak')
#plt.plot(regret_medium.cumsum(), label='medium')
plt.plot(regret_strong.cumsum(), label='strong')
#plt.plot(regret_no.cumsum(), label='no')
plt.plot(cumulative_regret_UCB, label='ucb')
plt.title("Cumulative regret")
plt.legend()
plt.show()

plt.subplot(122)
#plt.plot(bandit_weak.rewards.cumsum(), label='weak')
#plt.plot(bandit_medium.rewards.cumsum(), label='medium')
#plt.plot(bandit_strong.rewards.cumsum(), label='strong')
#plt.plot(bandit_no.rewards.cumsum(), label='no')
plt.plot(ucb_cumulative_reward, label='ucb')
plt.title("Cumulative reward")
plt.legend()
plt.show()



