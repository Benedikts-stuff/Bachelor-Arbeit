import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


#np.random.seed(42)


n_samples = 10000

n_features = 3  # Beispiel: 3 Features (Alter, Geschlecht_M, Interesse)


age = np.random.randint(18, 65, size=n_samples)

gender = np.random.randint(0, 2, size=n_samples)
campaign_id = np.random.randint(0,3,size = n_samples)

interest = np.random.rand(n_samples)


X = np.column_stack((age, gender, interest))
X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

weights_campaign_0 = np.array([ 0.25812542,  0.00331851, -0.32298334])
weights_campaign_1 = np.array([ 0.33253651,  0.01682478, 0.42691954])
weights_campaign_2 = np.array([ 0.47180655, 0.17512985, -0.18789989])


linear_ctr = np.zeros(n_samples)
for i in range(n_samples):
    if campaign_id[i] == 0:
        linear_ctr[i] = X_scaled[i] @ weights_campaign_0
    elif campaign_id[i] == 1:
        linear_ctr[i] = X_scaled[i] @ weights_campaign_1
    else:
        linear_ctr[i] = X_scaled[i] @ weights_campaign_2


ctr = np.clip(linear_ctr, 0.1, 1)
minmax = MinMaxScaler()

df = pd.DataFrame({
    'campaign_id': campaign_id,
    'age':  X_scaled[:, 0],
    'gender': X_scaled[:, 1],
    'interest': X_scaled[:, 2],
     'ctr':ctr
})
#df['ctr'] = minmax.fit_transform(df[['ctr']])
df.to_csv('data_linear.csv', index=False)

print(df.head())
