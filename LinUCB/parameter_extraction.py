import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Sampler:
    def __init__(self, data_path, seed):
        np.random.seed(seed)
        self.scaler = MinMaxScaler()
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path).drop(['reporting_start', 'reporting_end'], axis =1)
        self.data['gender'] = self.data['gender'].map({'M': 1, 'F': 0})
        self.data['age'] = self.data['age'].map({'30-34': 0, '35-39': 1, '40-44': 2, '45-49': 3})
        #self.data = pd.get_dummies(self.data, columns=['age'], drop_first=True)
        self.grouped_context = self.data.groupby(['age']).agg({
                                        'clicks': 'sum',
                                         'impressions': 'sum'
                                        }).reset_index()

        self.grouped_context['ctr'] = (self.grouped_context['clicks'] / self.grouped_context['impressions']) * 1000
        self.features = self.grouped_context.drop(columns=['clicks', 'impressions', 'ctr']).to_numpy()

        self.campaign_916 = self.data[self.data['campaign_id'] == 916]
        self.campaign_936 = self.data[self.data['campaign_id'] == 936]
        self.campaign_1178 = self.data[self.data['campaign_id'] == 1178]

    #return array with all possible feature vectors
    def get_features(self):
        return self.features

    # return vector with probabilities of the occurence of each context (index in this array of a context probability should match the index in features)
    def get_context_probs(self):
        grouped_context = self.data.groupby(['age', 'gender'])
        context_counts = grouped_context.size().reset_index(name='group_size')
        context_probs_df = context_counts['group_size'] / len(self.data)

        probs = context_probs_df.to_numpy()
        return probs

    # return linear model that eastimates the reward given the context vector for a specific campaign
    def get_model(self, action):
        y = 0
        X= pd.DataFrame()
        if(action == 0):
            grouped = self.campaign_916.groupby(['age', 'gender', 'interest1', 'interest2']).agg({
                'clicks': 'sum',
                'impressions': 'sum'
            }).reset_index()
            X =  grouped[['age','gender', 'interest1','interest2']]

            y = pd.DataFrame(grouped['clicks'] / grouped['impressions'], columns=['CTR'])

        elif(action == 1):
            grouped = self.campaign_936.groupby(['age_30-34','age_35-39','age_40-44', 'age_45-49', 'gender_F', 'gender_M' ,'interest1']).agg({
                'clicks': 'sum',
                'impressions': 'sum'
            }).reset_index()
            X =  grouped[['age_30-34','age_35-39','age_40-44', 'age_45-49', 'gender_F', 'gender_M' ,'interest1']]

            y = pd.DataFrame(grouped['clicks'] / grouped['impressions'], columns=['CTR'])

        elif(action == 2):
            grouped = self.campaign_1178.groupby(['age_30-34','age_35-39','age_40-44', 'age_45-49', 'gender_F', 'gender_M' ,'interest1']).agg({
                'clicks': 'sum',
                'impressions': 'sum'
            }).reset_index()
            X =  grouped[['age_30-34','age_35-39','age_40-44', 'age_45-49', 'gender_F', 'gender_M' ,'interest1']]

            y = pd.DataFrame(grouped['clicks'] / grouped['impressions'], columns=['CTR'])


        model = LinearRegression()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=34)
        # Modell trainieren
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R-squared: {r2}")

        return model



sampler = Sampler('../facebook-ad-campaign-data.csv', 0)
model0 = sampler.get_model(0)