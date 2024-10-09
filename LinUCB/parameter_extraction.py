import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Sampler:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path).drop(['reporting_start', 'reporting_end'], axis =1)
        self.encoded_data = pd.get_dummies(self.data, columns=['age', 'gender']).astype(int)
        self.age_col = self.encoded_data.columns[self.encoded_data.columns.str.contains('age')].tolist()
        self.interest_col = [col for col in self.encoded_data.columns if 'interest1' in col]
        self.gender_cols = self.encoded_data.columns[self.encoded_data.columns.str.contains('gender')].tolist()
        self.campaign_916 = self.encoded_data[self.encoded_data['campaign_id'] == 916]
        self.campaign_936 = self.encoded_data[self.encoded_data['campaign_id'] == 936]
        self.campaign_1178 = self.encoded_data[self.encoded_data['campaign_id'] == 1178]

        self.context_counts = self.encoded_data.groupby(self.age_col + self.gender_cols + self.interest_col).size().reset_index(name='count')
        self.total_counts = self.context_counts['count'].sum()
        self.context_counts['probability'] = self.context_counts['count'] / self.total_counts


    def sample_context(self):
        # Kontextvektoren und Wahrscheinlichkeiten extrahieren
        contexts = list(zip(self.context_counts['age_30-34'],self.context_counts['age_35-39'] ,self.context_counts['age_40-44'],
                            self.context_counts['age_45-49'], self.context_counts['gender_F'], self.context_counts['gender_M'], self.context_counts['interest1']))
        probabilities = self.context_counts['probability'].to_numpy()

        # Kontext basierend auf Wahrscheinlichkeiten auswählen
        sampled_context = np.random.choice(len(contexts), p=probabilities)
        column_names = ['age_30-34','age_35-39','age_40-44', 'age_45-49', 'gender_F', 'gender_M' ,'interest1']
        entry =contexts[sampled_context]
        return pd.DataFrame([entry], columns= column_names)

    def estimate_reward(self, action):
        y = 0
        X= pd.DataFrame()
        if(action == 0):
            grouped = self.campaign_916.groupby(['age_30-34','age_35-39','age_40-44', 'age_45-49', 'gender_F', 'gender_M' ,'interest1']).agg({
                'clicks': 'sum',
                'impressions': 'sum'
            }).reset_index()
            X =  grouped[['age_30-34','age_35-39','age_40-44', 'age_45-49', 'gender_F', 'gender_M' ,'interest1']]

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)
        # Modell trainieren
        model.fit(X_train, y_train)

        return model



#format data so that it is usable for the bandit
sampler = Sampler('../facebook-ad-campaign-data.csv')
