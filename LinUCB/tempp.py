import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def alter_mittelwert(altersgruppe):
    start, end = map(int, altersgruppe.split('-'))
    return (start + end) / 2

class Get_linear_model():
    def __init__(self,seed):
        self.test = pd.DataFrame()
        self.scaler = StandardScaler()
        #np.random.seed(seed)


    def get_model(self):
        # Datei einlesen
        file_path = '../facebook-ad-campaign-data.csv'
        data = pd.read_csv(file_path)
        scaler = StandardScaler()

        data['age'] = data['age'].apply(alter_mittelwert)
        data['gender'] = data['gender'].map({'M': 1, 'F': 0})

        noise_strength = 10
        data['age'] = data['age'] + np.random.normal(0, noise_strength, data.shape[0])
        data['interest2'] = data['interest2'] + np.random.normal(0, noise_strength, data.shape[0])
        data['interest3'] = data['interest3'] + np.random.normal(0, noise_strength, data.shape[0])
        data['gender'] = (data['gender'] + np.random.normal(0, noise_strength, data.shape[0])).round().clip(0, 1)
        data['impressions'] = data['impressions'] + np.random.normal(0, noise_strength, data.shape[0])
        data['clicks'] = data['clicks'] + np.random.normal(0, noise_strength, data.shape[0])


        data_916 = data[data['campaign_id'] == 916]
        data_936 = data[data['campaign_id'] == 936]
        data_1178 = data[data['campaign_id'] == 1178]

        data_916['ctr'] = (data_916['clicks'] / data_916['impressions']) * 1000
        data_916 = data_916.groupby(['age', 'gender', 'interest2', 'interest3'])['ctr'].mean().reset_index()

        data_936['ctr'] = (data_936['clicks'] / data_936['impressions']) * 1000
        data_936 = data_936.groupby(['age', 'gender', 'interest2', 'interest3'])['ctr'].mean().reset_index()

        data_1178['ctr'] = (data_1178['clicks'] / data_1178['impressions']) * 1000
        data_1178 = data_1178.groupby(['age', 'gender', 'interest2', 'interest3'])['ctr'].mean().reset_index()
        # Berechne die CTR (Click-Through-Rate)
        # data['ctr'] = data['clicks'] / data['impressions']


        # Fehlende oder unendliche Werte (z.B. durch division by zero) entfernen
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

        # Features und Zielvariable definieren
        X_train_916 = train_data_916[['age', 'gender', 'interest2', 'interest3']]
        print(train_data_916.corr()['ctr'])
        y_train_916 = train_data_916['ctr']

        X_test_916 = test_data_916[['age', 'gender', 'interest2', 'interest3']]
        y_test_916 = test_data_916['ctr']

        X_train_scaled_916 = self.scaler.fit_transform(X_train_916)
        X_test_scaled_916 = self.scaler.fit_transform(X_test_916)

        # Lineares Regressionsmodell erstellen und trainieren
        model_916 = LinearRegression()
        model_916.fit(X_train_scaled_916, y_train_916)

        # Vorhersagen auf den Testdaten
        y_pred_916 = model_916.predict(X_test_916)

        # Features und Zielvariable definieren
        X_train_936 = train_data_936[['age', 'gender', 'interest2', 'interest3']]
        print(train_data_936.corr()['ctr'])
        y_train_936 = train_data_936['ctr']

        X_test_936 = test_data_936[['age', 'gender', 'interest2', 'interest3']]
        y_test_936 = test_data_936['ctr']

        X_train_scaled_936 = self.scaler.fit_transform(X_train_936)
        X_test_scaled_936 = self.scaler.fit_transform(X_test_936)

        # Lineares Regressionsmodell erstellen und trainieren
        model_936 = LinearRegression()
        model_936.fit(X_train_scaled_936, y_train_936)

        # Vorhersagen auf den Testdaten
        y_pred_936 = model_936.predict(X_test_936)

        # Features und Zielvariable definieren
        X_train_1178 = train_data_1178[['age', 'gender', 'interest2', 'interest3']]
        print(train_data_1178.corr()['ctr'])
        y_train_1178 = train_data_1178['ctr']

        X_test_1178 = test_data_1178[['age', 'gender', 'interest2', 'interest3']]
        y_test_1178 = test_data_1178['ctr']

        X_train_scaled_1178 = self.scaler.fit_transform(X_train_1178)
        X_test_scaled_1178 = self.scaler.fit_transform(X_test_1178)

        # Lineares Regressionsmodell erstellen und trainieren
        model_1178 = LinearRegression()
        model_1178.fit(X_train_scaled_1178, y_train_1178)

        #  Vorhersagen auf den Testdaten
        y_pred_1178 = model_1178.predict(X_test_1178)

        # Berechnung von MSE und R²
        mse_916 = mean_squared_error(y_test_916, y_pred_916)
        mse_936 = mean_squared_error(y_test_936, y_pred_936)
        mse_1178 = mean_squared_error(y_test_1178, y_pred_1178)
        # r2 = r2_score(y_test, y_pred)
        r2_916 = model_916.score(X_test_scaled_916, y_test_916)
        r2_936 = model_936.score(X_test_scaled_936, y_test_936)
        r2_1178 = model_1178.score(X_test_scaled_1178, y_test_1178)

        # Ausgabe der Ergebnisse
        print("Mean Squared Error (MSE):", mse_916)
        print("Mean Squared Error (MSE):", mse_936)
        print("Mean Squared Error (MSE):", mse_1178)
        print("R²-Wert:", r2_916)
        print("R²-Wert:", r2_936)
        print("R²-Wert:", r2_1178)

        copy1 = test_data_916.copy()
        copy2 = test_data_916.copy()
        copy3 = test_data_916.copy()

        copy1['campaign_id'] = 916
        copy2['campaign_id'] = 936
        copy3['campaign_id'] = 1178
        self.test = pd.concat([copy1, copy2, copy3], ignore_index=True)

        grouped_context = data.groupby(['age', 'gender', 'interest2', 'interest3'])
        context_counts = grouped_context.size().reset_index(name='group_size')
        context_probs_df = context_counts['group_size'] / len(data)

        probs = context_probs_df.to_numpy()
        return [model_916, model_936, model_1178]


    def sample_contexts(self):
        # Datei einlesen
        file_path = '../facebook-ad-campaign-data.csv'
        data = pd.read_csv(file_path)

        data['age'] = data['age'].apply(alter_mittelwert)
        data['gender'] = data['gender'].map({'M': 1, 'F': 0})

        # Gruppenbildung und CTR-Berechnung
        grouped_context = data.groupby(['age', 'gender', 'interest2', 'interest3']).agg({
            'clicks': 'sum',
            'impressions': 'sum'
        }).reset_index()

        # Berechne die Gruppengrößen
        grouped_context['group_size'] = data.groupby(['age', 'gender', 'interest2', 'interest3']).size().values

        # Berechne CTR
        grouped_context['ctr'] = grouped_context['clicks'] / grouped_context['impressions']
        grouped_context = grouped_context.replace([None, float('inf'), float('-inf')], 0)

        # Wahrscheinlichkeiten basierend auf den Gruppengrößen berechnen
        context_probs_df = grouped_context['group_size'] / len(data)

        # Extrahiere die Kontext-Daten und Wahrscheinlichkeiten
        contexts = self.scaler.transform(grouped_context[['age', 'gender', 'interest2', 'interest3']])
        probs = context_probs_df.to_numpy()

        # 1000 Kontexte basierend auf den Wahrscheinlichkeiten ziehen
        chosen_indices = np.random.choice(len(contexts), size= 100000, p=probs)
        chosen_contexts = contexts[chosen_indices]

        return chosen_contexts

sampler = Get_linear_model(0)
sampler.get_model()

context = print(sampler.sample_contexts())