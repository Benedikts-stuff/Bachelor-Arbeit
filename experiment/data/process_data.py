import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


AGE = "age"
GENDER = "gender"
INTEREST1 = "interest1"
CLICKS = "clicks"
IMPRESSIONS = "impressions"
CTR = "CTR"
CPC = "CPC"
CAMPAIGN_ID = "campaign_id"
SPENT = "spent"
GROUP_SIZE = "group_size"
PROB = "prob"
GENDER_M = "gender_M"
GENDER_F = "gender_F"



class PreProcessData:
    def __init__(self, file_path: str):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Datei '{file_path}' nicht gefunden.")

        self.df = pd.read_csv(file_path)
        self.df = self.preprocess_context_data()

    def bin_age(self, age: str) -> float:
        start, end = map(int, age.split('-'))
        return (start + end) / 2

    def preprocess_context_data(self) -> pd.DataFrame:
        self.df[CTR] = self.df[CLICKS] / self.df[IMPRESSIONS]
        self.df[CPC] = np.where(self.df[CLICKS] == 0, 0, self.df[SPENT] / self.df[CLICKS])

        self.df[AGE] = self.df[AGE].apply(self.bin_age)

        stats = pd.get_dummies(self.df, columns=[GENDER], drop_first=False, dtype=int)

        scaler = MinMaxScaler()
        stats[['age', 'interest1', 'interest2', 'interest3']] = scaler.fit_transform(
            stats[['age', 'interest1', 'interest2', 'interest3']])

        return stats

    def train_models(self, data: pd.DataFrame) -> dict:
        models = {}
        for campaign_id, group in data.groupby(CAMPAIGN_ID):
            # Features und Zielvariablen definieren
            X = group[[AGE, GENDER_M, GENDER_F, INTEREST1]]
            y_reward = group[CTR]
            y_cost = group[SPENT]

            # Train-Test-Split
            Xr_train, _, yr_train, _ = train_test_split(X, y_reward, test_size=0.2, random_state=42)
            Xc_train, _, yc_train, _ = train_test_split(X, y_cost, test_size=0.2, random_state=42)

            # Modelle trainieren
            model_reward = LinearRegression().fit(Xr_train, yr_train)
            model_cost = LinearRegression().fit(Xc_train, yc_train)

            # Modelle speichern
            models[campaign_id] = (model_reward, model_cost)
        return models

    def get_kde(self):
        df = self.df

        # Kombiniere die vorbereiteten Daten zu einem Feature-Vektor
        feature_matrix = np.hstack([df[[GENDER_M, 'age', 'interest1', 'interest2', 'interest3']].values])

        # Gewichtung mit impressions
        weights = df['impressions'].values
        weights = weights / weights.sum()  # Normierung der Gewichte auf Summe 1

        # 4. Multivariate Kernel Density Estimation (KDE)
        kde = gaussian_kde(feature_matrix.T, bw_method=0.15,
                           weights=weights)  # Transponieren, da gaussian_kde Zeilen als Samples erwartet

        return kde

    def train_gp_models_reward(self) -> list:
        df = self.df


        # 5. Für jede campaign_id die Modelle trainieren
        models = []

        for campaign_id, group in df.groupby('campaign_id'):
            # Definiere den Feature-Vektor – hier als Beispiel:
            # Es werden die Spalten "gender_M", "age", "interest1", "interest2" und "interest3" verwendet.
            X = group[['gender_M', 'age', 'interest1', 'interest2', 'interest3']].values

            # Zielvariable: CTR (angenommen, die Spalte heißt 'ctr')
            y = group[CTR].values

            # Definiere den Kernel für den Gaussian Process
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                     + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))

            gp = GaussianProcessRegressor(kernel=kernel,
                                          n_restarts_optimizer=10,
                                          random_state=42)

            # Trainiere den GP für die aktuelle Kampagne
            gp.fit(X, y)

            # Speichere das trainierte Modell im Dictionary
            models.append(gp)

        return models


    def train_gp_models_cost(self) -> list:
        df = self.df


        # 5. Für jede campaign_id die Modelle trainieren
        models = []
        for campaign_id, group in df.groupby('campaign_id'):
            # Definiere den Feature-Vektor – hier als Beispiel:
            # Es werden die Spalten "gender_M", "age", "interest1", "interest2" und "interest3" verwendet.
            X = group[['gender_M', 'age', 'interest1', 'interest2', 'interest3']].values

            # Zielvariable: CTR (angenommen, die Spalte heißt 'ctr')
            y = group[CPC].values

            # Definiere den Kernel für den Gaussian Process
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                     + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))

            gp = GaussianProcessRegressor(kernel=kernel,
                                          n_restarts_optimizer=10,
                                          random_state=42)

            # Trainiere den GP für die aktuelle Kampagne
            gp.fit(X, y)

            # Speichere das trainierte Modell im Dictionary
            models.append(gp)

        return models

