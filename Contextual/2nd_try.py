import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Laden der Daten
df = pd.read_csv('../data.csv')

# CTR (Click-Through Rate) berechnen
df['ctr'] = df['clicks'] / df['impressions'].replace(0, 1)  # CTR: clicks / impressions

# Experten-Definition: Drei verschiedene Gruppen von Features
features_expert_1 = ['age', 'gender']  # Experte 1 verwendet age und gender
features_expert_2 = ['interest1', 'interest2', 'interest3']  # Experte 2 verwendet Interessen
features_expert_3 = ['age', 'gender', 'interest1', 'interest2', 'interest3']  # Experte 3 verwendet alle Features

# Kategorische Features in numerische Werte umwandeln
df['age'] = df['age'].astype('category').cat.codes
df['gender'] = df['gender'].astype('category').cat.codes

# Daten in X und y aufteilen
X1 = df[features_expert_1]  # Features für Experte 1
X2 = df[features_expert_2]  # Features für Experte 2
X3 = df[features_expert_3]  # Features für Experte 3
y = df['ctr']  # Zielvariable: CTR

# Lineare Regressionsmodelle für die Experten trainieren
expert_1_model = LinearRegression()
expert_2_model = LinearRegression()
expert_3_model = LinearRegression()

expert_1_model.fit(X1, y)
expert_2_model.fit(X2, y)
expert_3_model.fit(X3, y)
# Vorhersagen der Experten (Anfangsberatung)
expert_1_pred = expert_1_model.predict(X1)
expert_2_pred = expert_2_model.predict(X2)
expert_3_pred = expert_3_model.predict(X3)

# EXP4-Parameter initialisieren
num_experts = 3
eta = 0.05  # Lernrate
weights = np.ones(num_experts)  # Startgewichtungen der Experten
T = 1000  # Anzahl der Runden
ads = df['ad_id'].unique()  # Mögliche Aktionen (einzigartige Anzeigen)
num_ads = len(ads)
ad_count = {}


# Softmax-Funktion, um eine gültige Wahrscheinlichkeitsverteilung zu gewährleisten
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtrahiere den Maximalwert für numerische Stabilität
    return e_x / e_x.sum()


# Hilfsfunktion, um Wahrscheinlichkeiten auf Basis der Expertenprognosen zu berechnen
def calculate_probs(weights, experts_preds):
    weighted_sum = np.dot(weights, experts_preds)
    return softmax(weighted_sum)  # Softmax-Anwendung für Wahrscheinlichkeiten


# EXP4-Algorithmus-Simulation
for t in range(T):
    # Experten liefern ihre CTR-Vorhersagen für jede Anzeige
    experts_preds = np.vstack([expert_1_pred, expert_2_pred, expert_3_pred])

    # Wahrscheinlichkeitsverteilung über Aktionen (Anzeigen) berechnen
    P_t = calculate_probs(weights, experts_preds)

    # Wähle eine Anzeige basierend auf der Verteilung P_t
    chosen_ad = np.random.choice(ads, p=P_t)
    if chosen_ad in ad_count:
        ad_count[chosen_ad] += 1
    else:
        ad_count[chosen_ad] = 1

    # Beobachte die Belohnung (CTR) für die gewählte Anzeige
    actual_reward = np.random.binomial(1, p=df[df['ad_id'] == chosen_ad]['ctr'].mean())

    # Expertengewichte basierend auf ihren Vorhersagen und der tatsächlichen Belohnung aktualisieren
    for i in range(num_experts):
        pred = experts_preds[i][df['ad_id'] == chosen_ad].mean()
        estimated_reward = actual_reward * pred / P_t[ads == chosen_ad]
        weights[i] *= np.exp(eta * estimated_reward)

# Endgültige Expertengewichte nach den EXP4-Runden
print(weights)
print(ad_count)
