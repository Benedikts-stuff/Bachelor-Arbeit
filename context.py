import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Daten laden
df = pd.read_csv('Budgeted/Experiment/data/facebook-ad-campaign-data.csv')
# Überprüfen der Werte in den Spalten 'age' und 'gender'
#print(df[['age', 'gender']].head())
# Berechne die CTR für jede Kombination von Kontextmerkmalen
# Kombiniere die Kontextmerkmale in einer neuen Spalte

# Aggregiere die Klicks und Impressionen
ctr_df = df.groupby('context_combination').agg(
    clicks=('clicks', 'sum'),
    impressions=('ad_id', 'count')
).reset_index()

# Berechne die CTR
ctr_df['ctr'] = ctr_df['clicks'] / ctr_df['impressions']

# Sortiere nach CTR für bessere Visualisierung
ctr_df = ctr_df.sort_values(by='ctr', ascending=False)
ctr_df['context_index'] = np.arange(len(ctr_df))
# Plotten der CTR für alle Kontextkombinationen
plt.figure(figsize=(20, 10))
sns.barplot(data=ctr_df, x='context_index', y='ctr', errorbar=None)
plt.title('Click-Through Rate für alle Kontextkombinationen')
plt.xlabel('Kontextkombination (Alter_Geschlecht_Interesse)')
plt.ylabel('CTR')
plt.xticks(rotation=45, ha='right')  # Rotieren der x-Achsenbeschriftungen für bessere Lesbarkeit
plt.tight_layout()  # Verhindern von Überlappungen
plt.show()# One-Hot-Encoding auf 'age', 'gender', 'interest1', 'interest2', 'interest3'
#one_hot_encoded = pd.get_dummies(df, columns = ['age', 'gender'])
#context = ['interest1', 'interest2', 'interest3'] + \
                 # [col for col in one_hot_encoded.columns if 'age_' in col or 'gender_' in col]

# Das resultierende 'context'-Array sollte jetzt keine Nullen enthalten und korrekt encodiert sein
#print(context)

