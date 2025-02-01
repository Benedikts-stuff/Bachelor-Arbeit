import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
import os

np.random.seed(42)

# Statische Variablen für Spaltennamen
AGE = "age"
GENDER = "gender"
INTEREST1 = "interest1"
CLICKS = "clicks"
IMPRESSIONS = "impressions"
CTR = "CTR"
CAMPAIGN_ID = "campaign_id"
SPENT = "spent"
GROUP_SIZE = "group_size"
PROB = "prob"
GENDER_M = "gender_M"
GENDER_F = "gender_F"

# Weitere statische Variablen
FILE_PATH = "./facebook-ad-campaign-data.csv"  # Dateipfad zur CSV-Datei
OUTPUT_CONTEXT_PROBS = "context_probs.csv"  # Output-Datei für Kontextwahrscheinlichkeiten
SEED = 42  # Zufallsseed für Train-Test-Split


def load_data(file_path: str) -> pd.DataFrame:
    """Lädt die Daten aus einer CSV-Datei."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Datei '{file_path}' nicht gefunden.")
    return pd.read_csv(file_path)


def bin_age(age: str) -> float:
    """Berechnet den mittleren Wert eines Altersbereichs."""
    start, end = map(int, age.split('-'))
    return (start + end) / 2


def preprocess_context_data(data: pd.DataFrame, scaler) -> pd.DataFrame:
    """Vorverarbeitung der Kontextdaten."""
    # CTR berechnen
    data[CTR] = data[CLICKS] / data[IMPRESSIONS]

    # Gruppieren und Wahrscheinlichkeiten berechnen
    stats = data.groupby([AGE, GENDER, INTEREST1]).size().reset_index(name=GROUP_SIZE)
    stats[PROB] = stats[GROUP_SIZE] / len(data)

    # Altersbereiche binnen
    stats[AGE] = stats[AGE].apply(bin_age)

    # One-Hot-Encoding für Gender
    stats = pd.get_dummies(stats, columns=[GENDER], drop_first=False)

    # Skalierung
    stats[[AGE, INTEREST1]] = scaler.fit_transform(stats[[AGE, INTEREST1]])
    return stats


def compute_context_probs(file_path: str, scaler=MinMaxScaler()) -> None:
    """Berechnet Wahrscheinlichkeiten für Kontexte und speichert die Ergebnisse als CSV."""
    data = load_data(file_path)
    processed_data = preprocess_context_data(data, scaler)
    processed_data.to_csv(OUTPUT_CONTEXT_PROBS, index=False)
    print(f"Kontextwahrscheinlichkeiten gespeichert in '{OUTPUT_CONTEXT_PROBS}'.")


def preprocess_reward_data(data: pd.DataFrame, scaler) -> pd.DataFrame:
    """Vorverarbeitung der Reward-Daten."""
    # CTR berechnen
    data[CTR] = data[CLICKS] / data[IMPRESSIONS]
    data[AGE] = data[AGE].apply(bin_age)

    # Gruppieren
    stats = data.groupby([CAMPAIGN_ID, AGE, GENDER, INTEREST1]).agg(
        CTR=(CTR, "mean"),
        spent=(SPENT, "mean")
    ).reset_index()

    # One-Hot-Encoding für Gender
    stats = pd.get_dummies(stats, columns=[GENDER], drop_first=False)

    # Skalierung
    stats[[AGE, INTEREST1]] = scaler.transform(stats[[AGE, INTEREST1]])
    return stats


def train_models(data: pd.DataFrame) -> dict:
    """Trainiert lineare Modelle für jede Kampagne."""
    models = {}
    for campaign_id, group in data.groupby(CAMPAIGN_ID):
        # Features und Zielvariablen definieren
        X = group[[AGE, GENDER_M, GENDER_F, INTEREST1]]
        y_reward = group[CTR]
        y_cost = group[SPENT]

        # Train-Test-Split
        Xr_train, _, yr_train, _ = train_test_split(X, y_reward, test_size=0.2, random_state=SEED)
        Xc_train, _, yc_train, _ = train_test_split(X, y_cost, test_size=0.2, random_state=SEED)

        # Modelle trainieren
        model_reward = LinearRegression().fit(Xr_train, yr_train)
        model_cost = LinearRegression().fit(Xc_train, yc_train)

        # Modelle speichern
        models[campaign_id] = (model_reward, model_cost)
    return models


def compute_reward_probs(file_path: str, scaler=MinMaxScaler()) -> dict:
    """Berechnet und trainiert Modelle für Belohnungen und Kosten."""
    data = load_data(file_path)
    processed_data = preprocess_reward_data(data, scaler)
    models = train_models(processed_data)

    print(f"Anzahl der trainierten Modelle: {len(models)}")
    return models


def groups(file_path: str):
    data = load_data(file_path)
    data.groupby(["age","gender","interest1","interest2","interest3"])
    print(len(data))


def split_and_aggregate_by_campaign(filepath, output_dir="./"):

    # Datei einlesen
    df = pd.read_csv(filepath)

    # Drei zufällige, eindeutige Kampagnen-IDs auswählen
    campaign_ids = df["campaign_id"].drop_duplicates().sample(3, random_state=42).tolist()

    # Dictionary zum Speichern der Teil-Daten
    campaign_data = {}

    for campaign_id in campaign_ids:
        # Filtern nach der aktuellen Kampagne
        df_campaign = df[df["campaign_id"] == campaign_id]

        # Gruppierung nach Alter und Geschlecht
        aggregated_df = df_campaign.groupby(["age", "interest1"]).agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            spent=("spent", "sum"),
            group_size=("campaign_id", "count")  # Anzahl der Einträge in jeder Gruppe
        ).reset_index()

        # Speichern im Dictionary
        campaign_data[campaign_id] = aggregated_df

        # CSV-Datei speichern
        output_path = f"{output_dir}campaign_{campaign_id}.csv"
        aggregated_df.to_csv(output_path, index=False)
        print(f"Gespeichert: {output_path}")

    return campaign_data


# Beispielaufruf
file_path = "facebook-ad-campaign-data.csv"
#output_directory = "./"  # Speicherort der CSVs anpassen
#campaign_datasets = split_and_aggregate_by_campaign(file_path, output_directory)

def plot_kde_contour(feature_matrix, sampled_values):
    # Extrahiere "age" (Spalte 1) und "interest1" (Spalte 2)
    original_age = feature_matrix[:, 1]
    original_interest = feature_matrix[:, 2]
    sampled_age = sampled_values[:, 1]
    sampled_interest = sampled_values[:, 2]

    # Erstelle ein 2D-Gitter für den Konturplot
    x = np.linspace(0, 1, 100)  # Für "age"
    y = np.linspace(0, 1, 100)  # Für "interest1"
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Berechne die 2D-KDE der gesampelten Daten
    kde_sampled = gaussian_kde(np.vstack([sampled_age, sampled_interest]))
    Z_sampled = np.reshape(kde_sampled(positions).T, X.shape)

    # Plot
    plt.figure(figsize=(10, 6))

    # Kontur der KDE-Samples
    contour = plt.contourf(X, Y, Z_sampled, levels=15, cmap='Blues', alpha=0.7)
    plt.colorbar(contour, label='Dichte')

    # Originaldatenpunkte
    plt.scatter(original_age, original_interest,
                c='red', s=20, edgecolor='black',
                alpha=0.5, label='Originaldaten')

    plt.title('2D-Konturplot der KDE-Verteilung (Age vs. Interest1)')
    plt.xlabel('Age (skaliert)')
    plt.ylabel('Interest1 (skaliert)')
    plt.legend()
    plt.savefig('kde_distribution_contour.pdf', bbox_inches='tight')  # Speichern als PDF
    plt.show()


def plot_key_pairs(sampled_values):
    pairs = [
        ('age', 'interest1'),
        ('age', 'interest2'),
        ('age', 'interest3'),
        ('interest1', 'interest2'),
        ('interest1', 'interest3'),
        ('interest2', 'interest3')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (feat1, feat2) in enumerate(pairs):
        ax = axes[idx]
        # Extrahiere Spaltenindizes basierend auf Feature-Namen
        i = ['age', 'interest1', 'interest2', 'interest3'].index(feat1) + 1  # +1, da Gender Spalte 0
        j = ['age', 'interest1', 'interest2', 'interest3'].index(feat2) + 1

        # 2D-KDE
        kde = gaussian_kde(np.vstack([sampled_values[:, i], sampled_values[:, j]]))
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.reshape(kde(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)
        ax.contourf(X, Y, Z, levels=15, cmap='Blues', alpha=0.7)
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)

    plt.tight_layout()
    plt.savefig('kde_key_pairs.pdf', bbox_inches='tight')
    plt.show()


def plot_kde_grid(feature_matrix, sampled_values):
    features = ['age', 'interest1', 'interest2', 'interest3']
    n = len(features)

    # LaTeX-Einstellungen für Matplotlib
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.latex.preamble'] = (
        r'\usepackage{libertine}'
        r'\usepackage[libertine]{newtxmath}'
        r'\RequirePackage[scaled=.92]{sourcesanspro}'
    )

    fig, axes = plt.subplots(n, n, figsize=(15, 15))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # 1D-Dichte entlang der Diagonale
                sns.kdeplot(sampled_values[:, i + 1], ax=ax, fill=True)  # +1, da Gender Spalte 0 ist
                ax.set_title(rf"Density of $\mathtt{{{features[i]}}}$", fontsize=12)  # LaTeX-Rendering mit dynamischem Feature-Namen
            else:
                # 2D-Konturplots
                kde = gaussian_kde(np.vstack([sampled_values[:, i + 1], sampled_values[:, j + 1]]))
                x = np.linspace(0, 1, 100)
                y = np.linspace(0, 1, 100)
                X, Y = np.meshgrid(x, y)
                Z = np.reshape(kde(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)
                ax.contourf(X, Y, Z, levels=15, cmap='Blues', alpha=0.7)
                ax.scatter(feature_matrix[:, i + 1], feature_matrix[:, j + 1], c='red', s=5, alpha=0.3)
                ax.set_xlabel(rf"$\mathtt{{{features[i]}}}$", fontsize=12)  # LaTeX-Rendering für Achsenbeschriftungen
                ax.set_ylabel(rf"$\mathtt{{{features[j]}}}$", fontsize=12)

    plt.tight_layout()
    plt.savefig('kde_grid.pdf', bbox_inches='tight')
    plt.show()


def kde_data(file_path: str):
    # CSV-Datei einlesen
    df = pd.read_csv(file_path)

    df['age'] = df['age'].apply(bin_age)

    df = pd.get_dummies(df, columns=[GENDER], drop_first=True)

    # 3. Min-Max-Scaling für 'age' und 'interest1' zusammen
    scaler = MinMaxScaler()
    df[['age', 'interest1', 'interest2', 'interest3']] = scaler.fit_transform(df[['age', 'interest1', 'interest2', 'interest3']])

    # Kombiniere die vorbereiteten Daten zu einem Feature-Vektor
    feature_matrix = np.hstack([df[[GENDER_M, 'age', 'interest1', 'interest2', 'interest3']].values])

    # Gewichtung mit impressions
    weights = df['impressions'].values
    weights = weights / weights.sum()  # Normierung der Gewichte auf Summe 1

    # 4. Multivariate Kernel Density Estimation (KDE)
    kde = gaussian_kde(feature_matrix.T, bw_method=0.15, weights=weights)  # Transponieren, da gaussian_kde Zeilen als Samples erwartet

    # Neue Werte aus der multivariaten KDE samplen
    sampled_values =  np.clip(kde.resample(size=10000).T, 0, 1).astype(np.float64)

    #plot_kde_contour(feature_matrix, sampled_values)
    #plot_key_pairs(sampled_values)
    plot_kde_grid(feature_matrix, sampled_values)

    # Beispiel: Einen neuen Feature-Vektor aus der Verteilung samplen
    new_feature_vector = kde.resample(size=1).T
    print("Ein neuer gesampleter Feature-Vektor (gender, age, interest1):", new_feature_vector)

    plt.scatter(sampled_values[:, 1], sampled_values[:, 2], alpha=0.5, label='Sampled Values')
    plt.scatter(feature_matrix[:, 1], feature_matrix[:, 2], alpha=0.5, label='Original Data')
    plt.xlabel('Age (scaled)')
    plt.ylabel('Interest1 (scaled)')
    plt.legend()
    plt.show()

def plot_sin():
    # 1. Generiere synthetischen Kontext
    x = np.linspace(0, 1, 500)  # Variiere x1 von 0 bis 1
    x2 = 0.5  # Fixiere andere Kontextdimensionen
    x3 = 0.5

    # 2. Berechne Reward für jeden Kontext
    reward = 0.5 * np.sin(10 * x + 10 * x2 + 10 * x3) + 0.5  # Annahme: x · 10 = Summe(10*x_i)

    # 3. Plot mit Seaborn
    plt.figure(figsize=(12, 6))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'sourcesanspro'
    plt.rcParams[
        'text.latex.preamble'] = r'\usepackage{libertine}\usepackage[libertine]{newtxmath}\RequirePackage[scaled=.92]{sourcesanspro}'
    sns.lineplot(x=x, y=reward, color='green')
    # Style-Anpassungen
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel(r"$r_{a}(x)$", fontsize=18)
    plt.grid(alpha=0.3)

    plt.savefig("sinus_reward.pdf", bbox_inches='tight')
    plt.show()


kde_data(file_path)
#plot_sin()