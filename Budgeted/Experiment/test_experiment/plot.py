import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def interp_plot(csv_path, x_col="normalized_used_budget", y_col="cumulative_regret"):
    # Lese die CSV-Datei ein
    df = pd.read_csv(csv_path)

    # Konvertiere Spalten in float und entferne ungültige Werte
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])

    # X-Werte für die Interpolation
    new_axis_xs = np.linspace(0, 1, 100)

    results = []

    # Iteriere über jeden Algorithmus
    for algo, algo_group in df.groupby("algorithm"):
        interpolated_runs = []

        # Iteriere über jeden Run des Algorithmus
        for _, run_group in algo_group.groupby("run_index"):
            sorted_run = run_group.sort_values(by=x_col).drop_duplicates(x_col)

            # Interpoliere die Werte für diesen Run
            interp_y = np.interp(new_axis_xs, sorted_run[x_col], sorted_run[y_col])
            interpolated_runs.append(interp_y)

        # Konvertiere Liste in NumPy-Array für einfachere Berechnung
        interpolated_runs = np.array(interpolated_runs)

        # Berechne Mittelwert und statistische Werte
        midy = np.mean(interpolated_runs, axis=0)
        q25 = np.percentile(interpolated_runs, 25, axis=0)
        q75 = np.percentile(interpolated_runs, 75, axis=0)
        std = np.std(interpolated_runs, axis=0)
        ci_lower = np.percentile(interpolated_runs, 25, axis=0)
        ci_upper = np.percentile(interpolated_runs, 75, axis=0)
        summed = [np.max(regret) for regret in interpolated_runs]

        # Speichere Ergebnisse für diesen Algorithmus
        results.append(pd.DataFrame({
            "algorithm": algo,
            x_col: new_axis_xs,
            y_col: midy,
            "summed": np.pad(summed, (0, len(new_axis_xs) - len(summed)), mode='constant', constant_values=np.nan),
            "q25": q25,
            "q75": q75,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }))

    # Kombiniere Ergebnisse aller Algorithmen
    final_df = pd.concat(results, ignore_index=True)

    return final_df



# Globales Mapping der Farben erstellen
def create_global_color_mapping(plot_data):
    # Erstelle die Farbpallette basierend auf der Anzahl der Methoden
    n_colors = len(plot_data)
    palette = sns.color_palette("viridis", n_colors=n_colors) #sns.cubehelix_palette(n_colors=n_colors, dark= 0.25, start=2)

    color_mapping = {name: palette[i] for i, name in enumerate(plot_data)}
    return color_mapping


def plot_budget_normalised_regret(plot_data, color_mapping):
    # Initialisiere das Plot-Figure
    plt.figure(figsize=(10, 6))

    y_lim = 0  # Initialisiere den maximalen y-Wert für die y-Achsenbegrenzung
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'sourcesanspro'
    plt.rcParams[
        'text.latex.preamble'] = r'\usepackage{libertine}\usepackage[libertine]{newtxmath}\RequirePackage[scaled=.92]{sourcesanspro}'

    # Iteriere über jeden Algorithmus und plotte die Daten mit Errorbars
    for algorithm, data in plot_data.groupby("algorithm"):
        # Hole Farbe aus der Farbzuordnung
        color = color_mapping[algorithm]
        style = '-'  # Linienstil

        # Plotte den kumulativen Regret über das normalisierte Budget
        sns.lineplot(x="normalized_used_budget", y="cumulative_regret", data=data, label=algorithm, color=color,
                     linestyle=style, linewidth=2)

        # Wähle einige Punkte für Errorbars aus (z. B. jeden 15. Index)
        sampled_indices = np.arange(15, len(data), 15)
        sampled_data = data.iloc[sampled_indices]

        # Füge Errorbars für die 95%-Konfidenzintervalle hinzu
        plt.errorbar(sampled_data["normalized_used_budget"], sampled_data["cumulative_regret"],
                     yerr=[np.clip(sampled_data["cumulative_regret"] - sampled_data["ci_lower"], 0, None),
                           np.clip(sampled_data["ci_upper"] - sampled_data["cumulative_regret"], 0, None)],
                     fmt='o', color=color, capsize=3, elinewidth=1, markeredgewidth=1)

        # Aktualisiere den maximalen y-Wert für die y-Achsenbegrenzung
        y_lim = max(y_lim, data["cumulative_regret"].max() + data["std"].max())

    # Setze Achsenbeschriftungen und -grenzen
    plt.xlabel("Normalized Spent Budget", fontsize=14)
    plt.ylabel("Cumulative Regret", fontsize=14)
    plt.ylim(0, y_lim)  # Setze die y-Achsenbegrenzung
    plt.legend(fontsize=12)  # Zeige die Legende an
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Füge ein Gitter hinzu

    # Speichere den Plot als PDF
    plt.savefig("cumulative_regret_linear.pdf", dpi=300)  # Höhere Auflösung für bessere Qualität
    plt.show()  # Zeige den Plot an


def plot_violin_regret(plot_data, color_mapping):
    plt.figure(figsize=(10, 6))

    # Setze LaTeX-Schriftarten für Konsistenz
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'sourcesanspro'
    plt.rcParams[
        'text.latex.preamble'] = r'\usepackage{libertine}\usepackage[libertine]{newtxmath}\RequirePackage[scaled=.92]{sourcesanspro}'


    # Erstelle den Violin-Plot
    sns.violinplot(x="algorithm", y="summed", data=plot_data, inner="quart",
                   palette=color_mapping, scale="area", cut=0)

    # Setze Titel und Labels
    plt.ylabel(r"Cumulative Regret", fontsize=14, labelpad=15)
    plt.xlabel("Algorithm", fontsize=14)

    # Achsenbeschriftungen und Layout-Anpassungen
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Speichere den Plot als PDF
    plt.savefig("violin_plot_regret.pdf", dpi=300)
    plt.show()


# Erstelle eine Farbzuordnung für die Algorithmen
plot_data = interp_plot("experiment_logs.csv")
color_mapping = create_global_color_mapping(plot_data["algorithm"].unique())
plot_budget_normalised_regret(plot_data, color_mapping)
plot_violin_regret(plot_data, color_mapping)