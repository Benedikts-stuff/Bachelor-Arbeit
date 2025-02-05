import pandas as pd
import os


class Logger:
    def __init__(self, filename="experiment_logs.csv"):
        self.filename = filename
        self.logs = []

    def log(self, algo_name, round_num, reward, regret, normalized_used_budget, run_index, seed):
        self.logs.append({
            "algorithm": algo_name,
            "round": round_num,
            "reward": reward,
            "cumulative_regret": regret,
            "normalized_used_budget": normalized_used_budget,
            "run_index": run_index,
            "seed": seed
        })

    def save_to_csv(self):
        if not self.logs:
            return  # Falls keine neuen Logs vorhanden sind, nichts speichern

        df = pd.DataFrame(self.logs)

        # Explizite Datentypen setzen
        dtype_mapping = {
            "algorithm": "string",
            "round": "int64",
            "reward": "float64",
            "cumulative_regret": "float64",
            "normalized_used_budget": "float64",
            "run_index": "int64",
            "seed": "int64"
        }

        df = df.astype(dtype_mapping)

        file_exists = os.path.isfile(self.filename)

        # Speichere Daten im Append-Modus ('a'), falls Datei existiert, sonst erstelle mit Header
        df.to_csv(self.filename, index=False, mode='a', header=not file_exists)
        self.logs = []  # Speicher freigeben
