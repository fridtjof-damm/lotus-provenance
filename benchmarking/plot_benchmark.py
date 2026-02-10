import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

""""
This script loads the raw data from the jsonl files into a DuckDB instance to postprocess the benchmarking data into
"""
USECASE_ALIASES = {
    # Pipeline Use Cases
    "01-UC-EXTRACT-FILTER": "sem_extract, sem_filter",
    "02-UC-FILTER-AGG": "sem_filter -> sem_agg",
    "03-UC-JOIN-FILTER": "sem_join -> sem_filter",
    "04-UC-TOPK-MAP": "sem_topk -> sem_map",
    # Single Operator Use Cases
    "05-UC-EXTRACT": "sem_extract",
    "06-UC-FILTER": "sem_filter",
    "07-UC-AGG": "sem_agg",
    "08-UC-JOIN": "sem_join",
    "09-UC-MAP": "sem_map",
    "10-UC-TOPK": "sem_topk",
}


def ingest_raw_data_sql(db_path, raw_data_path):
    raw_data = raw_data_to_df(raw_data_path)
    conn = duckdb.connect(db_path)
    # ingest new data
    conn.execute("CREATE OR REPLACE TABLE results AS SELECT * FROM raw_data")
    # keep newest run in a view per n-iterations and rows processed
    conn.execute("""
                CREATE OR REPLACE VIEW latest_runs AS
                SELECT *
                FROM results QUALIFY row_number() OVER (
                PARTITION BY usecase_id, "metadata.n_rows", iterations 
                ORDER BY timestamp DESC
            ) = 1;
                 """)

    print(f"Updated database: Total raw entries: {conn.execute('SELECT count(*) FROM results').fetchone()[0]}")
    conn.close()
    return raw_data


def raw_data_to_df(dir_path) -> pd.DataFrame:
    files = sorted([f for f in Path(dir_path).glob("*.jsonl")])
    data = []
    print(files)
    for file in files:
        with open(file, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    # json_normalize macht aus verschachtelten JSONs flache Tabellen
    df = pd.json_normalize(data)
    # Optional: Zeitstempel konvertieren für besseres Sortieren
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Ergebnisse speichern
    return df


def get_run_data(db_path, n_rows, iterations, model):
    conn = duckdb.connect(db_path)
    # Wir laden das Ergebnis direkt als Pandas DataFrame
    df = conn.execute(f"""
        SELECT *
        FROM latest_runs
        WHERE "metadata.n_rows" = {n_rows} AND iterations = {iterations} AND "metadata.model" = '{model}'
        AND usecase_id BETWEEN '05' AND '11'
        ORDER BY CAST(str_split(usecase_id, '-')[1] AS INTEGER) ASC;
    """).df()
    conn.close()
    return df


def generate_plots_from_df(df):
    # Die Logik bleibt identisch, wir nutzen nur das übergebene df
    if df.empty:
        print("No data found for the given parameters.")
        return

    # Metadaten extrahieren
    model = df["metadata.model"].iloc[0]
    n_iters = df["iterations"].iloc[0]
    # dataset = df["function"].iloc[0].split("_", 1)[1] if "_" in df["function"].iloc[0] else "Data"
    n_rows = df["metadata.n_rows"].iloc[0]
    system = df["metadata.system"].iloc[0]

    metrics = [
        ("wall_mean", "Wall Time (s)", "wall_std"),
        ("cpu_mean", "CPU Time (s)", "cpu_std"),
        ("mem_mean", "Memory Usage (MB)", "mem_std"),
    ]

    for metric_key, label, std_key in metrics:
        plt.figure(figsize=(12, 6))
        usecases = [USECASE_ALIASES.get(uid, uid) for uid in df["usecase_id"].tolist()]

        # Sicherstellen, dass die Spaltennamen mit den DuckDB-Spalten übereinstimmen
        vanilla_vals = df[f"results.vanilla.{metric_key}"].tolist()
        prov_vals = df[f"results.provenance.{metric_key}"].tolist()

        x = np.arange(len(usecases))
        width = 0.35

        plt.bar(x - width / 2, vanilla_vals, width, label="LOTUS", color="#1d3557")
        plt.bar(x + width / 2, prov_vals, width, label="LOTUS + Provenance", color="#f1a10d")

        plt.ylabel(label)
        plt.title(f"Benchmarking {label} for LOTUS Pipelines")
        plt.xticks(x, usecases, rotation=45)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        # Overhead Text-Labels
        for i in range(len(usecases)):
            v, p = vanilla_vals[i], prov_vals[i]
            overhead = ((p / v) - 1) * 100
            plt.text(
                x[i] + width / 2,
                p + (max(prov_vals) * 0.02),
                f"{overhead:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        info_text = f"Model: {model}\nIterations: {n_iters}\nRows: {df['metadata.n_rows'].iloc[0]}"
        plt.gca().text(
            0.98,
            0.02,
            info_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
        )

        plt.tight_layout()
        Path("results/plots").mkdir(exist_ok=True)
        plt.savefig(f"results/plots/plot_{metric_key}_{model.replace('/','-')}_{n_rows}_{system}.pdf", format="pdf")
        plt.show()


if __name__ == "__main__":
    db_path = "benchmark_results.db"
    raw_data_path = "results/raw_data/"
    # 1. Daten in DB aktualisieren
    ingest_raw_data_sql(db_path, raw_data_path)
    # 2. Gezielte Daten für den Plot abfragen (n_rows=10, iters=5)

    # filtered_df = get_run_data(db_path, 100, 3, "ollama/llama3.2:3b")
    # 3. Plotten
    # generate_plots_from_df(filtered_df)
