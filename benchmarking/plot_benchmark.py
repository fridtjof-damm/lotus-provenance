import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_plots(jsonl_path):
    # 1. Load and Flatten Data
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.json_normalize(data)

    # 2. Extract Metadata for Plot Text (Assume consistency across runs)
    model = df["metadata.model"].iloc[0]
    n_iters = df["iterations"].iloc[0]
    # Extract dataset name from function name (e.g., 'movie_reviews')
    dataset = df["function"].iloc[0].split("_", 1)[1] if "_" in df["function"].iloc[0] else "Data"

    metrics = [
        ("wall_mean", "Wall Time (s)", "wall_std"),
        ("cpu_mean", "CPU Time (s)", "cpu_std"),
        ("mem_mean", "Memory Usage (MB)", "mem_std"),
    ]

    # 3. Create a Plot for each Metric
    for metric_key, label, std_key in metrics:
        plt.figure(figsize=(12, 6))

        usecases = df["usecase_id"].tolist()
        vanilla_vals = df[f"results.vanilla.{metric_key}"].tolist()
        prov_vals = df[f"results.provenance.{metric_key}"].tolist()

        # vanilla_std = df[f"results.vanilla.{std_key}"].tolist()
        # prov_std = df[f"results.provenance.{std_key}"].tolist()

        x = np.arange(len(usecases))
        width = 0.35

        # Grouped Bars
        plt.bar(x - width / 2, vanilla_vals, width, label="LOTUS", color="#1d3557")
        plt.bar(x + width / 2, prov_vals, width, label="LOTUS + Provenance", color="#f1a10d")

        # Labels and Title
        plt.ylabel(label)
        plt.title(f"Benchmarking {label} for LOTUS Pipelines")
        plt.xticks(x, usecases, rotation=45)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        for i in range(len(usecases)):
            v = vanilla_vals[i]
            p = prov_vals[i]
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

        info_text = f"Model: {model}\nIterations: {n_iters}\nDataset: {dataset}"
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
        pdf_filename = f"results/plots/plot_{metric_key}.pdf"
        plt.savefig(pdf_filename, format="pdf")
        plt.show()


if __name__ == "__main__":
    generate_plots("results/provenance_benchmarks.jsonl")
