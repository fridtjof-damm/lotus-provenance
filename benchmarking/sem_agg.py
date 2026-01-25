import os
import pandas as pd
import lotus
from lotus.models import LM

from sqllite_db import csv_to_sqlite, get_data_from_sql_as_dict, DEFAULT_DB_PATH


def main():
    # Ensure DB exists
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(repo_root, "IMDB Dataset.csv")
    if not os.path.exists(DEFAULT_DB_PATH):
        csv_to_sqlite(csv_path, DEFAULT_DB_PATH, limit=10000)

    # Load sample
    data = get_data_from_sql_as_dict(DEFAULT_DB_PATH, limit=60)
    df = pd.DataFrame(data)
    if "review" not in df.columns:
        raise ValueError("Expected 'review' column in SQLite table.")
    if "sentiment" not in df.columns:
        df["sentiment"] = "unknown"

    df = df[["sentiment", "review"]].head(20).copy()

    # LOTUS + Ollama (Mac local)
    os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    lotus.settings.configure(enable_cache=False)
    lotus.settings.configure(lm=LM(model=os.getenv("LOTUS_BENCH_MODEL", "ollama/llama3.2:3b")))

    # sem_agg only (grouped)
    summary = df.sem_agg(
        "You summarize ONE group of reviews (same {sentiment}).\n"
        "Output up to 3 bullets, each exactly:\n"
        "- Theme=<short>; Evidence=<short phrase>\n"
        "Use only {review}.",
        group_by=["sentiment"],
    )

    print("\n=== RESULT (sem_agg only, grouped by sentiment) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
