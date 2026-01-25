import os
import pandas as pd
import lotus
from lotus.models import LM

from sqllite_db import csv_to_sqlite, get_data_from_sql_as_dict, DEFAULT_DB_PATH


def main():
    # 1) Ensure DB exists (build from CSV if missing)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(repo_root, "IMDB Dataset.csv")

    if not os.path.exists(DEFAULT_DB_PATH):
        csv_to_sqlite(csv_path, DEFAULT_DB_PATH, limit=10000)

    # 2) Load rows from SQLite
    data = get_data_from_sql_as_dict(DEFAULT_DB_PATH, limit=500)  # a bit more helps topk
    df = pd.DataFrame(data)

    if "review" not in df.columns:
        raise ValueError("Expected column 'review' in SQLite table. Check sqllite_db.py schema.")
    if "sentiment" not in df.columns:
        df["sentiment"] = "unknown"

    # 3) Configure LOTUS + Ollama (Mac local)
    os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    lotus.settings.configure(enable_cache=False)
    lotus.settings.configure(lm=LM(model=os.getenv("LOTUS_BENCH_MODEL", "ollama/llama3.2:3b")))

    print(f"[OK] OLLAMA_API_BASE={os.environ['OLLAMA_API_BASE']}")
    print(f"[OK] Model={os.getenv('LOTUS_BENCH_MODEL', 'ollama/llama3.2:3b')}")

    # Optional: keep only positives so "most enthusiastic" makes sense
    df_pos = df[df["sentiment"].str.lower().eq("positive")].copy()
    if len(df_pos) < 10:
        df_pos = df.copy()  # fallback if sentiment isn't reliable

    # 4) SINGLE OPERATOR: sem_topk
    topk = df_pos.sem_topk(
        "Rank by strongest positive enthusiasm and excitement.\n"
        "Prefer reviews with intense praise, strong emotion, and superlatives.\n"
        "Use ONLY the review text: {review}",
        K=5,
    )

    print("\n=== RESULT (sem_topk only) ===")
    print(topk[["sentiment", "review"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
