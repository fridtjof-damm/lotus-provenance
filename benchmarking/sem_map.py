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
    data = get_data_from_sql_as_dict(DEFAULT_DB_PATH, limit=200)
    df = pd.DataFrame(data)

    # (Optional) keep only needed columns
    if "review" not in df.columns:
        raise ValueError("Expected column 'review' in SQLite table. Check sqllite_db.py schema.")
    if "sentiment" not in df.columns:
        # If your DB doesn't store sentiment, you can remove sentiment usage from prompts.
        df["sentiment"] = "unknown"

    # 3) Configure LOTUS + Ollama (Mac local)
    os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    lotus.settings.configure(enable_cache=False)
    lotus.settings.configure(lm=LM(model=os.getenv("LOTUS_BENCH_MODEL", "ollama/llama3.2:3b")))

    print(f"[OK] OLLAMA_API_BASE={os.environ['OLLAMA_API_BASE']}")
    print(f"[OK] Model={os.getenv('LOTUS_BENCH_MODEL', 'ollama/llama3.2:3b')}")

    # 4) SINGLE OPERATOR: sem_map
    # Example task: extract one dominant complaint/praise aspect (easy to check)
    mapped = df.sem_map(
        "From this IMDb review, output exactly ONE label from:\n"
        "acting/plot/pacing/dialogue/cinematography/sound/other\n"
        "Rules:\n"
        "- If the review is mostly positive or no clear single aspect => other\n"
        "- Return ONLY the label (one word), lowercase.\n"
        "Review: {review}",
        suffix="_aspect",
    )

    print("\n=== RESULT (sem_map only) ===")
    print(mapped[["sentiment", "review", "_aspect"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
