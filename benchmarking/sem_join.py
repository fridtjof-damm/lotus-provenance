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

    # 2) Load a small sample from SQLite
    data = get_data_from_sql_as_dict(DEFAULT_DB_PATH, limit=200)
    df_reviews = pd.DataFrame(data)

    if "review" not in df_reviews.columns:
        raise ValueError("Expected column 'review' in SQLite table (movies). Check sqllite_db.py schema.")
    if "sentiment" not in df_reviews.columns:
        df_reviews["sentiment"] = "unknown"

    # Keep only a few for quick testing
    df_reviews = df_reviews[["review", "sentiment"]].head(10).copy()

    # 3) Categories table (RIGHT side for sem_join)
    df_categories = pd.DataFrame(
        {
            "category": ["acting", "plot", "pacing", "dialogue", "cinematography", "sound", "other"],
            "definition": [
                "complaints/praise about performance, cast, acting quality",
                "story, narrative, twists, logic, writing of story",
                "slow/boring/dragging, too long, rhythm of the movie",
                "lines, script quality, conversations, writing of dialogue",
                "visuals, camera work, lighting, editing, VFX, look/feel",
                "music, audio, volume, sound effects, mixing",
                "no clear single category; general opinion or mixed feedback",
            ],
        }
    )

    # 4) Configure LOTUS + Ollama (Mac local)
    os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    lotus.settings.configure(enable_cache=False)
    lotus.settings.configure(lm=LM(model=os.getenv("LOTUS_BENCH_MODEL", "ollama/llama3.2:3b")))

    print(f"[OK] OLLAMA_API_BASE={os.environ['OLLAMA_API_BASE']}")
    print(f"[OK] Model={os.getenv('LOTUS_BENCH_MODEL', 'ollama/llama3.2:3b')}")

    # 5) SINGLE OPERATOR: sem_join
    # Goal: attach the best category to each review
    joined = df_reviews.sem_join(
        df_categories,
        "Choose the single best {category:right} for this review. "
        "If it matches {definition:right}, then assign it.\n\n"
        "Review: {review:left}",
    )

    print("\n=== RESULT (sem_join only) ===")
    cols = [c for c in ["sentiment", "review", "category", "definition"] if c in joined.columns]
    print(joined[cols].to_string(index=False))


if __name__ == "__main__":
    main()
