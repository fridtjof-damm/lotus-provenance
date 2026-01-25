import os
import sqlite3

import pandas as pd


def load_imdb_to_db():
    db_path = os.path.join(os.path.dirname(__file__), "imdb_reviews.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the table for IMDB reviews with review and sentiment columns
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS movie_reviews (
        review TEXT,
        sentiment TEXT
    );
    """)

    # Load data from CSV file into the database
    csv_path = "imdb_reviews.csv"
    df = pd.read_csv(csv_path)
    df.to_sql("imdb_reviews", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    load_imdb_to_db()
