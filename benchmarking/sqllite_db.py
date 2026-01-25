import pandas as pd
import sqlite3
import os

# Configuration
INPUT_TSV_PATH = "/mnt/c/Users/bassi/Desktop/TU/RDE_pro/benchmark/data/IMDB Dataset.csv"
DB_NAME = "/mnt/c/imdb_benchmark/imdb_review.db"

def read_csv_to_df(path, limit=None):
    """
    Reads the IMDb CSV file into a Pandas DataFrame.
    """
    print(f"Reading CSV from {path}...")
    # Using low_memory=False to ensure consistent data types across chunks
    return pd.read_csv(
        path, 
        sep=',', 
        low_memory=False, 
        nrows=limit
    )

def read_tsv_to_df(path, limit=None):
    """Reads raw IMDb TSV into a Pandas DataFrame."""
    print(f"Reading {path}...")
    return pd.read_csv(
        path, sep='\t', compression='gzip', 
        low_memory=False, na_values='\\N', nrows=limit
    )

def index_to_sqlite(df, db_path):
    """Writes the DataFrame into SQLite (The 'Indexing' step)."""
    print(f"Indexing data into SQLite: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df.to_sql("movies", conn, if_exists="replace", index=False)
    print("Database indexing complete.")

def get_data_from_sql_as_dict(db_path):
    """
    Reads the data back from SQLite and returns it as a Python dictionary.
    """
    print("Querying SQLite and converting to dictionary...")
    with sqlite3.connect(db_path) as conn:
        query = "SELECT * FROM movies"
        df_from_sql = pd.read_sql_query(query, conn)
        
        # 'records' creates a list of dictionaries: [{col: val}, {col: val}]
        data_dict = df_from_sql.to_dict(orient="records")
        
    print(f"Successfully retrieved {len(data_dict)} records.")
    return data_dict

if __name__ == "__main__":
    if not os.path.exists(INPUT_TSV_PATH):
        print(f"Error: {INPUT_TSV_PATH} not found in the current directory.")
    else:
        # 1. Load from CSV
        initial_df = read_csv_to_df(INPUT_TSV_PATH, limit=10000)
        
        # 2. Store in SQLite
        index_to_sqlite(initial_df, DB_NAME)
        
        # 3. Retrieve from SQL as Dictionary
        imdb_data = get_data_from_sql_as_dict(DB_NAME)
        
        # Example: Print the first record to verify
        if imdb_data:
            print("\nSample Data from Dictionary:")
            print(imdb_data[0])