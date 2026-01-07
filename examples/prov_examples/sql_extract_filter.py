import sqlite3

import pandas as pd

import lotus
from lotus.data_connectors import DataConnector
from lotus.models import LM

lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

conn = sqlite3.connect("example_movie_reviews.db")
cursor = conn.cursor()

"""
This example depends on a movies dataset from kaggle: link
"""
csv_path = "movie_reviews.csv"

# Create the table
cursor.execute("""
CREATE TABLE IF NOT EXISTS movie_reviews (
    id TEXT,
    reviewID INTEGER PRIMARY KEY,
    review TEXT,
    rating REAL,
    review_date TEXT,
    critic_name TEXT,
    is_top_critic INTEGER,
    original_score TEXT,
    review_state TEXT,
    publication_name TEXT,
    review_url TEXT
);
""")

# insert data from csv file
df = pd.read_csv(csv_path)
df.to_sql("movie_reviews", conn, if_exists="replace", index=False)

conn.commit()


query = "SELECT * FROM movie_reviews LIMIT 100;"
df = DataConnector.load_from_db("sqlite:///example_movie_reviews.db", query=query)

input_cols = ["reviewText"]

output_cols = {"key_aspects": "The key aspect of the movie discussed in the review"}


extracted_df = df.sem_extract(input_cols, output_cols, return_provenance=True)
filtered_extracted_df = extracted_df.sem_filter(
    "{key_aspects} are about the cinematography or displayed scenary.", return_provenance=True
)
print(filtered_extracted_df.head())
