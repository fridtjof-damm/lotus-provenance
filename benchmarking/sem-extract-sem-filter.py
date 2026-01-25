import pandas as pd
import lotus
from lotus.models import LM
import os
import subprocess

from examples.provenace_examples.sqllite_db import get_data_from_sql_as_dict

DB_NAME = "/mnt/c/imdb_benchmark/imdb_review.db"

data = get_data_from_sql_as_dict(DB_NAME)
if data:
    print("\nSample Data from Dictionary:")
    print(data[0])

def get_wsl_host_ip():
    try:
        # This grabs the 'default' route which is usually the Windows host
        cmd = "ip route show | grep default | awk '{print $3}'"
        return subprocess.check_output(cmd, shell=True).decode().strip()
    except:
        return "127.0.0.1"

host_ip = get_wsl_host_ip()
print(f"Connecting to Ollama on Windows at: {host_ip}")

# Tell LiteLLM exactly where the server is
os.environ["OLLAMA_API_BASE"] = f"http://{host_ip}:11434"

# Pass the custom API base to LiteLLM via LOTUS
lm = LM(model=f"ollama/llama3.2:3b", max_ctx_len = 128000, max_tokens = 512, rate_limit = None)

lotus.settings.configure(lm=lm)

df = pd.DataFrame(data[0:10])

# 2. sem_extract: Unified API to convert unstructured text into structured columns
# We define what we want to extract using natural language descriptions.
output_schema = {
    "Director_Name": "The full name of the movie director",
    "Runtime_Minutes": "The duration of the movie in minutes as an integer"
}

# This performs LLM-based extraction with optional quote-tracking for verification
structured_df = df.sem_extract(
    input_cols=["review"], 
    output_cols=output_schema, 
    extract_quotes=True
)

# 3. sem_filter: Use the newly created structured data for complex reasoning
# We benchmark how well Lotus handles a predicate involving extracted fields.
final_df = structured_df.sem_filter(
    "The {Director_Name} is known for large-scale blockbusters and the {Runtime_Minutes} is over 2 hours"
)
print(final_df)
