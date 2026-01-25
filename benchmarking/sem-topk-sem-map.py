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

top_reviews = df.sem_topk("Which {review} shows the most extreme positive enthusiasm?", K=5)
reasoned_reviews = top_reviews.sem_map(
    "Given the {review}, list the top 3 adjectives that express the user's joy. Output: [adj1, adj2, adj3]", 
    suffix="Extracted_Keywords"
)

print(reasoned_reviews)