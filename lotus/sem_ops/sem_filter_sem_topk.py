import pandas as pd
import lotus
from lotus.models import LM
import os
import subprocess


def get_wsl_host_ip():
    try:
        # This grabs the 'default' route which is usually the Windows host
        cmd = "ip route show | grep default | awk '{print $3}'"
        return subprocess.check_output(cmd, shell=True).decode().strip()
    except:
        return "127.0.0.1"

host_ip = get_wsl_host_ip()
print(f"Connecting to Ollama on Windows at: {host_ip}")

# Pass the custom API base to LiteLLM via LOTUS
lm = LM(model=f"ollama/llama3.2:3b")
# Tell LiteLLM exactly where the server is
os.environ["OLLAMA_API_BASE"] = f"http://{host_ip}:11434"

lotus.settings.configure(lm=lm)

data = {
    "Title": ["Update A", "Patch B", "News C", "Blog D", "Log E"],
    "Content": [
        "A critical zero-day vulnerability was found in the kernel.",
        "New UI updates for the mobile app including dark mode.",
        "Security researchers discover malware in popular library.",
        "How to bake sourdough bread in three easy steps.",
        "Refactoring the database connection pool for better performance."
    ]
}
df = pd.DataFrame(data)

security_news = df.sem_filter("{Content} is related to cybersecurity threats.", return_provenance=True)
top_news = security_news.sem_topk("Which {Content} describes the most urgent threat?", K=4, provenance=True, provenance_col= "provenance_id")
df = top_news.sem_agg("Summarize {Content}", group_by=["Title"], provenance=True, provenance_col= "provenance_id")
print(df)