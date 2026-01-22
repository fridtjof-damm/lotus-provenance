import os
import sqlite3
import inspect
from typing import Tuple

import pandas as pd
import lotus
from lotus.models import LM

# -----------------------------
# CONFIG
# -----------------------------
SQLITE_FILE = os.getenv("LOTUS_BENCH_SQLITE_FILE", "bench_lotus.db")
MODEL_NAME = os.getenv("LOTUS_BENCH_MODEL", "ollama/llama3.2:3b")
# Ollama:
# export OLLAMA_API_BASE="http://localhost:11434"

# -----------------------------
# UTIL: call LOTUS op with only supported kwargs
# -----------------------------
def call_op(op, *args, **kwargs):
    """
    Calls a pandas accessor op (df.sem_map / df.sem_filter / df.sem_topk / df.sem_agg)
    but only passes kwargs that exist in the op signature (compat across versions).
    """
    sig = inspect.signature(op)
    allowed = set(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return op(*args, **filtered_kwargs)

# -----------------------------
# SQLITE SETUP (tickets)
# -----------------------------
def init_sqlite_db(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE tickets (
            id INTEGER PRIMARY KEY,
            title TEXT,
            description TEXT,
            priority TEXT,
            created_at TEXT
        )
        """
    )

    cur.executemany(
        "INSERT INTO tickets(title, description, priority, created_at) VALUES (?, ?, ?, ?)",
        [
            ("Login failing", "Users report password reset loop and cannot access accounts.", "high", "2026-01-20"),
            ("Slow dashboard", "Analytics dashboard loads slowly after recent release.", "medium", "2026-01-19"),
            ("Invoice mismatch", "Finance sees incorrect totals on monthly invoices.", "high", "2026-01-18"),
            ("Feature request", "Add dark mode support for mobile app.", "low", "2026-01-17"),
            ("Suspicious traffic", "Possible credential stuffing detected from multiple IPs.", "high", "2026-01-20"),
            ("DB timeouts", "Database connection pool exhausted causing sporadic 500s.", "high", "2026-01-16"),
            ("Phishing report", "Multiple employees reported suspicious emails with fake login links.", "medium", "2026-01-15"),
            ("Crash on startup", "iOS app crashes immediately after launch on iOS 17.2", "high", "2026-01-14"),
        ],
    )

    conn.commit()
    conn.close()

def load_tickets_sqlite(path: str) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT id, title, description, priority, created_at FROM tickets;", conn)
    conn.close()
    return df

# -----------------------------
# LOTUS SETUP
# -----------------------------
def configure_lotus() -> None:
    lotus.settings.configure(enable_cache=False)
    lotus.settings.configure(lm=LM(model=MODEL_NAME))

# -----------------------------
# USE CASE: SQL → map → filter → topk → agg(group_by)
# -----------------------------
def run_uc_sql_02_ticket_triage() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_tickets_sqlite(SQLITE_FILE)

    # 1) MAP: triage label
    triaged = call_op(
        df.sem_map,
        "For ticket with {title} and {description}, output ONLY one label from: "
        "bug/performance/security/feature. Return just the label.",
        suffix="_triage_label",
        return_provenance=True,
        provenance_col="id",
    )

    # 2) MAP: action suggestion
    triaged = call_op(
        triaged.sem_map,
        "Write a one-line triage action for {title} and {description}. "
        "Start with a verb. Max 12 words.",
        suffix="_triage_action",
        return_provenance=True,
        provenance_col="id",
    )

    # 3) FILTER: urgent items
    urgent = call_op(
        triaged.sem_filter,
        "Keep rows where {priority} is high OR {_triage_label} is security.",
    )

    # 4) TOPK: rank most urgent
    top_urgent = call_op(
        urgent.sem_topk,
        "Rank by urgency and risk. Security incidents and outages first. "
        "Use {title}, {description}, {priority}, {_triage_label}.",
        K=5,
        provenance=True,
        provenance_col="id",
    )

    # 5) AGG: grouped summary by triage label
    summary = call_op(
        top_urgent.sem_agg,
        "Summarize the key issues and suggested actions from {title}, {description}, {_triage_action}. "
        "Use 2-4 bullet points.",
        group_by=["_triage_label"],
        provenance=True,
        provenance_col="id",
    )

    print("\n=== TOP URGENT (ranked) ===")
    cols = ["id", "title", "priority", "_triage_label", "_triage_action"]
    print(top_urgent[cols].to_string(index=False))

    print("\n=== GROUPED SUMMARY ===")
    print(summary.to_string(index=False))

    return top_urgent, summary

if __name__ == "__main__":
    init_sqlite_db(SQLITE_FILE)
    configure_lotus()
    run_uc_sql_02_ticket_triage()
