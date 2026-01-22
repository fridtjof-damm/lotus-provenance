import os
import re
import inspect
from typing import Tuple

import pandas as pd
from datasets import load_dataset

import lotus
from lotus.models import LM


# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = os.getenv("LOTUS_BENCH_MODEL", "ollama/llama3.2:3b")
N_SAMPLE = int(os.getenv("IMDB_SAMPLE_N", "40"))
SEED = int(os.getenv("IMDB_SEED", "42"))

ALLOWED_ASPECTS = {"acting", "plot", "pacing", "dialogue", "cinematography", "sound", "other"}


# -----------------------------
# UTIL
# -----------------------------
def call_op(op, *args, **kwargs):
    sig = inspect.signature(op)
    allowed = set(sig.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return op(*args, **filtered_kwargs)


# -----------------------------
# LOTUS SETUP (LM only)
# -----------------------------
def configure_lotus() -> None:
    base = os.getenv("OLLAMA_API_BASE", "").strip()

    # hard-fix common broken values like "http://:11434"
    if (not base) or ("://:" in base) or (base in ["http://:11434", "https://:11434"]):
        base = "http://localhost:11434"
    if not base.startswith(("http://", "https://")):
        base = "http://localhost:11434"

    os.environ["OLLAMA_API_BASE"] = base

    lotus.settings.configure(enable_cache=False)
    lotus.settings.configure(lm=LM(model=MODEL_NAME))

    print(f"[OK] OLLAMA_API_BASE={os.environ['OLLAMA_API_BASE']}")
    print(f"[OK] LM configured: {MODEL_NAME}")


# -----------------------------
# DATA
# -----------------------------
def load_imdb(n: int) -> pd.DataFrame:
    ds = load_dataset("ajaykarthick/imdb-movie-reviews", split="train")
    df = ds.shuffle(seed=SEED).select(range(n)).to_pandas()

    df["sentiment"] = df["label"].map({0: "negative", 1: "positive"}).fillna("unknown")
    df["review"] = df["review"].astype(str)
    df["row_id"] = range(1, len(df) + 1)
    df["review_preview"] = df["review"].str.replace("\n", " ", regex=False).str.slice(0, 260)

    return df[["row_id", "sentiment", "review", "review_preview"]]


# -----------------------------
# HELPERS
# -----------------------------
def normalize_aspect(x: object) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z]+", "", s)

    synonym_map = {
        "visuals": "cinematography",
        "cinematics": "cinematography",
        "camera": "cinematography",
        "lighting": "cinematography",
        "editing": "cinematography",
        "vfx": "cinematography",
        "effects": "cinematography",
        "music": "sound",
        "audio": "sound",
        "score": "sound",
        "writing": "dialogue",
        "lines": "dialogue",
        "script": "dialogue",
        "story": "plot",
        "narrative": "plot",
        "characters": "plot",
        "boring": "pacing",
        "slow": "pacing",
        "dragging": "pacing",
        "pace": "pacing",
        "performance": "acting",
        "cast": "acting",
        "casting": "acting",
    }
    s = synonym_map.get(s, s)
    return s if s in ALLOWED_ASPECTS else "other"


def regex_pacing_complaint(text: str) -> bool:
    t = (text or "").lower()
    patterns = [
        r"\bslow\b",
        r"\bboring\b",
        r"\bdrag(ging)?\b",
        r"\bplodd(ing)?\b",
        r"\bdull\b",
        r"\btoo long\b",
        r"\bpacing\b",
    ]
    return any(re.search(p, t) for p in patterns)


# -----------------------------
# USE CASE: sem_map → (python filter) → sem_agg
# -----------------------------
def run_uc_imdb_map_filter_agg() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_imdb(N_SAMPLE)

    # 1) MAP: aspect
    tagged = call_op(
        df.sem_map,
        "Task: classify the SINGLE dominant complaint aspect.\n"
        "Pick exactly ONE label from:\n"
        "acting, plot, pacing, dialogue, cinematography, sound, other\n\n"
        "Rules:\n"
        "- Output ONLY the label (one word), lowercase.\n"
        "- Use 'other' ONLY if mostly praise OR no clear complaint.\n"
        "- If multiple complaints, pick the strongest negative one.\n"
        "- editing/visual effects/camera/lighting => cinematography\n\n"
        "Review: {review}",
        suffix="_aspect_raw",
    )
    tagged["_aspect"] = tagged["_aspect_raw"].apply(normalize_aspect)

    # 2) FILTER (deterministic)
    focused = tagged[
        (tagged["sentiment"] == "negative") | (tagged["review"].apply(regex_pacing_complaint))
    ].copy()

    # Safety: if filter ends up empty, just keep negatives (fallback)
    if focused.empty:
        focused = tagged[tagged["sentiment"] == "negative"].copy()

    # Optional: partition id (harmless)
    focused.loc[:, "_lotus_partition_id"] = focused["_aspect"].astype("category").cat.codes

    # Build per-group allowed ids list + group size (DON'T call it n_in_group)
    allowed_ids_map = focused.groupby("_aspect")["row_id"].apply(lambda s: "[" + ",".join(map(str, s.tolist())) + "]")
    group_size_map = focused.groupby("_aspect")["row_id"].size()

    focused["allowed_ids"] = focused["_aspect"].map(allowed_ids_map)
    focused["group_size"] = focused["_aspect"].map(group_size_map).astype(int)

    # 3) AGG: strict + grounded
    summary = call_op(
        focused.sem_agg,
        "You summarize ONE group where all rows share the same _aspect.\n"
        "Allowed Evidence IDs (use ONLY these): {allowed_ids}\n"
        "Group size: {group_size}\n\n"
        "Output up to 3 bullets.\n"
        "Each bullet MUST be exactly one line in this format:\n"
        "- Issue=<short>; Fix=<short>; Evidence=[row_id,row_id]\n\n"
        "Rules:\n"
        "1) Evidence must be ONLY from Allowed Evidence IDs.\n"
        "2) Do NOT mention any columns/fields other than the review content and row_id.\n"
        "3) If _aspect is 'other': do NOT invent issues. Write:\n"
        "   Issue=<why it is other>; Fix=<how to reclassify or what extra info needed>; Evidence=[...]\n"
        "4) If fewer distinct issues exist, output fewer bullets.\n\n"
        "Inputs:\n"
        "row_id={row_id}\n"
        "review={review_preview}",
        group_by=["_aspect"],
    )

    print("\n=== SAMPLE (focused rows) ===")
    show_cols = ["row_id", "sentiment", "_aspect", "review_preview"]
    print(focused[show_cols].head(12).to_string(index=False))

    print("\n=== SUMMARY BY ASPECT ===")
    print(summary.to_string(index=False))

    return focused, summary


if __name__ == "__main__":
    configure_lotus()
    run_uc_imdb_map_filter_agg()
