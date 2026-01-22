import os
import gc
import time
import json
import random
import sqlite3
import tracemalloc
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

import lotus
from lotus.models import LM

# -----------------------------
# CONFIG
# -----------------------------
# ✅ Ollama model by default
MODEL_NAME = os.getenv("LOTUS_BENCH_MODEL", "ollama/llama3.2:3b")

# ✅ For OpenAI models (optional): set OPENAI_API_KEY + MODEL_NAME=gpt-...
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RUNS_PER_CASE = int(os.getenv("LOTUS_BENCH_RUNS", "3"))          # repetitions per mode per case
SQLITE_PATH = os.getenv("LOTUS_BENCH_SQLITE", "bench_lotus.db")  # sqlite path

# Benchmark hygiene
WARMUP = int(os.getenv("LOTUS_BENCH_WARMUP", "1"))               # warmup runs per mode per case (not recorded)
COOLDOWN_SEC = float(os.getenv("LOTUS_BENCH_COOLDOWN_SEC", "0.0"))
SHUFFLE_MODES = os.getenv("LOTUS_BENCH_SHUFFLE", "1") == "1"

# Optional: debug baseline only (helps isolate monkeypatch issues)
ONLY_BASELINE = os.getenv("LOTUS_ONLY_BASELINE", "0") == "1"

# Outputs
OUT_DIR = os.getenv("LOTUS_BENCH_OUT_DIR", ".")
RAW_RUNS_CSV = os.path.join(OUT_DIR, "bench_runs_raw.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "bench_summary.csv")
OVERHEAD_CSV = os.path.join(OUT_DIR, "bench_overhead.csv")
META_JSON = os.path.join(OUT_DIR, "bench_meta.json")


# -----------------------------
# PROVENANCE RECORD
# -----------------------------
@dataclass
class ProvEvent:
    op: str
    langex: str
    rows_in: Optional[int] = None
    rows_out: Optional[int] = None
    cols_used: Optional[List[str]] = None
    duration_ms: Optional[float] = None
    model: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


PROV_LOG: List[ProvEvent] = []


def _get_model_name() -> str:
    try:
        lm = lotus.settings.lm
        return getattr(lm, "model", str(lm))
    except Exception:
        return "unknown"


def _append_prov(df: pd.DataFrame, event: ProvEvent) -> None:
    # Store provenance events on the output dataframe (pipeline-level provenance)
    if not hasattr(df, "attrs"):
        return
    df.attrs.setdefault("_prov", [])
    df.attrs["_prov"].append(asdict(event))


# -----------------------------
# MONKEYPATCH WRAPPERS
# -----------------------------
_ORIGINALS: Dict[Tuple[Any, str], Any] = {}


def enable_provenance_monkeypatch() -> None:
    """
    Wrap LOTUS dataframe accessors to record provenance + timing.
    Benchmark-oriented; no LOTUS core modifications required.
    """
    global _ORIGINALS

    # Import inside to avoid import issues at module import time.
    # NOTE: if your LOTUS version changed paths, this is the only section you might need to adjust.
    from lotus.sem_ops.sem_filter import SemFilterDataframe
    from lotus.sem_ops.sem_join import SemJoinDataframe
    from lotus.sem_ops.sem_map import SemMapDataframe
    from lotus.sem_ops.sem_topk import SemTopKDataframe
    from lotus.sem_ops.sem_agg import SemAggDataframe

    def safe_parse_cols(expr: str) -> List[str]:
        try:
            return lotus.nl_expression.parse_cols(expr)
        except Exception:
            return []

    def wrap(op_name: str, cls, extract_langex: Callable[..., str], extract_cols: Callable[..., List[str]]):
        key = (cls, "__call__")
        if key in _ORIGINALS:
            return

        orig = cls.__call__
        _ORIGINALS[key] = orig

        def patched(self, *args, **kwargs):
            start = time.perf_counter()

            # rows_in: best effort
            try:
                rows_in = len(self._obj)
            except Exception:
                rows_in = None

            # identify langex and cols (best effort)
            try:
                langex = extract_langex(*args, **kwargs)
            except Exception:
                langex = "<unknown>"

            try:
                cols_used = extract_cols(*args, **kwargs)
            except Exception:
                cols_used = None

            out = orig(self, *args, **kwargs)

            end = time.perf_counter()
            duration_ms = (end - start) * 1000.0

            # out can be (df, stats) in some ops
            df_out = out[0] if isinstance(out, tuple) else out
            try:
                rows_out = len(df_out)
            except Exception:
                rows_out = None

            ev = ProvEvent(
                op=op_name,
                langex=langex,
                rows_in=rows_in,
                rows_out=rows_out,
                cols_used=cols_used,
                duration_ms=duration_ms,
                model=_get_model_name(),
            )
            PROV_LOG.append(ev)

            if isinstance(df_out, pd.DataFrame):
                _append_prov(df_out, ev)

            return out

        cls.__call__ = patched

    wrap(
        "sem_filter",
        SemFilterDataframe,
        extract_langex=lambda user_instruction, **kw: user_instruction,
        extract_cols=lambda user_instruction, **kw: safe_parse_cols(user_instruction),
    )
    wrap(
        "sem_join",
        SemJoinDataframe,
        extract_langex=lambda other, join_instruction, **kw: join_instruction,
        extract_cols=lambda other, join_instruction, **kw: safe_parse_cols(join_instruction),
    )
    wrap(
        "sem_map",
        SemMapDataframe,
        extract_langex=lambda user_instruction, **kw: user_instruction,
        extract_cols=lambda user_instruction, **kw: safe_parse_cols(user_instruction),
    )
    wrap(
        "sem_topk",
        SemTopKDataframe,
        extract_langex=lambda user_instruction, K, **kw: user_instruction,
        extract_cols=lambda user_instruction, K, **kw: safe_parse_cols(user_instruction),
    )
    wrap(
        "sem_agg",
        SemAggDataframe,
        extract_langex=lambda user_instruction, **kw: user_instruction,
        extract_cols=lambda user_instruction, **kw: safe_parse_cols(user_instruction),
    )


def disable_provenance_monkeypatch() -> None:
    global _ORIGINALS
    for (cls, name), orig in list(_ORIGINALS.items()):
        setattr(cls, name, orig)
    _ORIGINALS = {}


# -----------------------------
# SQLITE SETUP (4+ use cases must read from SQL)
# -----------------------------
def init_sqlite_db(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE reviews (
            id INTEGER PRIMARY KEY,
            product TEXT,
            review_text TEXT,
            rating INTEGER
        )
    """)
    cur.executemany(
        "INSERT INTO reviews(product, review_text, rating) VALUES (?, ?, ?)",
        [
            ("Laptop", "This laptop is absolutely fantastic! Super fast and battery lasts all day.", 5),
            ("Shoes", "Not bad but sizing was weird and comfort is average.", 3),
            ("Headphones", "Amazing sound quality! Best headphones I've ever used.", 5),
            ("Camera", "Picture quality is fine but low-light performance is disappointing.", 2),
            ("Watch", "Beautiful design and premium feel. Totally worth it.", 5),
        ],
    )

    cur.execute("""
        CREATE TABLE tickets (
            id INTEGER PRIMARY KEY,
            title TEXT,
            description TEXT,
            priority TEXT
        )
    """)
    cur.executemany(
        "INSERT INTO tickets(title, description, priority) VALUES (?, ?, ?)",
        [
            ("Login failing", "Users report password reset loop and cannot access accounts.", "high"),
            ("Slow dashboard", "Analytics dashboard loads slowly after recent release.", "medium"),
            ("Invoice mismatch", "Finance sees incorrect totals on monthly invoices.", "high"),
            ("Feature request", "Add dark mode support for mobile app.", "low"),
        ],
    )

    cur.execute("""
        CREATE TABLE products (
            sku TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            category TEXT
        )
    """)
    cur.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        [
            ("SKU1", "Gaming Laptop", "High performance laptop for gaming and AI workloads", "electronics"),
            ("SKU2", "Noise Cancelling Headphones", "Over-ear ANC headphones with deep bass", "electronics"),
            ("SKU3", "Running Shoes", "Lightweight shoes for long-distance running", "apparel"),
            ("SKU4", "Espresso Machine", "Compact espresso machine for home baristas", "home"),
        ],
    )

    cur.execute("""
        CREATE TABLE categories (
            category TEXT PRIMARY KEY,
            description TEXT
        )
    """)
    cur.executemany(
        "INSERT INTO categories VALUES (?, ?)",
        [
            ("electronics", "Devices, computers, audio"),
            ("apparel", "Clothing and footwear"),
            ("home", "Home appliances and kitchen"),
        ],
    )

    conn.commit()
    conn.close()


def read_sql(query: str, path: str) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


# -----------------------------
# 10 USE CASES (mix in-memory + SQLite)
# -----------------------------
def case_1_filter_reviews_sql() -> pd.DataFrame:
    df = read_sql("SELECT product, review_text, rating FROM reviews", SQLITE_PATH)
    return df.sem_filter("Keep rows where {review_text} is clearly positive.")


def case_2_topk_reviews_sql() -> pd.DataFrame:
    df = read_sql("SELECT product, review_text, rating FROM reviews", SQLITE_PATH)
    df_pos = df.sem_filter("Keep rows where {review_text} is clearly positive.")
    return df_pos.sem_topk("Return the {review_text} that sounds most enthusiastic.", K=2)


def case_3_map_ticket_summary_sql() -> pd.DataFrame:
    df = read_sql("SELECT id, title, description, priority FROM tickets", SQLITE_PATH)
    return df.sem_map("Summarize {description} in one concise sentence.", suffix="_summary")


def case_4_join_products_categories_sql() -> pd.DataFrame:
    products = read_sql("SELECT sku, name, description, category FROM products", SQLITE_PATH)
    categories = read_sql("SELECT category, description FROM categories", SQLITE_PATH)
    return products.sem_join(categories, "The {description:left} belongs to the {category:right}.")


def case_5_filter_in_memory_movies() -> pd.DataFrame:
    df = pd.DataFrame({
        "title": ["Interstellar", "The Godfather", "Apollo 13"],
        "overview": [
            "A team travels through a wormhole in space in an attempt to ensure humanity's survival.",
            "A mafia family drama about power and loyalty.",
            "A space mission suffers critical failures and must return safely."
        ],
        "rating": [8.6, 9.2, 7.6],
    })
    return df.sem_filter("Keep rows where {overview} is about science fiction or space.")


def case_6_map_in_memory_sentiment() -> pd.DataFrame:
    df = pd.DataFrame({
        "text": [
            "I love this product, it works perfectly!",
            "It’s okay, not great, not terrible.",
            "Worst purchase ever."
        ]
    })
    return df.sem_map("Label sentiment of {text} as Positive/Neutral/Negative. Answer in one word.", suffix="_sent")


def case_7_agg_in_memory_notes() -> pd.DataFrame:
    df = pd.DataFrame({
        "note": [
            "We shipped v1 of the API.",
            "Customer reported intermittent 500 errors.",
            "We added caching and reduced latency.",
        ]
    })
    return df.sem_agg("Create a short weekly update from {note}.")


def case_8_join_in_memory_courses_skills() -> pd.DataFrame:
    courses = pd.DataFrame({
        "Course": ["Intro to SQL", "Deep Learning", "Data Viz with Power BI"],
        "Desc": ["SQL basics for analytics", "Neural networks and training", "Dashboards, DAX, KPIs"]
    })
    skills = pd.DataFrame({"Skill": ["SQL", "Machine Learning", "Power BI"]})
    return courses.sem_join(skills, "Taking {Desc:left} helps learn {Skill:right}.")


def case_9_filter_then_map_pipeline() -> pd.DataFrame:
    df = pd.DataFrame({
        "doc": [
            "The system crashed due to a null pointer exception.",
            "Great performance improvements in the new release!",
            "Security vulnerability found in dependency."
        ]
    })
    filtered = df.sem_filter("Keep rows where {doc} is about a technical problem or incident.")
    return filtered.sem_map(
        "Classify the incident type from {doc} as one of: bug/performance/security.",
        suffix="_type"
    )


def case_10_topk_in_memory_titles() -> pd.DataFrame:
    df = pd.DataFrame({
        "title": ["AI guide", "ML tutorial", "Data science intro", "Cooking tips"],
        "desc": [
            "A practical AI guide for beginners",
            "Hands-on ML tutorial with examples",
            "Intro to data science concepts",
            "Healthy cooking tips and recipes"
        ]
    })
    return df.sem_topk("Rank by how relevant {desc} is for learning AI.", K=3)


USE_CASES: List[Tuple[str, Callable[[], pd.DataFrame]]] = [
    ("case_1_filter_reviews_sql", case_1_filter_reviews_sql),
    ("case_2_topk_reviews_sql", case_2_topk_reviews_sql),
    ("case_3_map_ticket_summary_sql", case_3_map_ticket_summary_sql),
    ("case_4_join_products_categories_sql", case_4_join_products_categories_sql),
    ("case_5_filter_in_memory_movies", case_5_filter_in_memory_movies),
    ("case_6_map_in_memory_sentiment", case_6_map_in_memory_sentiment),
    ("case_7_agg_in_memory_notes", case_7_agg_in_memory_notes),
    ("case_8_join_in_memory_courses_skills", case_8_join_in_memory_courses_skills),
    ("case_9_filter_then_map_pipeline", case_9_filter_then_map_pipeline),
    ("case_10_topk_in_memory_titles", case_10_topk_in_memory_titles),
]


# -----------------------------
# BENCH RUNNER
# -----------------------------
@dataclass
class BenchResult:
    case: str
    mode: str  # "baseline" or "prov"
    iter_idx: int
    wall_ms: float
    cpu_ms: float
    peak_mem_mb: float
    rows_out: Optional[int]
    prov_events: int


def _run_once(fn: Callable[[], pd.DataFrame]) -> Tuple[Any, float, float, float]:
    """
    Runs fn once, measuring wall, cpu, and peak memory.
    Returns (output, wall_ms, cpu_ms, peak_mem_mb).
    """
    gc.collect()
    tracemalloc.start()

    s_wall = time.perf_counter()
    s_cpu = time.process_time()

    out = fn()

    e_cpu = time.process_time()
    e_wall = time.perf_counter()

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_ms = (e_wall - s_wall) * 1000.0
    cpu_ms = (e_cpu - s_cpu) * 1000.0
    peak_mem_mb = peak / 1e6
    return out, wall_ms, cpu_ms, peak_mem_mb


def _configure_lotus_lm() -> None:
    """
    Configure LOTUS to use either:
    - Ollama (default) if MODEL_NAME startswith 'ollama/'
    - OpenAI if not, requiring OPENAI_API_KEY
    """
    model = (MODEL_NAME or "").strip()

    # Always disable cache so benchmark measures real overhead
    lotus.settings.configure(enable_cache=False)

    if model.startswith("ollama/"):
        # Hard-fix common broken values like "http://:11434"
        base = os.getenv("OLLAMA_API_BASE", "").strip()
        if (not base) or ("://:" in base) or (base in ["http://:11434", "https://:11434"]):
            base = "http://localhost:11434"
        if not base.startswith(("http://", "https://")):
            base = "http://localhost:11434"
        os.environ["OLLAMA_API_BASE"] = base

        lotus.settings.configure(lm=LM(model=model))
        print(f"[OK] OLLAMA_API_BASE={os.environ['OLLAMA_API_BASE']}")
        print(f"[OK] LM configured (Ollama): {model}")
        return

    # OpenAI path (optional)
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set, and LOTUS_BENCH_MODEL is not an ollama/* model.")
    lotus.settings.configure(lm=LM(model=model, api_key=OPENAI_API_KEY))
    print(f"[OK] LM configured (OpenAI): {model}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Configure LOTUS LM (Ollama by default)
    _configure_lotus_lm()

    # Init sqlite
    init_sqlite_db(SQLITE_PATH)

    meta = {
        "model": MODEL_NAME,
        "runs_per_case": RUNS_PER_CASE,
        "warmup": WARMUP,
        "cooldown_sec": COOLDOWN_SEC,
        "shuffle_modes": SHUFFLE_MODES,
        "only_baseline": ONLY_BASELINE,
        "sqlite_path": SQLITE_PATH,
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    results: List[BenchResult] = []

    for case_name, case_fn in USE_CASES:
        # Warmup (not recorded)
        for _ in range(WARMUP):
            disable_provenance_monkeypatch()
            _run_once(case_fn)

            if not ONLY_BASELINE:
                enable_provenance_monkeypatch()
                _run_once(case_fn)

        # Measured runs
        for i in range(RUNS_PER_CASE):
            modes = ["baseline"] if ONLY_BASELINE else ["baseline", "prov"]
            if SHUFFLE_MODES and not ONLY_BASELINE:
                random.shuffle(modes)

            for mode in modes:
                PROV_LOG.clear()

                if mode == "prov":
                    enable_provenance_monkeypatch()
                else:
                    disable_provenance_monkeypatch()

                out, wall_ms, cpu_ms, peak_mb = _run_once(case_fn)

                df_out = out[0] if isinstance(out, tuple) else out
                rows_out = len(df_out) if isinstance(df_out, pd.DataFrame) else None

                results.append(
                    BenchResult(
                        case=case_name,
                        mode=mode,
                        iter_idx=i,
                        wall_ms=wall_ms,
                        cpu_ms=cpu_ms,
                        peak_mem_mb=peak_mb,
                        rows_out=rows_out,
                        prov_events=len(PROV_LOG),
                    )
                )

                if COOLDOWN_SEC > 0:
                    time.sleep(COOLDOWN_SEC)

    # Always restore originals after benchmark
    disable_provenance_monkeypatch()

    # Raw runs
    res_df = pd.DataFrame([asdict(x) for x in results])
    res_df.to_csv(RAW_RUNS_CSV, index=False)

    # Summary
    summary = (
        res_df.groupby(["case", "mode"])
        .agg(
            avg_wall_ms=("wall_ms", "mean"),
            std_wall_ms=("wall_ms", "std"),
            avg_cpu_ms=("cpu_ms", "mean"),
            std_cpu_ms=("cpu_ms", "std"),
            avg_peak_mem_mb=("peak_mem_mb", "mean"),
            std_peak_mem_mb=("peak_mem_mb", "std"),
            rows_out=("rows_out", "max"),
            prov_events=("prov_events", "mean"),
        )
        .reset_index()
        .sort_values(["case", "mode"])
    )

    summary.to_csv(SUMMARY_CSV, index=False)

    # Overhead vs baseline (per case) — only if prov exists
    if not ONLY_BASELINE:
        base = summary[summary["mode"] == "baseline"][["case", "avg_wall_ms", "avg_cpu_ms", "avg_peak_mem_mb"]].rename(
            columns={
                "avg_wall_ms": "baseline_wall_ms",
                "avg_cpu_ms": "baseline_cpu_ms",
                "avg_peak_mem_mb": "baseline_mem_mb",
            }
        )
        prov = summary[summary["mode"] == "prov"][["case", "avg_wall_ms", "avg_cpu_ms", "avg_peak_mem_mb"]].rename(
            columns={
                "avg_wall_ms": "prov_wall_ms",
                "avg_cpu_ms": "prov_cpu_ms",
                "avg_peak_mem_mb": "prov_mem_mb",
            }
        )

        overhead = base.merge(prov, on="case")
        overhead["wall_overhead_ms"] = overhead["prov_wall_ms"] - overhead["baseline_wall_ms"]
        overhead["wall_overhead_pct"] = (overhead["wall_overhead_ms"] / overhead["baseline_wall_ms"]) * 100.0
        overhead["cpu_overhead_ms"] = overhead["prov_cpu_ms"] - overhead["baseline_cpu_ms"]
        overhead["cpu_overhead_pct"] = (overhead["cpu_overhead_ms"] / overhead["baseline_cpu_ms"]) * 100.0
        overhead["mem_overhead_mb"] = overhead["prov_mem_mb"] - overhead["baseline_mem_mb"]
        overhead["mem_overhead_pct"] = (overhead["mem_overhead_mb"] / overhead["baseline_mem_mb"]) * 100.0

        overhead.to_csv(OVERHEAD_CSV, index=False)

        print("\n=== BENCH SUMMARY (avg over runs) ===")
        print(summary.to_string(index=False))

        print("\n=== OVERHEAD (prov vs baseline) ===")
        print(overhead.sort_values("wall_overhead_pct", ascending=False).to_string(index=False))

        print(f"\nSaved: {SUMMARY_CSV}, {OVERHEAD_CSV}, {RAW_RUNS_CSV}, {META_JSON}")
    else:
        print("\n=== BENCH SUMMARY (baseline only) ===")
        print(summary.to_string(index=False))
        print(f"\nSaved: {SUMMARY_CSV}, {RAW_RUNS_CSV}, {META_JSON}")


if __name__ == "__main__":
    main()
