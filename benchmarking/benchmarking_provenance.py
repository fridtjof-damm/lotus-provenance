import gc
import json
import random
import statistics
import time
import tracemalloc
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

import pandas as pd

import lotus
from benchmarking.mock_lm import MockLM
from lotus.data_connectors import DataConnector

BENCHMARKING_MODEL = "ollama/llama3.2:3b"
BENCHMARKING_SYSTEM = ""
RUN_ID = str(uuid.uuid4())[:8]


def benchmark_provenance_overhead(usecase_id, n_iterations=3):
    """Benchmark the time and memory overhead of provenance tracking in LOTUS."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_df = args[0]
            current_df = original_df

            default_row_limit = 100
            row_limit = kwargs.pop("row_limit", default_row_limit)
            dataset_name = original_df.attrs.get("dataset_name", "unknown_dataset")

            if row_limit is not None:
                current_df = original_df.head(row_limit)

            actual_rows = len(current_df)
            debug = kwargs.pop("debug", True)
            raw_data = {"vanilla": [], "provenance": []}

            print(f" Recording LLM cache for {usecase_id}: {actual_rows} rows")
            cache_file = f"cache/{usecase_id}.json"
            mock_lm = MockLM(BENCHMARKING_MODEL, cache_file, mode="record")
            lotus.settings.configure(lm=mock_lm)

            func(current_df, *args[1:], **kwargs, use_prov=False)
            func(current_df, *args[1:], **kwargs, use_prov=True)

            print(
                f"\nStarting benchmark for use case: {usecase_id} with {n_iterations} iterations and {actual_rows} rows...",
                flush=True,
            )
            mock_lm.mode = "replay"
            lotus.settings.configure(lm=mock_lm)

            for i in range(n_iterations):
                modes = ["vanilla", "provenance"]
                random.shuffle(modes)

                print(f"  Iteration {i+1}/{n_iterations}... ", end="", flush=True)
                for mode in modes:
                    # set provenance flag based on mode
                    prov_flag = mode == "provenance"
                    gc.collect()
                    tracemalloc.start()

                    s_wall, s_cpu = time.perf_counter(), time.process_time()
                    func(current_df, *args[1:], **kwargs, use_prov=prov_flag)
                    e_cpu, e_wall = time.process_time(), time.perf_counter()

                    _, peak_mem = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    time.sleep(3)
                    raw_data[mode].append(
                        {
                            "wall": e_wall - s_wall,
                            "cpu": e_cpu - s_cpu,
                            "mem": peak_mem / 10**6,
                        }
                    )
                print(
                    f"Iteration {i+1} done.",
                )
            time.sleep(5)

            # Calculate statistics
            stats = {}
            for idx, m in enumerate(["vanilla", "provenance"]):
                walls = [run["wall"] for run in raw_data[m]]
                cpus = [run["cpu"] for run in raw_data[m]]
                mems = [run["mem"] for run in raw_data[m]]

                stats[m] = {
                    "wall_mean": round(statistics.mean(walls), 4),
                    "wall_std": round(statistics.stdev(walls), 4) if n_iterations > 1 else 0.0,
                    "cpu_mean": round(statistics.mean(cpus), 4),
                    "cpu_std": round(statistics.stdev(cpus), 4) if n_iterations > 1 else 0.0,
                    "mem_mean": round(statistics.mean(mems), 2),
                    "mem_std": round(statistics.stdev(mems), 2) if n_iterations > 1 else 0.0,
                }

            log_entry = {
                "run_id": RUN_ID,
                "usecase_id": usecase_id,
                "timestamp": datetime.now().isoformat(),
                "function": func.__name__,
                "iterations": n_iterations,
                "results": stats,
                "metadata": {
                    "model": BENCHMARKING_MODEL,
                    "n_rows": actual_rows,
                    "dataset": dataset_name,
                    "mock_lm": True,
                    "system": BENCHMARKING_SYSTEM,
                },
            }

            Path("results").mkdir(exist_ok=True)
            with open(f"results/raw_data/{usecase_id}.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if debug:
                print_summary(func.__name__, stats, n_iterations)

            return stats

        return wrapper

    return decorator


def set_benchmark_env():
    """Configure LOTUS specific benchmarking settings."""
    # disable caching
    lotus.settings.configure(enable_cache=False)
    lm = lotus.models.LM(model=BENCHMARKING_MODEL)
    lotus.settings.configure(lm=lm)


def get_data_sql(db_path):
    """
    Select all columns from database table.
    Set query to load data and the database path.

    Update: Reading data from VIEW that only stores 200 char long reviews

    Args:
        db_path (str): Path to the database.
        Example: "sqlite:///path/to/database.db"
    """
    query = "SELECT * FROM short_text_reviews;"
    data = DataConnector.load_from_db(db_path, query=query)
    data.attrs["dataset_name"] = db_path.split("/")[-1]
    return data


def get_csv_data(file_path):
    data = pd.read_csv(file_path)
    data.attrs["dataset_name"] = file_path.split("/")[-1]
    return data


def print_summary(func_name, stats, n_iterations):
    v, p = stats["vanilla"], stats["provenance"]
    print(f"\n---- Benchmark: {func_name} (Avg over {n_iterations} runs) ----")
    print(
        f"Wall Time: {v['wall_mean']:.2f}s vs {p['wall_mean']:.2f}s | OH: {(p['wall_mean'] / v['wall_mean']) - 1:+.2%}"
    )
    print(f"CPU Time:  {v['cpu_mean']:.2f}s vs {p['cpu_mean']:.2f}s | OH: {(p['cpu_mean'] / v['cpu_mean']) - 1:+.2%}")
    print(f"Memory:    {v['mem_mean']:.2f}MB vs {p['mem_mean']:.2f}MB | OH: {(p['mem_mean'] / v['mem_mean']) - 1:+.2%}")


# ==========================================
#   Use Case Section
# ==========================================

# ==========================================
#   Combined Use Cases
# ==========================================


@benchmark_provenance_overhead(usecase_id="01-UC-EXTRACT-FILTER", n_iterations=50)
def extract_filter_movie_reviews(df, use_prov=False, debug=False):
    """
    01:
    SQL Use case with sem_extract and sem_filter on movie reviews dataset.
    Depends on sqlite file in created in db_examples/sql_extract_filter.py
    """
    # setup

    # define extract parameters
    input_cols = ["review"]
    output_cols = {"key_aspects": "The key aspect of the movie discussed in the review."}

    # run sem_ops transform
    extracted_df = df.sem_extract(input_cols, output_cols, provenance=use_prov)
    filtered_extracted_df = extracted_df.sem_filter(
        "{key_aspects} are positive regarding the movie production.", provenance=use_prov
    )
    if debug:
        print(filtered_extracted_df.head())


@benchmark_provenance_overhead(usecase_id="02-UC-FILTER-AGG", n_iterations=50)
def filter_agg_movie_reviews(df, use_prov=False, debug=False):
    bench_df = df.sem_filter("{review} mentions cinematography or visual style in detail", provenance=use_prov).sem_agg(
        "Summarize the common visual critiques found in these {review}s", provenance=use_prov
    )
    if debug:
        print(bench_df.head())


@benchmark_provenance_overhead(usecase_id="03-UC-JOIN-FILTER", n_iterations=50)
def join_filter_movie_reviews(df, use_prov=False, debug=False):
    categories = {
        "category": [
            "technical and analytical",
            "emotional and subjective",
            "humorous and sarcastic",
            "brief and casual",
            "professional film criticism",
        ]
    }
    categories = pd.DataFrame(categories)
    joined_df = df.sem_join(
        categories, "{review} primarily falls under the {category} style of writing", provenance=use_prov
    )
    filtered = joined_df.sem_filter("The {review} expresses a negative sentiment toward the film", provenance=use_prov)
    if debug:
        print(filtered.head())


@benchmark_provenance_overhead(usecase_id="04-UC-TOPK-MAP", n_iterations=50)
def topk_map_movie_reviews(df, use_prov=False, debug=False):
    top_reviews = df.sem_topk("Which {review} shows the most extreme positive enthusiasm?", K=5, provenance=use_prov)
    reasoned_reviews = top_reviews.sem_map(
        "Given the {review}, list the top 3 adjectives that express the user's joy. Output: [adj1, adj2, adj3]",
        suffix="Extracted_Keywords",
        provenance=use_prov,
    )
    if debug:
        print(reasoned_reviews.head())


# ==========================================
#   Single operator Use Cases
# ==========================================


@benchmark_provenance_overhead(usecase_id="05-UC-EXTRACT", n_iterations=50)
def extract_movie_reviews(df, use_prov=False, debug=True):
    input_cols = ["review"]
    output_cols = {"key_aspects": "The key aspects of the movie plot discussed in the review."}
    extracted_df = df.sem_extract(input_cols, output_cols, provenance=use_prov)
    if debug:
        print(extracted_df.head())


@benchmark_provenance_overhead(usecase_id="06-UC-FILTER", n_iterations=50)
def filter_movie_reviews(df, use_prov=False, debug=True):
    filtered_df = df.sem_filter("The {review} is positive about the movie's storyline?", provenance=use_prov)
    if debug:
        print(filtered_df.head())


@benchmark_provenance_overhead(usecase_id="07-UC-AGG", n_iterations=50)
def agg_movie_reviews(df, use_prov=False, debug=True):
    summary = df.sem_agg(
        "You summarize ONE group of reviews (same {sentiment}).\n"
        "Output up to 3 bullets, each exactly:\n"
        "- Theme=<short>; Evidence=<short phrase>\n"
        "Use only {review}.",
        group_by=["sentiment"],
        provenance=use_prov,
    )
    if debug:
        print(summary.head())


#@benchmark_provenance_overhead(usecase_id="08-UC-JOIN", n_iterations=50)
def join_movie_reviews(df, use_prov=False, debug=True):
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
    joined = df.sem_join(
        df_categories,
        "Choose the single best {category:right} for this review. "
        "If it matches {definition:right}, then assign it.\n\n"
        "Review: {review:left}",
        provenance=use_prov,
    )
    if debug:
        print(joined.head())
    return joined


@benchmark_provenance_overhead(usecase_id="09-UC-MAP", n_iterations=50)
def map_movie_reviews(df, use_prov=False, debug=False):
    mapped = df.sem_map(
        "From this IMDb review, output exactly ONE label from:\n"
        "acting/plot/pacing/dialogue/cinematography/sound/other\n"
        "Rules:\n"
        "- If the review is mostly positive or no clear single aspect => other\n"
        "- Return ONLY the label (one word), lowercase.\n"
        "Review: {review}",
        suffix="_aspect",
        provenance=use_prov,
    )
    if debug:
        print(mapped.head())


@benchmark_provenance_overhead(usecase_id="10-UC-TOPK", n_iterations=1)
def top_k_movie_reviews(df, use_prov=False, debug=False):
    topk = df.sem_topk(
        "Rank by strongest positive enthusiasm and excitement.\n"
        "Prefer reviews with intense praise, strong emotion, and superlatives.\n"
        "Use ONLY the review text: {review}",
        K=5,
        provenance=use_prov,
    )
    if debug:
        print(topk.head())


if __name__ == "__main__":
    set_benchmark_env()
    # Set correct db path and query first to load from the correct database table
    df = get_data_sql("sqlite:///../examples/db_examples/imdb_reviews.db")
    row_limits = [100, 500]
    for limit in row_limits:
        """        extract_filter_movie_reviews(df, debug=True, row_limit=limit)
                time.sleep(5)
                filter_agg_movie_reviews(df, debug=True, row_limit=limit)
                time.sleep(5)
                join_filter_movie_reviews(df, debug=True, row_limit=limit)
                time.sleep(5)
                topk_map_movie_reviews(df, debug=True, row_limit=limit)
                time.sleep(5)"""
        extract_movie_reviews(df, debug=True, row_limit=limit)
        time.sleep(5)
        filter_movie_reviews(df, debug=True, row_limit=limit)
        time.sleep(5)
        agg_movie_reviews(df, debug=True, row_limit=limit)
        time.sleep(5)
        join_movie_reviews(df, debug=True, row_limit=limit)
        time.sleep(5)
        map_movie_reviews(df, debug=True, row_limit=limit)
        time.sleep(5)
        """
        top_k_movie_reviews(df, debug=True, row_limit=5)
        joined_result = join_movie_reviews(df[:10],use_prov=True,debug=True)
        joined_result.to_csv("join_result.csv", index=False)"""

