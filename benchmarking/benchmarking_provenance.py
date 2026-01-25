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
from lotus.data_connectors import DataConnector

BENCHMARKING_MODEL = "ollama/llama3.2:3b"
RUN_ID = str(uuid.uuid4())[:8]


def benchmark_provenance_overhead(usecase_id, n_iterations=5):
    """Benchmark the time and memory overhead of provenance tracking in LOTUS."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            debug = kwargs.pop("debug", True)
            raw_data = {"vanilla": [], "provenance": []}

            print(f"\nStarting benchmark for use case {usecase_id} with {n_iterations} iterations...", flush=True)

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
                    func(*args, **kwargs, use_prov=prov_flag)
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
                    " done.",
                )
            time.sleep(5)

            # Calculate statistics
            stats = {}
            for m in ["vanilla", "provenance"]:
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
                "metadata": {"model": BENCHMARKING_MODEL},
            }

            Path("results").mkdir(exist_ok=True)
            with open("results/provenance_benchmarks.jsonl", "a") as f:
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
    # lm = lotus.models.LM(model="ollama/llama3.1:8b")
    # lm = lotus.models.LM(model="gpt-4.1-nano")
    lm = lotus.models.LM(model=BENCHMARKING_MODEL)
    lotus.settings.configure(lm=lm)


def get_movie_review_data(db_path, limit=10):
    # select all columns from movie_reviews table with a limit
    query = f"SELECT * FROM movie_reviews LIMIT {limit};"
    df = DataConnector.load_from_db(db_path, query=query)
    return df


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


@benchmark_provenance_overhead(usecase_id="UC-01", n_iterations=10)
def extract_filter_movie_reviews(db_path, use_prov=False, debug=False):
    """
    01:
    SQL Use case with sem_extract and sem_filter on movie reviews dataset.
    Depends on sqlite file in created in db_examples/sql_extract_filter.py

    Source: https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews
    """
    # setup
    query = "SELECT * FROM movie_reviews LIMIT 10;"
    df = DataConnector.load_from_db(db_path, query=query)

    # define extract parameters
    input_cols = ["reviewText"]
    output_cols = {"key_aspects": "The key aspect of the movie discussed in the review."}

    # run sem_ops transform
    extracted_df = df.sem_extract(input_cols, output_cols, return_provenance=use_prov)
    filtered_extracted_df = extracted_df.sem_filter(
        "{key_aspects} are about the cinematography or the displayed scenary.", return_provenance=use_prov
    )

    if debug:
        print(filtered_extracted_df.head())


@benchmark_provenance_overhead(usecase_id="UC-02", n_iterations=10)
def extract_filter_map_join_movie_reviews(db_path, use_prov=False, debug=False):
    """
    02:

    """
    query = "SELECT * FROM movie_reviews LIMIT 10;"
    df = DataConnector.load_from_db(db_path, query=query)

    input_cols = ["reviewText"]
    output_cols = {"key_aspects": "The key aspects of the movie plot discussed in the review."}
    extracted_df = df.sem_extract(input_cols, output_cols, return_provenance=use_prov)

    mapped_df = extracted_df.sem_map(
        "Based on {key_aspects}, describe the primary setting of the movie.",
        suffix="setting",
        return_provenance=use_prov,
    )
    genres_df = {
        "genre": ["Science Fiction", "Drama", "Action", "Horror", "Documentary"],
        "description": [
            "Futuristic and high-tech",
            "Emotional and human-centric",
            "High energy and stunts",
            "Scary and dark",
            "Real-world facts",
        ],
    }
    genres_df = pd.DataFrame(genres_df)
    joined_df = mapped_df.sem_join(
        genres_df,
        "The {setting} is typical for a {genre} movie because it is {description}",
        return_provenance=use_prov,
    )

    if debug:
        print(joined_df.head())


# ==========================================
#   Single operator Use Cases
# ==========================================


@benchmark_provenance_overhead(usecase_id="UC-03", n_iterations=3)
def extract_movie_reviews(db_path, use_prov=False, debug=True):
    df = get_movie_review_data(db_path, limit=100)
    input_cols = ["reviewText"]
    output_cols = {"key_aspects": "The key aspects of the movie plot discussed in the review."}
    extracted_df = df.sem_extract(input_cols, output_cols, return_provenance=use_prov)
    if debug:
        print(extracted_df.head())


@benchmark_provenance_overhead(usecase_id="UC-04", n_iterations=3)
def filter_movie_reviews(db_path, use_prov=False, debug=True):
    df = get_movie_review_data(db_path, limit=500)
    filtered_df = df.sem_filter("The {reviewText} is positive about the movie's storyline?", return_provenance=use_prov)
    if debug:
        print(filtered_df.head())


# TODO: Add more use case functions here following the same pattern

if __name__ == "__main__":
    set_benchmark_env()
    # benchmark use case 01 -> "UC-01" as id
    # extract_filter_movie_reviews("sqlite:///../examples/db_examples/example_movie_reviews.db", debug=True)
    # extract_filter_map_join_movie_reviews("sqlite:///../examples/db_examples/example_movie_reviews.db", debug=True)
    # extract_movie_reviews("sqlite:///../examples/db_examples/example_movie_reviews.db", debug=True)
    filter_movie_reviews("sqlite:///../examples/db_examples/example_movie_reviews.db", debug=True)
