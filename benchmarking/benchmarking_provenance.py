import gc
import random
import time
import tracemalloc
from functools import wraps

import lotus
from lotus.data_connectors import DataConnector


def benchmark_provenance_overhead(func):
    """Benchmark the time and memory overhead of provenance tracking in LOTUS."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        debug = kwargs.pop("debug", True)
        results = {}

        modes = ["vanilla", "provenance"]
        random.shuffle(modes)

        for mode in modes:
            # set provenance flag based on mode
            prov_flag = mode == "provenance"
            gc.collect()
            tracemalloc.start()

            start_wall = time.perf_counter()
            start_cpu = time.process_time()

            # Execute the use case pipeline and pass the provenance flag
            func(*args, **kwargs, use_prov=prov_flag)

            end_cpu = time.process_time()
            end_wall = time.perf_counter()
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            results[mode] = {
                "wall_sec": end_wall - start_wall,
                "cpu_sec": end_cpu - start_cpu,
                "mem_mb": peak_mem / 10**6,
            }

        # Calculate overheads
        wall_time_oh = (results["provenance"]["wall_sec"] / results["vanilla"]["wall_sec"]) - 1
        cpu_time_oh = (results["provenance"]["cpu_sec"] / results["vanilla"]["cpu_sec"]) - 1
        mem_oh = (results["provenance"]["mem_mb"] / results["vanilla"]["mem_mb"]) - 1

        if debug:
            print(f"\n---- Benchmark: {func.__name__} ----")
            print(
                f"Wall Time: Vanilla LOTUS {results['vanilla']['wall_sec']:.4f}s | Provenance LOTUS {results['provenance']['wall_sec']:.4f}s | Overhead: {wall_time_oh:+.2%}"
            )
            print(
                f"CPU Time: Vanilla LOTUS {results['vanilla']['cpu_sec']:.4f}s | Provenance LOTUS {results['provenance']['cpu_sec']:.4f}s | Overhead: {cpu_time_oh:+.2%}"
            )
            print(
                f"Memory: Vanilla LOTUS {results['vanilla']['mem_mb']:.2f}MB | Provenance LOTUS {results['provenance']['mem_mb']:.2f}MB | Overhead: {mem_oh:+.2%}"
            )

        return results

    return wrapper


def set_benchmark_env():
    """Configure LOTUS specific benchmarking settings."""
    # disable caching
    lotus.settings.configure(enable_cache=False)
    # lm = lotus.models.LM(model="ollama/llama3.1:8b")
    lm = lotus.models.LM(model="ollama/gemma:7b")
    # lm = lotus.models.LM(model="gpt-4.1-nano")
    lotus.settings.configure(lm=lm)


# ==========================================
#   Use Case Section
# ==========================================


@benchmark_provenance_overhead
def run_extract_filter_movie_reviews(db_path, use_prov=False, debug=False):
    """
    01:
    SQL Use case with sem_extract and sem_filter on movie reviews dataset.
    Depends on sqlite file in created in db_examples/sql_extract_filter.py
    """
    # setup
    query = "SELECT * FROM movie_reviews LIMIT 50;"
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


# TODO: Add more use case functions here following the same pattern

if __name__ == "__main__":
    set_benchmark_env()
    run_extract_filter_movie_reviews("sqlite:///../examples/db_examples/example_movie_reviews.db", debug=True)
