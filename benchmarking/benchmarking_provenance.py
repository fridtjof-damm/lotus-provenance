import gc
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

        for mode in ["vanilla", "provenance"]:
            # set provenance flag based on mode
            prov_flag = True if mode == "provenance" else False

            # Reset and start tracing
            gc.collect()
            tracemalloc.start()
            baseline_mem, _ = tracemalloc.get_traced_memory()
            start_time = time.perf_counter()

            # Execute the use case pipeline and pass the provenance flag
            func(*args, **kwargs, use_prov=prov_flag)

            # stop timing and memory tracking
            end_time = time.perf_counter()
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak = peak_mem - baseline_mem

            results[mode] = {"time_sec": end_time - start_time, "peak_mem_mb": peak / 10**6}

        # Calculate overheads
        time_overhead = (results["provenance"]["time_sec"] / results["vanilla"]["time_sec"]) - 1
        mem_overhead = (results["provenance"]["peak_mem_mb"] / results["vanilla"]["peak_mem_mb"]) - 1

        if debug:
            print(f"\n---- Benchmark: {func.__name__} ----")
            print(
                f"Time: Vanilla LOTUS {results['vanilla']['time_sec']:.4f}s | Provenance LOTUS {results['provenance']['time_sec']:.4f}s | Overhead: {time_overhead:+.2%}"
            )
            print(
                f"Memory: Vanilla LOTUS {results['vanilla']['peak_mem_mb']:.2f}MB | Provenance LOTUS {results['provenance']['peak_mem_mb']:.2f}MB | Overhead: {mem_overhead:+.2%}"
            )

        return results

    return wrapper


def set_becnhmark_env():
    """Configure LOTUS specific benchmarking settings."""
    # disable caching
    lotus.settings.configure(enable_cache=False)
    lm = lotus.models.LM(model="gpt-4o-mini")
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
    query = "SELECT * FROM movie_reviews LIMIT 100;"
    df = DataConnector.load_from_db(db_path, query=query)
    set_becnhmark_env()

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
    run_extract_filter_movie_reviews("sqlite:///../examples/db_examples/example_movie_reviews.db", debug=True)
