import pandas as pd

import lotus


def validate_provenance_pipeline():
    lm = lotus.models.LM(model="ollama/gemma:7b")
    lotus.settings.configure(lm=lm)

    data = {
        "review": [
            "The cinematography was breathtaking and the acting was superb.",
            "I hated the movie, it was a waste of time and money.",
            "An absolute masterpiece of modern cinema with great visuals.",
        ]
    }
    df = pd.DataFrame(data, index=[101, 202, 303])

    print("--- Stage 0: Original Data ---")
    print(df)

    df_extracted = df.sem_extract(
        input_cols=["review"],
        output_cols={
            "visual_quality": "Answer with only one word: 'High' if the visuals are praised, 'Low' otherwise."
        },
        provenance=True,
        provenance_col="provenance_id",
    )

    print("\n--- Stage 1: After sem_extract ---")
    print(df_extracted[["review", "visual_quality", "provenance_id"]])

    df_filtered = df_extracted.sem_filter(
        "The {visual_quality} was High.", provenance=True, provenance_col="provenance_id"
    )

    print("\n--- Stage 2: After sem_filter ---")
    print(df_filtered[["visual_quality", "provenance_id"]])

    # verification logic
    original_indices = [101, 202, 303]
    actual_provenance_ids = df_extracted["provenance_id"].tolist()

    is_valid = all(p_id in original_indices for p_id in actual_provenance_ids)

    print("\n--- Validation Result ---")
    if is_valid:
        print(f"SUCCESS: Provenance IDs {actual_provenance_ids} correctly map to original indices.")
    else:
        print(f"FAILURE: Provenance IDs {actual_provenance_ids} corrupted.")


if __name__ == "__main__":
    validate_provenance_pipeline()
