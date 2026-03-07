"""Run the TESTING.md protocol and persist datasets/results.

Outputs:
- test datasets: tests/tmp/dataset_<dataset_id>_r<n_rows>_f<n_features>_c<n_clusters>_seed_<seed>.feather
- protocol result table: result.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.generate import generate_synthetic_dataset
from dataset.testing import (
    DEFAULT_DATASET_SETTINGS,
    DEFAULT_RELATIONSHIP_MODES,
    default_selectors,
    evaluate_one_dataset,
    warm_up_selectors,
)


def run_protocol() -> pd.DataFrame:
    seeds = [0, 1, 2, 3, 4]
    relationship_modes = list(DEFAULT_RELATIONSHIP_MODES)
    target = "f_1_c_1"
    the_most_relevant = "f_1_c_3"
    methodologies = default_selectors()
    dataset_settings = list(DEFAULT_DATASET_SETTINGS)

    datasets_dir = Path("tests") / "tmp"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | int]] = []

    if not seeds:
        raise ValueError("seeds must contain at least one seed")

    warm_up_selectors(
        methodologies,
        target=target,
        the_most_relevant=the_most_relevant,
    )

    dataset_id = 1
    for n_rows, n_features, n_clusters in dataset_settings:
        for seed in seeds:
            for relationship_mode in relationship_modes:
                df = generate_synthetic_dataset(
                    n_rows=n_rows,
                    n_features=n_features,
                    n_clusters=n_clusters,
                    target=target,
                    the_most_relevant=the_most_relevant,
                    relationship_mode=relationship_mode,
                    seed=seed,
                )

                dataset_path = datasets_dir / (
                    f"dataset_{dataset_id}_r{n_rows}_f{n_features}_c{n_clusters}_seed_{seed}.feather"
                )
                df.to_feather(dataset_path)

                for selection_method, selector in methodologies:
                    row = evaluate_one_dataset(
                        df,
                        target=target,
                        the_ground_truth=the_most_relevant,
                        selector=selector,
                    )
                    row["selection_method"] = selection_method
                    row["n_rows"] = n_rows
                    row["n_features"] = n_features
                    row["n_clusters"] = n_clusters
                    row["dataset_id"] = dataset_id
                    row["seed"] = seed
                    row["target"] = target
                    row["relationship_mode"] = relationship_mode
                    rows.append(row)
                dataset_id += 1

    required_columns = [
        "selection_method",
        "n_rows",
        "n_features",
        "n_clusters",
        "selected_feature",
        "dbi_selected_feature",
        "corr_selected_feature",
        "computation_time_sec",
        "the_ground_truth",
        "dbi_ground_truth_feature",
        "corr_ground_truth_feature",
    ]
    optional_columns = ["dataset_id", "seed", "target", "relationship_mode"]

    result = pd.DataFrame(rows)[required_columns + optional_columns]
    expected_rows = len(dataset_settings) * len(seeds) * len(relationship_modes) * len(methodologies)
    if len(result) != expected_rows:
        raise RuntimeError(f"result must contain exactly {expected_rows} rows, got {len(result)}")

    rounded_result = result.round(3)
    rounded_result.to_csv("result.csv", index=False)
    return rounded_result


if __name__ == "__main__":
    protocol_result = run_protocol()
    print(protocol_result.to_string(index=False))
    print("\nSaved: result.csv")
    print("Saved datasets in: tests/tmp")
