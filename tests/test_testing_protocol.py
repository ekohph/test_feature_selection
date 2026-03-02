"""Tests for the TESTING.md protocol implementation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.testing import run_three_dataset_test


def _cluster_id(name: str) -> int:
    return int(name.rsplit("_c_", maxsplit=1)[1])


def test_run_three_dataset_test_schema_and_rows():
    result = run_three_dataset_test()

    assert len(result) == 9

    required_columns = [
        "selection_method",
        "selected_feature",
        "dbi_selected_feature",
        "corr_selected_feature",
        "the_ground_truth",
        "dbi_ground_truth_feature",
        "corr_ground_truth_feature",
        "computation_time_sec",
    ]
    for col in required_columns:
        assert col in result.columns


def test_selection_respects_cluster_exclusion_and_runtime_is_non_negative():
    result = run_three_dataset_test(target="f_1_c_1", the_most_relevant="f_1_c_3")

    for _, row in result.iterrows():
        assert row["selected_feature"] != row["target"]
        assert _cluster_id(row["selected_feature"]) != _cluster_id(row["target"])
        assert row["computation_time_sec"] >= 0.0
