"""Tests for the dataset generation utilities."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.generate import generate_synthetic_dataset, make_and_save


def _build_three_random_datasets() -> list[pd.DataFrame]:
    """Create three independent random datasets used by testing docs/protocol.

    We pin three seeds so test runs are reproducible while still representing
    three distinct random generations.
    """
    seeds = [0, 1, 2]
    datasets: list[pd.DataFrame] = []

    for seed in seeds:
        # Ground-truth feature (`the_most_relevant`) is intentionally in a
        # different cluster from `target` to match the selection constraints.
        df = generate_synthetic_dataset(
            n_features=100,
            n_clusters=5,
            measurements_per_feature=2,
            target="f_1_c_1",
            the_most_relevant="f_1_c_3",
            seed=seed,
        )
        datasets.append(df)

    return datasets


def test_basic_shape(tmp_path):
    df = generate_synthetic_dataset(
        n_features=100,
        n_clusters=5,
        measurements_per_feature=2,
        target="f_1_c_1",
        the_most_relevant="f_1_c_3",
        seed=0,
    )
    # expect 100*2 rows and 1 metadata column + 100 features
    assert df.shape[0] == 200
    assert df.shape[1] == 101
    for col in [
        "config",
        "f_1_c_1",
        "f_1_c_3",
    ]:
        assert col in df.columns


def test_generate_three_random_datasets_for_testing_protocol():
    datasets = _build_three_random_datasets()

    # The testing protocol requires exactly three generated datasets.
    assert len(datasets) == 3

    for df in datasets:
        assert "config" in df.columns
        assert "f_1_c_1" in df.columns
        assert "f_1_c_3" in df.columns


def test_save_and_load(tmp_path):
    path = tmp_path / "out.feather"
    df = make_and_save(
        str(path),
        n_features=50,
        n_clusters=3,
        n_rows=100,
        target="f_1_c_1",
        the_most_relevant="f_1_c_3",
        seed=1,
    )
    assert path.exists()
    loaded = pd.read_feather(path)
    # loaded frame should equal original
    pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded.reset_index(drop=True))


def test_most_relevant_is_nonlinear_and_highly_related():
    df = generate_synthetic_dataset(
        n_features=100,
        n_clusters=5,
        measurements_per_feature=2,
        target="f_1_c_1",
        the_most_relevant="f_1_c_3",
        seed=0,
    )

    target = df["f_1_c_1"].to_numpy(dtype=float)
    ground_truth = df["f_1_c_3"].to_numpy(dtype=float)

    # Check nonlinear pattern: cubic fit should improve over linear fit.
    corr_linear = np.corrcoef(target, ground_truth)[0, 1]
    corr_cubic = np.corrcoef(np.power(target, 3), ground_truth)[0, 1]
    assert abs(corr_cubic) > abs(corr_linear)

    # Check it is the most related candidate by absolute Pearson correlation.
    corr_gt = abs(float(df["f_1_c_3"].corr(df["f_1_c_1"])))
    candidates = [c for c in df.columns if c not in ("config", "f_1_c_1")]
    max_other = max(abs(float(df[c].corr(df["f_1_c_1"]))) for c in candidates if c != "f_1_c_3")
    assert corr_gt > max_other


if __name__ == "__main__":
    test_save_and_load(tmp_path=Path("tmp"))
