"""Testing utilities for the three-random-dataset evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from dataset.generate import generate_synthetic_dataset


Selector = Callable[[pd.DataFrame, str, Sequence[str]], str]
DatasetSetting = tuple[int, int, int]  # (n_rows, n_features, n_clusters)
_SHAP_MODULE = None
_RF_REGRESSOR = None
_MI_REGRESSION = None

DEFAULT_DATASET_SETTINGS: tuple[DatasetSetting, ...] = (
    (100, 300, 3),
    (1000, 300, 3),
    (1000, 3000, 30),
)


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration for one generated dataset in the testing protocol."""

    seed: int
    target: str
    the_most_relevant: str


def _prepare_selector_dependencies(selector: Selector) -> None:
    """Load optional dependencies outside measured timing windows."""
    if selector is select_feature_by_abs_shapley:
        _get_shap_dependencies()
    elif selector is select_feature_by_mutual_info:
        _get_mi_dependency()


def _get_shap_dependencies():
    global _SHAP_MODULE, _RF_REGRESSOR
    if _SHAP_MODULE is not None and _RF_REGRESSOR is not None:
        return _SHAP_MODULE, _RF_REGRESSOR

    try:
        import shap
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise ImportError(
            "shap selector requires 'shap' and 'scikit-learn'. "
            "Install with: pip install shap scikit-learn"
        ) from exc

    _SHAP_MODULE = shap
    _RF_REGRESSOR = RandomForestRegressor
    return _SHAP_MODULE, _RF_REGRESSOR


def _get_mi_dependency():
    global _MI_REGRESSION
    if _MI_REGRESSION is not None:
        return _MI_REGRESSION

    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError as exc:
        raise ImportError(
            "mi selector requires 'scikit-learn'. Install with: pip install scikit-learn"
        ) from exc

    _MI_REGRESSION = mutual_info_regression
    return _MI_REGRESSION


def _extract_cluster_id(feature_name: str) -> int:
    """Extract cluster id from names formatted like ``f_{i}_c_{j}``."""
    parts = feature_name.rsplit("_c_", maxsplit=1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"feature '{feature_name}' does not follow 'f_{{i}}_c_{{j}}' format")
    return int(parts[1])


def _candidate_features(df: pd.DataFrame, target: str) -> list[str]:
    """Return all valid candidates excluding target and target-cluster features."""
    if "config" not in df.columns:
        raise ValueError("dataframe must contain a 'config' column")
    if target not in df.columns:
        raise ValueError(f"target '{target}' is not present in dataframe")

    target_cluster = _extract_cluster_id(target)
    candidates: list[str] = []
    for col in df.columns:
        if col == "config" or col == target:
            continue
        if _extract_cluster_id(col) == target_cluster:
            continue
        candidates.append(col)

    if not candidates:
        raise ValueError("no valid candidate features after applying cluster exclusion")
    return candidates


def select_feature_by_abs_corr(df: pd.DataFrame, target: str, candidates: Sequence[str]) -> str:
    """Select the candidate with maximum absolute Pearson correlation to target."""
    target_series = df[target]
    best_feature: str | None = None
    best_score = -np.inf

    for feature in candidates:
        score = abs(float(df[feature].corr(target_series)))
        if np.isnan(score):
            continue
        if score > best_score or (score == best_score and (best_feature is None or feature < best_feature)):
            best_feature = feature
            best_score = score

    if best_feature is None:
        raise ValueError("all candidate correlations are NaN")
    return best_feature


def select_feature_by_abs_spearman(df: pd.DataFrame, target: str, candidates: Sequence[str]) -> str:
    """Select the candidate with maximum absolute Spearman correlation to target."""
    target_rank = df[target].rank(method="average")
    best_feature: str | None = None
    best_score = -np.inf

    for feature in candidates:
        # Spearman without scipy: Pearson correlation on ranked values.
        score = abs(float(df[feature].rank(method="average").corr(target_rank)))
        if np.isnan(score):
            continue
        if score > best_score or (score == best_score and (best_feature is None or feature < best_feature)):
            best_feature = feature
            best_score = score

    if best_feature is None:
        raise ValueError("all candidate correlations are NaN")
    return best_feature


def select_feature_by_min_dbi(df: pd.DataFrame, target: str, candidates: Sequence[str]) -> str:
    """Select the candidate with minimum DBI on [candidate, target] by config labels."""
    best_feature: str | None = None
    best_score = np.inf

    for feature in candidates:
        score = davies_bouldin_index(df[feature], df[target], df["config"])
        if np.isnan(score):
            continue
        if score < best_score or (score == best_score and (best_feature is None or feature < best_feature)):
            best_feature = feature
            best_score = score

    if best_feature is None:
        raise ValueError("all candidate DBI scores are NaN")
    return best_feature


def select_feature_by_abs_shapley(df: pd.DataFrame, target: str, candidates: Sequence[str]) -> str:
    """Select feature by max absolute SHAP index using shap.TreeExplainer."""
    shap, RandomForestRegressor = _get_shap_dependencies()

    x = df[list(candidates)].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(x, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        # Multi-output shape: [n_outputs, n_samples, n_features]
        mean_abs = np.mean(np.abs(shap_array), axis=(0, 1))
    elif shap_array.ndim == 2:
        # Standard regression shape: [n_samples, n_features]
        mean_abs = np.mean(np.abs(shap_array), axis=0)
    else:
        raise ValueError(f"unexpected SHAP output shape: {shap_array.shape}")

    best_feature: str | None = None
    best_score = -np.inf
    for feature, score in zip(candidates, mean_abs):
        score_f = float(score)
        if np.isnan(score_f):
            continue
        if score_f > best_score or (score_f == best_score and (best_feature is None or feature < best_feature)):
            best_feature = feature
            best_score = score_f

    if best_feature is None:
        raise ValueError("all candidate SHAP scores are NaN")
    return best_feature


def select_feature_by_mutual_info(df: pd.DataFrame, target: str, candidates: Sequence[str]) -> str:
    """Select feature by maximum mutual information to target."""
    mutual_info_regression = _get_mi_dependency()

    x = df[list(candidates)].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    scores = mutual_info_regression(x, y, random_state=0)

    best_feature: str | None = None
    best_score = -np.inf
    for feature, score in zip(candidates, scores):
        score_f = float(score)
        if np.isnan(score_f):
            continue
        if score_f > best_score or (score_f == best_score and (best_feature is None or feature < best_feature)):
            best_feature = feature
            best_score = score_f

    if best_feature is None:
        raise ValueError("all candidate mutual information scores are NaN")
    return best_feature


def default_selectors() -> list[tuple[str, Selector]]:
    """Return default selector methods used by protocol runs."""
    return [
        ("abs_pearson", select_feature_by_abs_corr),
        ("min_dbi", select_feature_by_min_dbi),
        ("shap", select_feature_by_abs_shapley),
        ("mi", select_feature_by_mutual_info),
    ]


def davies_bouldin_index(feature_values: pd.Series, target_values: pd.Series, labels: pd.Series) -> float:
    """Compute DBI on 2D points [feature, target] grouped by labels."""
    x = np.column_stack((feature_values.to_numpy(dtype=float), target_values.to_numpy(dtype=float)))
    label_values = labels.to_numpy()
    unique_labels = pd.unique(label_values)

    if unique_labels.size < 2:
        raise ValueError("DBI requires at least two unique config labels")

    centroids: list[np.ndarray] = []
    scatters: list[float] = []
    for label in unique_labels:
        cluster_points = x[label_values == label]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        scatters.append(float(distances.mean()))

    c = np.vstack(centroids)
    s = np.array(scatters)

    dbi_terms: list[float] = []
    for i in range(len(unique_labels)):
        max_ratio = -np.inf
        for j in range(len(unique_labels)):
            if i == j:
                continue
            centroid_distance = float(np.linalg.norm(c[i] - c[j]))
            if centroid_distance == 0.0:
                ratio = np.inf
            else:
                ratio = float((s[i] + s[j]) / centroid_distance)
            if ratio > max_ratio:
                max_ratio = ratio
        dbi_terms.append(max_ratio)

    return float(np.mean(dbi_terms))


def evaluate_one_dataset(
    df: pd.DataFrame,
    *,
    target: str,
    the_ground_truth: str,
    selector: Selector | None = None,
) -> dict[str, float | str]:
    """Evaluate one dataset and return one result-table row."""
    if selector is None:
        selector = select_feature_by_abs_corr

    candidates = _candidate_features(df, target)
    if the_ground_truth not in df.columns:
        raise ValueError(f"the_ground_truth '{the_ground_truth}' is not present in dataframe")

    _prepare_selector_dependencies(selector)
    start = perf_counter()
    selected = selector(df, target, candidates)
    computation_time_sec = perf_counter() - start

    if selected == target:
        raise ValueError("selector returned target; this violates testing protocol")
    if _extract_cluster_id(selected) == _extract_cluster_id(target):
        raise ValueError("selector returned same-cluster feature; violates cluster exclusion")

    return {
        "selected_feature": selected,
        "dbi_selected_feature": davies_bouldin_index(df[selected], df[target], df["config"]),
        "corr_selected_feature": float(df[selected].corr(df[target])),
        "the_ground_truth": the_ground_truth,
        "dbi_ground_truth_feature": davies_bouldin_index(df[the_ground_truth], df[target], df["config"]),
        "corr_ground_truth_feature": float(df[the_ground_truth].corr(df[target])),
        "computation_time_sec": float(computation_time_sec),
    }


def warm_up_selectors(
    methods: Sequence[tuple[str, Selector]],
    *,
    target: str,
    the_most_relevant: str,
    seed: int = 12345,
) -> None:
    """Warm up selector dependencies and one dry-run to reduce cold-start bias."""
    if not methods:
        return

    warm_df = generate_synthetic_dataset(
        n_rows=30,
        n_features=60,
        n_clusters=3,
        target=target,
        the_most_relevant=the_most_relevant,
        seed=seed,
    )
    candidates = _candidate_features(warm_df, target)
    for _, selector in methods:
        _prepare_selector_dependencies(selector)
        selector(warm_df, target, candidates)


def run_three_dataset_test(
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    target: str = "f_1_c_1",
    the_most_relevant: str = "f_1_c_3",
    dataset_settings: Iterable[DatasetSetting] = DEFAULT_DATASET_SETTINGS,
    selector: Selector | None = None,
) -> pd.DataFrame:
    """Run the TESTING.md protocol and return result rows.

    The returned table has one row per (dataset_setting x seed x selection method) and required columns:
    selection_method, selected_feature, dbi_selected_feature, corr_selected_feature,
    the_ground_truth, dbi_ground_truth_feature, corr_ground_truth_feature, computation_time_sec.
    If ``selector`` is provided, one method is used. Otherwise defaults are used.
    """
    seed_list = [int(s) for s in seeds]
    settings_list = [(int(r), int(f), int(c)) for r, f, c in dataset_settings]
    if not settings_list:
        raise ValueError("dataset_settings must contain at least one setting")
    if not seed_list:
        raise ValueError("seeds must contain at least one seed")

    methods: list[tuple[str, Selector]]
    if selector is None:
        methods = default_selectors()
    else:
        methods = [("custom_selector", selector)]

    warm_up_selectors(
        methods,
        target=target,
        the_most_relevant=the_most_relevant,
    )

    rows: list[dict[str, float | str]] = []
    dataset_id = 1
    for n_rows, n_features, n_clusters in settings_list:
        for seed in seed_list:
            df = generate_synthetic_dataset(
                n_rows=n_rows,
                n_features=n_features,
                n_clusters=n_clusters,
                target=target,
                the_most_relevant=the_most_relevant,
                seed=seed,
            )
            for method_name, method_selector in methods:
                row = evaluate_one_dataset(
                    df,
                    target=target,
                    the_ground_truth=the_most_relevant,
                    selector=method_selector,
                )
                row["selection_method"] = method_name
                row["n_rows"] = n_rows
                row["n_features"] = n_features
                row["n_clusters"] = n_clusters
                row["dataset_id"] = dataset_id
                row["seed"] = seed
                row["target"] = target
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
        "the_ground_truth",
        "dbi_ground_truth_feature",
        "corr_ground_truth_feature",
        "computation_time_sec",
    ]

    result = pd.DataFrame(rows)
    missing = [col for col in required_columns if col not in result.columns]
    if missing:
        raise RuntimeError(f"result table missing required columns: {missing}")

    ordered_columns = required_columns + ["dataset_id", "seed", "target"]
    result = result[ordered_columns]

    expected_rows = len(settings_list) * len(seed_list) * len(methods)
    if len(result) != expected_rows:
        raise RuntimeError(f"result table must contain {expected_rows} rows, got {len(result)}")

    return result
