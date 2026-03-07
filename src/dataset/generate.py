"""Utilities for synthesizing feature-relevance datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _canonical_normalize_feature_columns(values: dict[str, np.ndarray]) -> None:
    """Apply per-feature canonical normalization (z-score) in place."""
    for feature_name, feature_values in values.items():
        mean = float(np.mean(feature_values))
        std = float(np.std(feature_values))
        if std == 0.0:
            values[feature_name] = feature_values - mean
            continue
        values[feature_name] = (feature_values - mean) / std


def _build_feature_names(n_features: int, n_clusters: int) -> tuple[list[str], list[int]]:
    """Build feature names in the form ``f_{i}_c_{j}`` and their cluster ids."""
    if n_features < 1:
        raise ValueError("n_features must be at least 1")
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if n_features < n_clusters:
        raise ValueError("n_features must be at least n_clusters")

    base = n_features // n_clusters
    remainder = n_features % n_clusters

    names: list[str] = []
    cluster_ids: list[int] = []
    for cluster in range(1, n_clusters + 1):
        count = base + (1 if cluster <= remainder else 0)
        for i in range(1, count + 1):
            names.append(f"f_{i}_c_{cluster}")
            cluster_ids.append(cluster)
    return names, cluster_ids


def generate_synthetic_dataset(
    *,
    n_configs: int | Iterable[int] = (2, 8),
    n_clusters: int = 30,
    n_features: int = 20000,
    measurements_per_feature: int = 5,
    n_rows: int | None = None,
    target: str | None = None,
    the_most_relevant: str | None = None,
    config_effect_scale: float = 3.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Create a synthetic wide-format dataset for feature relevance experiments.

    Returned shape is:
    ``n_rows x [1 metadata column + n_features feature columns]``.
    """
    rng = np.random.default_rng(seed)

    if isinstance(n_configs, int):
        n_configs = (n_configs, n_configs)
    min_cfg, max_cfg = int(n_configs[0]), int(n_configs[1])
    if min_cfg < 1 or max_cfg < 1 or min_cfg > max_cfg:
        raise ValueError("n_configs must be >=1 and min<=max")

    feature_names, feature_clusters = _build_feature_names(n_features, n_clusters)
    feature_to_cluster = {name: cluster for name, cluster in zip(feature_names, feature_clusters)}

    if target is not None and target not in feature_to_cluster:
        raise ValueError(f"target '{target}' is not a generated feature name")
    if the_most_relevant is not None and the_most_relevant not in feature_to_cluster:
        raise ValueError(f"the_most_relevant '{the_most_relevant}' is not a generated feature name")

    if target is None:
        target = feature_names[0]
    if the_most_relevant is None:
        target_cluster = feature_to_cluster[target]
        candidates = [name for name in feature_names if feature_to_cluster[name] != target_cluster]
        the_most_relevant = candidates[0]
    if feature_to_cluster[target] == feature_to_cluster[the_most_relevant]:
        raise ValueError("target and the_most_relevant must be in different clusters")

    if n_rows is None:
        if measurements_per_feature < 1:
            raise ValueError("measurements_per_feature must be at least 1")
        n_rows = n_features * measurements_per_feature
    elif n_rows < 1:
        raise ValueError("n_rows must be at least 1")
    if config_effect_scale <= 0:
        raise ValueError("config_effect_scale must be > 0")

    cfg_count = int(rng.integers(min_cfg, max_cfg + 1))
    configs = [f"cfg_{idx}" for idx in range(cfg_count)]
    config = rng.choice(configs, size=n_rows, replace=True)

    # Features in the same cluster share the same latent signal, making them correlated.
    cluster_latent = rng.normal(size=(n_rows, n_clusters))
    values: dict[str, np.ndarray] = {}
    for feature_name, cluster in zip(feature_names, feature_clusters):
        values[feature_name] = cluster_latent[:, cluster - 1] + rng.normal(scale=0.30, size=n_rows)

    # Couple target and most relevant with a shared signal.
    shared_signal = rng.normal(size=n_rows)
    values[target] = 0.6 * values[target] + 1.4 * shared_signal + rng.normal(scale=0.10, size=n_rows)

    # Most relevant also varies strongly by config.
    # It is intentionally a nonlinear transform of target while keeping
    # a strong overall association so it is the most relevant candidate.
    cfg_effect_map = {
        cfg_name: float(rng.normal(scale=config_effect_scale)) for cfg_name in pd.unique(config)
    }
    cfg_effect = np.array([cfg_effect_map[cfg_name] for cfg_name in config])
    target_values = values[target]
    values[the_most_relevant] = (
        1.1 * np.tanh(1.2 * target_values)
        + 0.25 * np.power(target_values, 3)
        + 0.15 * values[the_most_relevant]
        + 0.40 * cfg_effect
        + rng.normal(scale=0.03, size=n_rows)
    )

    # Canonical normalization is applied before any downstream feature selection.
    _canonical_normalize_feature_columns(values)

    df = pd.DataFrame({"config": config})
    for feature_name in feature_names:
        df[feature_name] = values[feature_name]
    return df


def make_and_save(path: str, **kwargs) -> pd.DataFrame:
    """Generate a dataset and persist it to ``path`` in Feather format."""
    df = generate_synthetic_dataset(**kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(path)
    return df
