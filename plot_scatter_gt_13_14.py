"""Plot ground-truth-vs-target scatter for specific dataset IDs.

Default behavior:
- reads metadata from result.csv
- loads dataset_13 and dataset_14 feather files from tests/tmp
- saves visuals/scatter_gt_13_14.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

AXIS_LABEL_FONTSIZE = 15


def _dataset_path_for_row(row: pd.Series, datasets_dir: Path) -> Path:
    dataset_id = int(row["dataset_id"])
    n_rows = int(row["n_rows"])
    n_features = int(row["n_features"])
    n_clusters = int(row["n_clusters"])
    seed = int(row["seed"])

    expected = datasets_dir / (
        f"dataset_{dataset_id}_r{n_rows}_f{n_features}_c{n_clusters}_seed_{seed}.feather"
    )
    if expected.exists():
        return expected

    fallback = list(datasets_dir.glob(f"dataset_{dataset_id}_*seed_{seed}.feather"))
    if fallback:
        return fallback[0]
    raise FileNotFoundError(
        f"dataset feather file not found for dataset_id={dataset_id}, seed={seed}"
    )


def _extract_cluster_id(feature_name: str) -> int:
    parts = feature_name.rsplit("_c_", maxsplit=1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"feature '{feature_name}' does not follow 'f_{{i}}_c_{{j}}' format")
    return int(parts[1])


def _pick_same_cluster_feature(df: pd.DataFrame, target_col: str) -> str:
    target_cluster = _extract_cluster_id(target_col)
    candidates: list[str] = []
    for col in df.columns:
        if col in {"config", target_col}:
            continue
        try:
            cluster_id = _extract_cluster_id(str(col))
        except ValueError:
            continue
        if cluster_id == target_cluster:
            candidates.append(str(col))
    if not candidates:
        raise ValueError(f"no same-cluster feature found for target '{target_col}'")
    return sorted(candidates)[0]


def _pick_random_off_cluster_feature(
    df: pd.DataFrame,
    target_col: str,
    gt_col: str,
    rng: np.random.Generator,
) -> str:
    target_cluster = _extract_cluster_id(target_col)
    gt_cluster = _extract_cluster_id(gt_col)
    excluded_clusters = {target_cluster, gt_cluster}

    candidates: list[str] = []
    for col in df.columns:
        if col in {"config", target_col, gt_col}:
            continue
        try:
            cluster_id = _extract_cluster_id(str(col))
        except ValueError:
            continue
        if cluster_id not in excluded_clusters:
            candidates.append(str(col))

    if not candidates:
        raise ValueError(
            f"no off-cluster feature found excluding clusters of target '{target_col}' and ground-truth '{gt_col}'"
        )
    return str(rng.choice(candidates))


def _build_config_style(
    dataset_frames: dict[int, pd.DataFrame],
) -> tuple[list[str], list[str], dict[str, int], ListedColormap, BoundaryNorm]:
    all_labels: set[str] = set()
    for df in dataset_frames.values():
        all_labels.update(df["config"].astype(str).unique().tolist())

    config_labels = sorted(all_labels)
    config_display_labels = [label.replace("cfg_", "DOE_") for label in config_labels]
    config_to_idx = {cfg: idx for idx, cfg in enumerate(config_labels)}

    # Keep categorical style consistent with existing scatter figures.
    base_cmap = plt.get_cmap("tab20")
    if len(config_labels) <= base_cmap.N:
        colors = base_cmap(np.linspace(0.0, 1.0, len(config_labels)))
    else:
        base_colors = base_cmap(np.linspace(0.0, 1.0, base_cmap.N))
        repeats = int(np.ceil(len(config_labels) / base_cmap.N))
        colors = np.vstack([base_colors for _ in range(repeats)])[: len(config_labels)]
    config_cmap = ListedColormap(colors)
    config_norm = BoundaryNorm(np.arange(-0.5, len(config_labels) + 0.5, 1), config_cmap.N)
    return config_labels, config_display_labels, config_to_idx, config_cmap, config_norm


def plot_scatter_ground_truth_subset(
    *,
    result_csv: Path,
    datasets_dir: Path,
    output_path: Path,
    dataset_ids: Sequence[int],
    dpi: int,
) -> None:
    result_df = pd.read_csv(result_csv)
    required = {"dataset_id", "seed", "n_rows", "n_features", "n_clusters", "target", "the_ground_truth"}
    missing = sorted(required - set(result_df.columns))
    if missing:
        raise ValueError(f"result.csv missing required columns: {missing}")

    target_ids = [int(x) for x in dataset_ids]
    if not target_ids:
        raise ValueError("dataset_ids must not be empty")

    dataset_frames: dict[int, pd.DataFrame] = {}
    row_by_id: dict[int, pd.Series] = {}
    for dataset_id in target_ids:
        subset = result_df[result_df["dataset_id"].astype(int) == dataset_id]
        if subset.empty:
            raise ValueError(f"dataset_id {dataset_id} not found in result.csv")
        row0 = subset.iloc[0]
        row_by_id[dataset_id] = row0

        dataset_path = _dataset_path_for_row(row0, datasets_dir)
        df = pd.read_feather(dataset_path)
        if "config" not in df.columns:
            raise KeyError(f"'config' column not found in {dataset_path}")
        dataset_frames[dataset_id] = df

    (
        config_labels,
        config_display_labels,
        config_to_idx,
        config_cmap,
        config_norm,
    ) = _build_config_style(dataset_frames)

    fig, axes = plt.subplots(
        len(target_ids),
        3,
        figsize=(18, 5 * len(target_ids)),
        squeeze=False,
        constrained_layout=True,
    )

    row_scatters: list[plt.Artist | None] = [None for _ in target_ids]
    for row_idx, dataset_id in enumerate(target_ids):
        ax_col1 = axes[row_idx][0]
        ax_col2 = axes[row_idx][1]
        ax_col3 = axes[row_idx][2]
        row0 = row_by_id[dataset_id]
        target_col = str(row0["target"])
        gt_col = str(row0["the_ground_truth"])
        df = dataset_frames[dataset_id]

        if target_col not in df.columns:
            raise KeyError(f"target '{target_col}' not found for dataset {dataset_id}")
        if gt_col not in df.columns:
            raise KeyError(f"the_ground_truth '{gt_col}' not found for dataset {dataset_id}")

        config_codes = df["config"].astype(str).map(config_to_idx)
        row_scatters[row_idx] = ax_col1.scatter(
            df[gt_col],
            df[target_col],
            c=config_codes,
            cmap=config_cmap,
            norm=config_norm,
            alpha=0.65,
            s=12,
        )
        ax_col1.set_title(f"dataset_{dataset_id} | ground_truth")
        ax_col1.set_xlabel("most_relevant_feature", fontsize=AXIS_LABEL_FONTSIZE)
        ax_col1.set_ylabel("target_feature", fontsize=AXIS_LABEL_FONTSIZE)

        same_cluster_feature = _pick_same_cluster_feature(df, target_col)
        ax_col2.scatter(
            df[same_cluster_feature],
            df[target_col],
            c=config_codes,
            cmap=config_cmap,
            norm=config_norm,
            alpha=0.65,
            s=12,
        )
        ax_col2.set_title(f"dataset_{dataset_id} | same_cluster")
        ax_col2.set_xlabel("same-cluster feature", fontsize=AXIS_LABEL_FONTSIZE)
        ax_col2.set_ylabel("target_feature", fontsize=AXIS_LABEL_FONTSIZE)

        rng = np.random.default_rng(int(row0["seed"]) + dataset_id)
        off_cluster_feature = _pick_random_off_cluster_feature(df, target_col, gt_col, rng)
        ax_col3.scatter(
            df[off_cluster_feature],
            df[target_col],
            c=config_codes,
            cmap=config_cmap,
            norm=config_norm,
            alpha=0.65,
            s=12,
        )
        ax_col3.set_title(f"dataset_{dataset_id} | off_cluster_random")
        ax_col3.set_xlabel("off-cluster feature", fontsize=AXIS_LABEL_FONTSIZE)
        ax_col3.set_ylabel("target_feature", fontsize=AXIS_LABEL_FONTSIZE)

    for row_idx in range(len(target_ids)):
        scatter_artist = row_scatters[row_idx]
        if scatter_artist is None:
            continue
        cbar = fig.colorbar(
            scatter_artist,
            ax=axes[row_idx, :].tolist(),
            shrink=0.80,
            fraction=0.045,
            pad=0.02,
        )
        cbar.set_label("DOE", fontsize=9)
        cbar.set_ticks(range(len(config_labels)))
        cbar.set_ticklabels(config_display_labels)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Columns: Ground Truth vs Target | Same-Cluster vs Target | Off-Cluster vs Target",
        y=1.01,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=max(200, int(dpi)), bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ground-truth vs target scatter for selected dataset IDs")
    parser.add_argument("--result-csv", type=Path, default=Path("result.csv"))
    parser.add_argument("--datasets-dir", type=Path, default=Path("tests/tmp"))
    parser.add_argument("--output", type=Path, default=Path("visuals/scatter_gt_13_14.png"))
    parser.add_argument("--dataset-ids", type=int, nargs="+", default=[13, 14])
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_scatter_ground_truth_subset(
        result_csv=args.result_csv,
        datasets_dir=args.datasets_dir,
        output_path=args.output,
        dataset_ids=args.dataset_ids,
        dpi=args.dpi,
    )
    print(f"Saved: {args.output}")
