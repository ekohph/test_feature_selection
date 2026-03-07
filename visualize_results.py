"""Visualize protocol outputs from result.csv and generated dataset files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap


REQUIRED_RESULT_COLUMNS = {
    "selection_method",
    "dataset_id",
    "seed",
    "n_rows",
    "n_features",
    "n_clusters",
    "selected_feature",
    "the_ground_truth",
    "target",
    "computation_time_sec",
}


def _sanitize_filename_token(value: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in value.strip())
    return token or "unknown"


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
    raise FileNotFoundError(f"dataset feather file not found for dataset_id={dataset_id}, seed={seed}")


def _load_dataset_frames(
    result_df: pd.DataFrame,
    datasets_dir: Path,
) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for dataset_id, subset in result_df.groupby("dataset_id", sort=True):
        dataset_path = _dataset_path_for_row(subset.iloc[0], datasets_dir)
        df = pd.read_feather(dataset_path)
        if "config" not in df.columns:
            raise KeyError(f"'config' column not found in {dataset_path}")
        frames[int(dataset_id)] = df
    return frames


def _build_config_style(
    dataset_frames: dict[int, pd.DataFrame],
) -> tuple[list[str], dict[str, int], ListedColormap, BoundaryNorm]:
    all_labels: set[str] = set()
    for df in dataset_frames.values():
        all_labels.update(df["config"].astype(str).unique().tolist())
    config_labels = sorted(all_labels)
    config_to_idx = {cfg: idx for idx, cfg in enumerate(config_labels)}

    # Use a qualitative palette so neighboring config labels are easier to distinguish.
    base_cmap = plt.get_cmap("tab20")
    if len(config_labels) <= base_cmap.N:
        colors = base_cmap(np.linspace(0.0, 1.0, len(config_labels)))
    else:
        base_colors = base_cmap(np.linspace(0.0, 1.0, base_cmap.N))
        repeats = int(np.ceil(len(config_labels) / base_cmap.N))
        colors = np.vstack([base_colors for _ in range(repeats)])[: len(config_labels)]
    config_cmap = ListedColormap(colors)

    config_norm = BoundaryNorm(np.arange(-0.5, len(config_labels) + 0.5, 1), config_cmap.N)
    return config_labels, config_to_idx, config_cmap, config_norm


def _apply_config_colorbar(
    fig: plt.Figure,
    axes: np.ndarray,
    scatter_artist: plt.Artist | None,
    config_labels: list[str],
) -> None:
    if scatter_artist is None or not config_labels:
        return
    cbar = fig.colorbar(scatter_artist, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("config")
    cbar.set_ticks(range(len(config_labels)))
    cbar.set_ticklabels(config_labels)


def _plot_scatter_ground_truth(
    result_df: pd.DataFrame,
    dataset_frames: dict[int, pd.DataFrame],
    config_labels: list[str],
    config_to_idx: dict[str, int],
    config_cmap: ListedColormap,
    config_norm: BoundaryNorm,
    output_dir: Path,
) -> None:
    grouped = list(result_df.groupby("dataset_id", sort=True))
    n_plots = len(grouped)
    if n_plots == 0:
        raise ValueError("result.csv has no rows to plot")

    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    last_scatter = None
    for idx, (dataset_id, subset) in enumerate(grouped):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        row0 = subset.iloc[0]
        target = str(row0["target"])
        gt_feature = str(row0["the_ground_truth"])
        df = dataset_frames[int(dataset_id)]

        if target not in df.columns:
            raise KeyError(f"target '{target}' not found for dataset {int(dataset_id)}")
        if gt_feature not in df.columns:
            raise KeyError(f"the_ground_truth '{gt_feature}' not found for dataset {int(dataset_id)}")

        config_codes = df["config"].astype(str).map(config_to_idx)
        last_scatter = ax.scatter(
            df[gt_feature],
            df[target],
            c=config_codes,
            cmap=config_cmap,
            norm=config_norm,
            alpha=0.65,
            s=12,
        )
        ax.set_title(f"dataset_{int(dataset_id)} | ground_truth")
        ax.set_xlabel(gt_feature)
        ax.set_ylabel(target)

    for idx in range(n_plots, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")

    _apply_config_colorbar(fig, axes, last_scatter, config_labels)
    fig.suptitle("Ground Truth Feature vs Target", y=1.03)
    fig.savefig(output_dir / "scatter_gt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter_selected_mismatch(
    result_df: pd.DataFrame,
    dataset_frames: dict[int, pd.DataFrame],
    config_labels: list[str],
    config_to_idx: dict[str, int],
    config_cmap: ListedColormap,
    config_norm: BoundaryNorm,
    output_dir: Path,
) -> None:
    mismatch_df = result_df[result_df["selected_feature"] != result_df["the_ground_truth"]].copy()
    if mismatch_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No selected_feature differs from the_ground_truth.",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        fig.savefig(output_dir / "scatter_selected.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    mismatch_df = mismatch_df.sort_values(["dataset_id", "selection_method"]).reset_index(drop=True)
    n_plots = len(mismatch_df)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    last_scatter = None
    for idx, (_, row) in enumerate(mismatch_df.iterrows()):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        dataset_id = int(row["dataset_id"])
        method = str(row["selection_method"])
        selected_feature = str(row["selected_feature"])
        target = str(row["target"])
        df = dataset_frames[dataset_id]

        if selected_feature not in df.columns:
            raise KeyError(f"selected_feature '{selected_feature}' not found for dataset {dataset_id}")
        if target not in df.columns:
            raise KeyError(f"target '{target}' not found for dataset {dataset_id}")

        config_codes = df["config"].astype(str).map(config_to_idx)
        last_scatter = ax.scatter(
            df[selected_feature],
            df[target],
            c=config_codes,
            cmap=config_cmap,
            norm=config_norm,
            alpha=0.65,
            s=12,
        )
        ax.set_title(f"dataset_{dataset_id} | {method}")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel(target)

    for idx in range(n_plots, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")

    _apply_config_colorbar(fig, axes, last_scatter, config_labels)
    fig.suptitle("Selected Feature (Mismatch) vs Target", y=1.03)
    fig.savefig(output_dir / "scatter_selected.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_computation_time_bar(
    result_df: pd.DataFrame,
    output_dir: Path,
    summary: str = "median_iqr",
    output_name: str = "time_bar.png",
    title_context: str | None = None,
) -> None:
    method_order = ["abs_pearson", "min_dbi", "mi", "shap"]
    if summary not in {"mean_std", "median_iqr"}:
        raise ValueError("summary must be one of: mean_std, median_iqr")

    grouped = result_df.groupby(["n_rows", "n_features", "selection_method"])["computation_time_sec"]
    if summary == "mean_std":
        stats = grouped.agg(center="mean", spread_low="std", spread_high="std").reset_index()
        stats["spread_low"] = stats["spread_low"].fillna(0.0)
        stats["spread_high"] = stats["spread_high"].fillna(0.0)
        title_suffix = "Mean +/- Std"
    else:
        stats = grouped.agg(
            center="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
        ).reset_index()
        stats["spread_low"] = (stats["center"] - stats["q1"]).clip(lower=0.0)
        stats["spread_high"] = (stats["q3"] - stats["center"]).clip(lower=0.0)
        title_suffix = "Median +/- IQR"

    stats = stats.sort_values(["n_rows", "n_features"])
    stats["shape_label"] = stats.apply(
        lambda row: f"{int(row['n_rows'])} x {int(row['n_features'])}",
        axis=1,
    )

    shape_order = (
        stats[["n_rows", "n_features", "shape_label"]]
        .drop_duplicates()
        .sort_values(["n_rows", "n_features"])["shape_label"]
        .tolist()
    )
    ordered_methods = method_order + [m for m in stats["selection_method"].unique() if m not in method_order]

    mean_pivot = (
        stats.pivot_table(
            index="shape_label",
            columns="selection_method",
            values="center",
            aggfunc="first",
        )
        .reindex(shape_order)
        .reindex(columns=ordered_methods)
    )
    low_pivot = (
        stats.pivot_table(
            index="shape_label",
            columns="selection_method",
            values="spread_low",
            aggfunc="first",
        )
        .reindex(shape_order)
        .reindex(columns=ordered_methods)
    )
    high_pivot = (
        stats.pivot_table(
            index="shape_label",
            columns="selection_method",
            values="spread_high",
            aggfunc="first",
        )
        .reindex(shape_order)
        .reindex(columns=ordered_methods)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(mean_pivot.index))
    n_methods = len(mean_pivot.columns)
    width = 0.8 / max(n_methods, 1)
    blue_palette = plt.cm.Blues(np.linspace(0.35, 0.9, n_methods))

    for i, method in enumerate(mean_pivot.columns):
        means = mean_pivot[method].to_numpy(dtype=float)
        spread_low = low_pivot[method].to_numpy(dtype=float)
        spread_high = high_pivot[method].to_numpy(dtype=float)
        yerr = np.vstack([spread_low, spread_high])
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            label=method,
            color=blue_palette[i],
            yerr=yerr,
            capsize=4,
            error_kw={"elinewidth": 1.0, "alpha": 0.9},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(mean_pivot.index, rotation=0)
    title = f"Computation Time {title_suffix} by Dataset Shape and Selection Method"
    if title_context:
        title = f"{title} ({title_context})"
    ax.set_title(title)
    ax.set_xlabel("dataset_shape (#rows x #features)")
    ax.set_ylabel("computation_time_sec")
    ax.set_yscale("log")
    ax.legend(title="selection_method")
    fig.tight_layout()
    fig.savefig(output_dir / output_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize(result_csv: Path, datasets_dir: Path, output_dir: Path, time_summary: str = "median_iqr") -> None:
    result_df = pd.read_csv(result_csv)
    missing_cols = sorted(REQUIRED_RESULT_COLUMNS - set(result_df.columns))
    if missing_cols:
        raise ValueError(f"result.csv missing required columns: {missing_cols}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_frames = _load_dataset_frames(result_df, datasets_dir)
    config_labels, config_to_idx, config_cmap, config_norm = _build_config_style(dataset_frames)
    _plot_scatter_ground_truth(
        result_df,
        dataset_frames,
        config_labels,
        config_to_idx,
        config_cmap,
        config_norm,
        output_dir,
    )
    _plot_scatter_selected_mismatch(
        result_df,
        dataset_frames,
        config_labels,
        config_to_idx,
        config_cmap,
        config_norm,
        output_dir,
    )
    _plot_computation_time_bar(result_df, output_dir, summary=time_summary, output_name="time_bar.png")

    if "relationship_mode" in result_df.columns:
        modes = sorted(result_df["relationship_mode"].dropna().astype(str).unique().tolist())
        for mode in modes:
            mode_df = result_df[result_df["relationship_mode"].astype(str) == mode].copy()
            mode_token = _sanitize_filename_token(mode)
            _plot_computation_time_bar(
                mode_df,
                output_dir,
                summary=time_summary,
                output_name=f"time_bar_{mode_token}.png",
                title_context=f"relationship_mode={mode}",
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize protocol result.csv")
    parser.add_argument("--result-csv", type=Path, default=Path("result.csv"))
    parser.add_argument("--datasets-dir", type=Path, default=Path("tests/tmp"))
    parser.add_argument("--output-dir", type=Path, default=Path("visuals"))
    parser.add_argument(
        "--time-summary",
        choices=["mean_std", "median_iqr"],
        default="median_iqr",
        help="Aggregation for time_bar whiskers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualize(args.result_csv, args.datasets_dir, args.output_dir, time_summary=args.time_summary)
    print(f"Saved plots to: {args.output_dir}")
