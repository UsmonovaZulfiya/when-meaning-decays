"""Create compact description-similarity figures from existing metric tables.

This is a visual-only pass over the already-computed
all_guess_similarity_description_similarity results. It keeps the data fixed
and writes a paper-friendly Pipeline A/B x category grid without a supertitle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .run_model_comparison_analysis import (
        CATEGORY_ORDER,
        MODEL_COLORS,
        MODEL_ORDER,
        PIPELINE_LABELS,
        require_matplotlib,
        repo_root,
    )
except ImportError:  # Direct execution from this directory is supported.
    from run_model_comparison_analysis import (
        CATEGORY_ORDER,
        MODEL_COLORS,
        MODEL_ORDER,
        PIPELINE_LABELS,
        require_matplotlib,
        repo_root,
    )


METRIC_COL = "description_semantic_similarity_to_step0"
METRIC_LABEL = "Semantic Similarity"
PIPELINES = ["pipeline_a", "pipeline_b"]

PANEL_TITLE_FONTSIZE = 11
AXIS_LABEL_FONTSIZE = 11
TICK_LABEL_FONTSIZE = 9
ROW_LABEL_FONTSIZE = 12
CATEGORY_LEGEND_FONTSIZE = 10
ALL_CATEGORIES_LEGEND_FONTSIZE = 7.5


def load_overview(path: Path) -> pd.DataFrame:
    overview = pd.read_csv(path)
    if METRIC_COL not in overview.columns:
        raise ValueError(f"{path} is missing {METRIC_COL!r}")
    overview["step"] = pd.to_numeric(overview["step"], errors="coerce")
    overview = overview[overview["step"].notna()].copy()
    return overview


def aggregate_all_categories(overview: pd.DataFrame) -> pd.DataFrame:
    all_rows = overview[overview["category"].astype(str).eq("All Categories")].copy()
    if not all_rows.empty:
        return all_rows

    rows = []
    weight_col = "description_similarity_n"
    for key, sub in overview.groupby(["model_name", "pipeline", "step"], observed=True):
        values = pd.to_numeric(sub[METRIC_COL], errors="coerce")
        valid = values.notna()
        if not valid.any():
            continue
        if weight_col in sub.columns:
            weights = pd.to_numeric(sub.loc[valid, weight_col], errors="coerce").fillna(0)
            score = np.average(values.loc[valid], weights=weights) if weights.sum() else values.loc[valid].mean()
        else:
            score = values.loc[valid].mean()
        rows.append(
            {
                "model_name": key[0],
                "pipeline": key[1],
                "category": "All Categories",
                "step": key[2],
                METRIC_COL: score,
            }
        )
    return pd.DataFrame(rows)


def model_order_for(overview: pd.DataFrame) -> list[str]:
    present = set(overview["model_name"].astype(str))
    ordered = [model for model in MODEL_ORDER if model in present]
    extra = sorted(present - set(ordered))
    return ordered + extra


def plot_panel(
    ax,
    overview: pd.DataFrame,
    pipeline: str,
    category: str,
    models: list[str],
    show_ylabel: bool,
    show_xlabel: bool,
    title: str,
) -> list:
    handles = []
    data = overview[
        overview["pipeline"].astype(str).eq(pipeline)
        & overview["category"].astype(str).eq(category)
    ].copy()

    for model in models:
        sub = data[data["model_name"].astype(str).eq(model)].sort_values("step")
        if sub.empty:
            continue
        line = ax.plot(
            sub["step"],
            sub[METRIC_COL],
            marker="o",
            markersize=2.8,
            linewidth=1.35,
            label=model,
            color=MODEL_COLORS.get(model),
        )[0]
        handles.append(line)

        ci_low = f"{METRIC_COL}_ci_low"
        ci_high = f"{METRIC_COL}_ci_high"
        if ci_low in sub.columns and ci_high in sub.columns:
            lower = pd.to_numeric(sub[ci_low], errors="coerce")
            upper = pd.to_numeric(sub[ci_high], errors="coerce")
            if lower.notna().any() and upper.notna().any():
                ax.fill_between(
                    sub["step"],
                    lower,
                    upper,
                    color=MODEL_COLORS.get(model),
                    alpha=0.10,
                    linewidth=0,
                )

    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE, pad=6)
    if show_ylabel:
        ax.set_ylabel(METRIC_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("Step", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(range(0, 11))
    ax.set_ylim(-0.02, 1.03)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.22, linewidth=0.55)
    return handles


def unique_legend(handles: list) -> tuple[list, list[str]]:
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle in handles:
        label = handle.get_label()
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    return unique_handles, unique_labels


def save_category_grid(overview: pd.DataFrame, output_path: Path) -> None:
    plt = require_matplotlib()
    models = model_order_for(overview)
    fig, axes = plt.subplots(2, len(CATEGORY_ORDER), figsize=(13.8, 6.0), sharex=True, sharey=True)
    handles = []

    for row_idx, pipeline in enumerate(PIPELINES):
        for col_idx, category in enumerate(CATEGORY_ORDER):
            handles.extend(
                plot_panel(
                    ax=axes[row_idx, col_idx],
                    overview=overview,
                    pipeline=pipeline,
                    category=category,
                    models=models,
                    show_ylabel=col_idx == 0,
                    show_xlabel=row_idx == 1,
                    title=category if row_idx == 0 else "",
                )
            )
            if col_idx == 0:
                axes[row_idx, col_idx].text(
                    -0.22,
                    0.5,
                    PIPELINE_LABELS[pipeline],
                    transform=axes[row_idx, col_idx].transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=ROW_LABEL_FONTSIZE,
                )

    legend_handles, legend_labels = unique_legend(handles)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=max(len(legend_labels), 1),
        mode="expand",
        bbox_to_anchor=(0.12, 0.01, 0.76, 0.045),
        handlelength=2.5,
        columnspacing=0.9,
        frameon=True,
        fontsize=CATEGORY_LEGEND_FONTSIZE,
    )
    fig.tight_layout(rect=[0.02, 0.08, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def save_all_categories_grid(overview: pd.DataFrame, output_path: Path) -> None:
    plt = require_matplotlib()
    all_overview = aggregate_all_categories(overview)
    models = model_order_for(overview)
    fig, axes = plt.subplots(1, len(PIPELINES), figsize=(8.2, 3.5), sharex=True, sharey=True)
    handles = []

    for col_idx, pipeline in enumerate(PIPELINES):
        handles.extend(
            plot_panel(
                ax=axes[col_idx],
                overview=all_overview,
                pipeline=pipeline,
                category="All Categories",
                models=models,
                show_ylabel=col_idx == 0,
                show_xlabel=True,
                title=PIPELINE_LABELS[pipeline],
            )
        )

    legend_handles, legend_labels = unique_legend(handles)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=max(len(legend_labels), 1),
        mode="expand",
        bbox_to_anchor=(0.05, 0.01, 0.90, 0.055),
        handlelength=2.0,
        columnspacing=0.5,
        frameon=True,
        fontsize=ALL_CATEGORIES_LEGEND_FONTSIZE,
    )
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    result_dir = repo_root() / "data_analysis" / "results_analysis" / "all_guess_similarity_description_similarity"
    parser = argparse.ArgumentParser(description="Create compact description-similarity figures.")
    parser.add_argument(
        "--overview",
        type=Path,
        default=result_dir / "tables" / "overview_metrics_all_selected_pipelines.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=result_dir / "combined_figures",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    overview = load_overview(args.overview)
    for fmt in args.formats:
        save_category_grid(
            overview,
            args.output_dir / f"description_similarity_categories_compact.{fmt}",
        )
        save_all_categories_grid(
            overview,
            args.output_dir / f"description_similarity_all_categories_compact.{fmt}",
        )
    print(f"Wrote compact description-similarity figures to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
