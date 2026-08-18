"""Build compact multi-panel Semantic Drift comparison figures.

This script reads an overview CSV produced by run_model_comparison_analysis.py
and creates:
  1. A 2x3 all-categories figure: pipelines x metrics.
  2. One 2x4 category-grid figure per metric: pipelines x categories.

Examples:
    python data_analysis/basic_evaluation_set/plot_combined_model_comparison.py
    python data_analysis/basic_evaluation_set/plot_combined_model_comparison.py --overview data_analysis/results_analysis/all_exact_entropy_guess_similarity/tables/overview_metrics_all_selected_pipelines.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .run_model_comparison_analysis import (
        CATEGORY_ORDER,
        METRIC_TITLES,
        MODEL_COLORS,
        MODEL_ORDER,
        PIPELINE_LABELS,
        load_human_accuracy,
        require_matplotlib,
        repo_root,
        slugify,
    )
except ImportError:  # Direct execution from this directory is supported.
    from run_model_comparison_analysis import (
        CATEGORY_ORDER,
        METRIC_TITLES,
        MODEL_COLORS,
        MODEL_ORDER,
        PIPELINE_LABELS,
        load_human_accuracy,
        require_matplotlib,
        repo_root,
        slugify,
    )


DEFAULT_METRICS = [
    "exact_accuracy",
    "word_guess_semantic_similarity",
    "normalized_entropy",
]

METRIC_SHORT_TITLES = {
    "exact_accuracy": "Exact Accuracy",
    "word_guess_semantic_similarity": "Semantic Similarity",
    "shannon_entropy": "Shannon Entropy",
    "normalized_entropy": "Normalized Entropy",
}

HUMAN_LABEL = "Human Study"
HUMAN_COLOR = "#666666"

# Font sizes for the compact figures can be adjusted with these constants.
PANEL_TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
ROW_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 12


def find_latest_overview(base_dir: Path, metrics: list[str]) -> Path:
    candidates = sorted(
        base_dir.glob("**/overview_metrics_all_selected_pipelines.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            header = pd.read_csv(candidate, nrows=0).columns
        except Exception:
            continue
        if all(metric in header for metric in metrics):
            return candidate
    required = ", ".join(metrics)
    raise FileNotFoundError(
        f"No overview_metrics_all_selected_pipelines.csv under {base_dir} contains: {required}"
    )


def load_overview(path: Path, models: list[str] | None) -> pd.DataFrame:
    overview = pd.read_csv(path)
    if models:
        overview = overview[overview["model_name"].isin(models)].copy()
    overview["step"] = pd.to_numeric(overview["step"], errors="coerce")
    return overview[overview["step"].notna()].copy()


def load_human_semantic(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["category", "step", "human_word_guess_semantic_similarity"])

    human = pd.read_csv(path)
    if {"category", "step", "word_similarity"}.issubset(human.columns):
        category_df = (
            human.groupby(["category", "step"], as_index=False, observed=True)
            .agg(human_word_guess_semantic_similarity=("word_similarity", "mean"))
        )
        overall = (
            human.groupby("step", as_index=False, observed=True)
            .agg(human_word_guess_semantic_similarity=("word_similarity", "mean"))
            .assign(category="All Categories")
        )
        return pd.concat([category_df, overall], ignore_index=True)

    if {"step", "score"}.issubset(human.columns):
        return human.rename(columns={"score": "human_word_guess_semantic_similarity"}).assign(
            category="All Categories"
        )

    raise ValueError(
        "Human semantic CSV must contain either category/step/word_similarity "
        "or step/score columns."
    )


def human_line_for_metric(
    metric_col: str,
    pipeline: str,
    category: str,
    human_accuracy: pd.DataFrame,
    human_semantic: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    if pipeline != "pipeline_a":
        return None, None
    if metric_col == "exact_accuracy" and not human_accuracy.empty:
        sub = human_accuracy[human_accuracy["category"].eq(category)].copy()
        return sub.rename(columns={"human_exact_accuracy": "score"}), "score"
    if metric_col == "word_guess_semantic_similarity" and not human_semantic.empty:
        sub = human_semantic[human_semantic["category"].eq(category)].copy()
        return sub.rename(columns={"human_word_guess_semantic_similarity": "score"}), "score"
    return None, None


def set_metric_limits(ax, metric_col: str, values: pd.Series) -> None:
    if metric_col in {"exact_accuracy", "word_guess_semantic_similarity", "normalized_entropy"}:
        ax.set_ylim(-0.03, 1.03)
        return
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return
    lower = min(0.0, float(numeric.min()))
    upper = float(numeric.max())
    padding = max((upper - lower) * 0.08, 0.1)
    ax.set_ylim(lower - padding, upper + padding)


def plot_panel(
    ax,
    overview: pd.DataFrame,
    metric_col: str,
    pipeline: str,
    category: str,
    models: list[str],
    human_accuracy: pd.DataFrame,
    human_semantic: pd.DataFrame,
    show_ylabel: bool,
    show_xlabel: bool,
    show_title: str,
) -> list:
    lines = []
    data = overview[
        overview["pipeline"].eq(pipeline)
        & overview["category"].astype(str).eq(category)
    ].copy()

    for model in models:
        sub = data[data["model_name"].eq(model)].sort_values("step")
        if sub.empty:
            continue
        line = ax.plot(
            sub["step"],
            sub[metric_col],
            marker="o",
            markersize=3.2,
            linewidth=1.5,
            label=model,
            color=MODEL_COLORS.get(model),
        )[0]
        lines.append(line)

        ci_low = f"{metric_col}_ci_low"
        ci_high = f"{metric_col}_ci_high"
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

    human_sub, human_col = human_line_for_metric(
        metric_col,
        pipeline,
        category,
        human_accuracy,
        human_semantic,
    )
    if human_sub is not None and not human_sub.empty:
        human_sub = human_sub.sort_values("step")
        line = ax.plot(
            human_sub["step"],
            human_sub[human_col],
            linestyle="--",
            linewidth=1.8,
            color=HUMAN_COLOR,
            label=HUMAN_LABEL,
        )[0]
        lines.append(line)

    ax.set_title(show_title, fontsize=PANEL_TITLE_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel(
            METRIC_SHORT_TITLES.get(metric_col, METRIC_TITLES.get(metric_col, metric_col)),
            fontsize=AXIS_LABEL_FONTSIZE,
        )
    if show_xlabel:
        ax.set_xlabel("Step", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(range(0, 11))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.28, linewidth=0.6)
    set_metric_limits(ax, metric_col, data[metric_col] if metric_col in data else pd.Series(dtype=float))
    return lines


def unique_legend_handles(handles: list) -> tuple[list, list[str]]:
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


def save_all_categories_figure(
    overview: pd.DataFrame,
    metrics: list[str],
    models: list[str],
    output_path: Path,
    human_accuracy: pd.DataFrame,
    human_semantic: pd.DataFrame,
) -> None:
    plt = require_matplotlib()
    fig, axes = plt.subplots(2, len(metrics), figsize=(17.0, 7.2), sharex=True)
    handles = []

    for row_idx, pipeline in enumerate(["pipeline_a", "pipeline_b"]):
        for col_idx, metric_col in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            title = METRIC_SHORT_TITLES.get(metric_col, metric_col) if row_idx == 0 else ""
            handles.extend(
                plot_panel(
                    ax=ax,
                    overview=overview,
                    metric_col=metric_col,
                    pipeline=pipeline,
                    category="All Categories",
                    models=models,
                    human_accuracy=human_accuracy,
                    human_semantic=human_semantic,
                    show_ylabel=False,
                    show_xlabel=row_idx == 1,
                    show_title=title,
                )
            )
            if col_idx == 0:
                ax.text(
                    -0.11,
                    0.5,
                    PIPELINE_LABELS[pipeline],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=ROW_LABEL_FONTSIZE,
                    fontweight="normal",
                )

    legend_handles, legend_labels = unique_legend_handles(handles)
    #fig.suptitle("Semantic Drift Model Comparison - All Categories", fontsize=13)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=max(len(legend_labels), 1),
        mode="expand",
        bbox_to_anchor=(0.08, 0.01, 0.84, 0.04),
        handlelength=2.6,
        columnspacing=1.0,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_category_metric_figure(
    overview: pd.DataFrame,
    metric_col: str,
    models: list[str],
    output_path: Path,
    human_accuracy: pd.DataFrame,
    human_semantic: pd.DataFrame,
) -> None:
    plt = require_matplotlib()
    fig, axes = plt.subplots(2, len(CATEGORY_ORDER), figsize=(17.2, 7.2), sharex=True)
    handles = []

    for row_idx, pipeline in enumerate(["pipeline_a", "pipeline_b"]):
        for col_idx, category in enumerate(CATEGORY_ORDER):
            ax = axes[row_idx, col_idx]
            title = category if row_idx == 0 else ""
            handles.extend(
                plot_panel(
                    ax=ax,
                    overview=overview,
                    metric_col=metric_col,
                    pipeline=pipeline,
                    category=category,
                    models=models,
                    human_accuracy=human_accuracy,
                    human_semantic=human_semantic,
                    show_ylabel=col_idx == 0,
                    show_xlabel=row_idx == 1,
                    show_title=title,
                )
            )
            if col_idx == 0:
                ax.text(
                    -0.23,
                    0.5,
                    PIPELINE_LABELS[pipeline],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=ROW_LABEL_FONTSIZE,
                    fontweight="normal",
                )

    legend_handles, legend_labels = unique_legend_handles(handles)
    #fig.suptitle(f"{METRIC_SHORT_TITLES.get(metric_col, metric_col)} by Category", fontsize=)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=max(len(legend_labels), 1),
        mode="expand",
        bbox_to_anchor=(0.08, 0.01, 0.84, 0.04),
        handlelength=2.6,
        columnspacing=1.0,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    fig.tight_layout(rect=[0.02, 0.08, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create compact multi-panel comparison figures.")
    parser.add_argument(
        "--overview",
        type=Path,
        help="Overview metrics CSV from run_model_comparison_analysis.py. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "data_analysis" / "results_analysis" / "combined_figures",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--human-accuracy-csv",
        type=Path,
        default=repo_root() / "human_study" / "all_chains_grouped_by_category_percentages.csv",
        help="Human-study exact-accuracy category percentage CSV.",
    )
    parser.add_argument(
        "--human-semantic-csv",
        type=Path,
        default=repo_root()
        / "data_analysis"
        / "results_analysis"
        / "word_stability"
        / "human_word_guess_semantic_similarity_by_word.csv",
        help="Human-study semantic-similarity CSV, preferably the by-word export.",
    )
    parser.add_argument(
        "--entropy-metric",
        choices=["normalized_entropy", "shannon_entropy"],
        default="normalized_entropy",
        help="Entropy column to use in the compact figures.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional subset of model names. Defaults to all known models present in the overview CSV.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Output figure formats.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    metrics = ["exact_accuracy", "word_guess_semantic_similarity", args.entropy_metric, ]
    overview_path = (
        args.overview.resolve()
        if args.overview
        else find_latest_overview(repo_root() / "data_analysis" / "results_analysis", metrics)
    )
    overview = load_overview(overview_path, args.models)
    missing = [metric for metric in metrics if metric not in overview.columns]
    if missing:
        raise ValueError(f"Overview CSV is missing required columns: {', '.join(missing)}")

    models = args.models or [model for model in MODEL_ORDER if model in set(overview["model_name"])]
    human_accuracy = load_human_accuracy(args.human_accuracy_csv) if args.human_accuracy_csv.exists() else pd.DataFrame()
    human_semantic = load_human_semantic(args.human_semantic_csv)
    if human_semantic.empty:
        print(
            f"Warning: no human semantic-similarity line loaded from {args.human_semantic_csv}. "
            "Semantic panels will omit the gray human line.",
            file=sys.stderr,
        )

    print(f"Using overview: {overview_path}")
    print(f"Models: {', '.join(models)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in args.formats:
        save_all_categories_figure(
            overview=overview,
            metrics=metrics,
            models=models,
            output_path=args.output_dir / f"all_categories_six_panel.{fmt}",
            human_accuracy=human_accuracy,
            human_semantic=human_semantic,
        )
        for metric in metrics:
            save_category_metric_figure(
                overview=overview,
                metric_col=metric,
                models=models,
                output_path=args.output_dir / f"categories_pipeline_grid_{slugify(metric)}.{fmt}",
                human_accuracy=human_accuracy,
                human_semantic=human_semantic,
            )

    print(f"Wrote combined figures to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
