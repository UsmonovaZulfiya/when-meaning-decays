"""Create Llama 3.1 8B Pipeline A ablation comparison figures.

Reads the already-computed ablation overview tables and compares:
  - Llama 3.1 8B (T=0.8)
  - Llama 3.1 8B (T=0.2)
  - Llama 3.1 8B Q4 (T=0.8)

The output includes an all-categories 1x3 metric panel and a 4x3 category grid
for Pipeline A only. Confidence bands come from the existing bootstrap CI
columns in the ablation overview tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.basic_evaluation_set.run_model_comparison_analysis import (
    CATEGORY_ORDER,
    require_matplotlib,
    repo_root,
)


METRICS = [
    "exact_accuracy",
    "word_guess_semantic_similarity",
    "normalized_entropy",
]

METRIC_TITLES = {
    "exact_accuracy": "Exact Accuracy",
    "word_guess_semantic_similarity": "Semantic Similarity",
    "normalized_entropy": "Normalized Entropy",
}

MODEL_ORDER = [
    "Llama 3.1 8B (T=0.8)",
    "Llama 3.1 8B (T=0.2)",
    "Llama 3.1 8B Q4 (T=0.8)",
]

MODEL_COLORS = {
    "Llama 3.1 8B (T=0.8)": "#1f77b4",
    "Llama 3.1 8B (T=0.2)": "#ff7f0e",
    "Llama 3.1 8B Q4 (T=0.8)": "#e377c2",
}

PANEL_TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
ROW_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12


def load_llama_pipeline_a_overview(temperature_overview: Path, quant_overview: Path) -> pd.DataFrame:
    temp = pd.read_csv(temperature_overview)
    quant = pd.read_csv(quant_overview)

    temp_models = {
        "Llama 3.1 8B (T=0.8)",
        "Llama 3.1 8B (T=0.2)",
    }
    quant_models = {"Llama 3.1 8B Q4 (T=0.8)"}

    temp = temp[
        temp["pipeline"].astype(str).eq("pipeline_a")
        & temp["model_name"].astype(str).isin(temp_models)
    ].copy()
    quant = quant[
        quant["pipeline"].astype(str).eq("pipeline_a")
        & quant["model_name"].astype(str).isin(quant_models)
    ].copy()

    overview = pd.concat([temp, quant], ignore_index=True)
    overview["step"] = pd.to_numeric(overview["step"], errors="coerce")
    overview = overview[overview["step"].notna()].copy()
    overview["model_name"] = pd.Categorical(overview["model_name"], MODEL_ORDER, ordered=True)
    overview["category"] = pd.Categorical(
        overview["category"],
        ["All Categories", *CATEGORY_ORDER],
        ordered=True,
    )
    overview = overview.sort_values(["category", "model_name", "step"]).reset_index(drop=True)

    missing = [metric for metric in METRICS if metric not in overview.columns]
    if missing:
        raise ValueError(f"Overview is missing required metric columns: {', '.join(missing)}")
    return overview


def set_metric_limits(ax, metric_col: str) -> None:
    if metric_col in {"exact_accuracy", "word_guess_semantic_similarity", "normalized_entropy"}:
        ax.set_ylim(-0.03, 1.03)


def plot_metric_panel(
    ax,
    overview: pd.DataFrame,
    category: str,
    metric_col: str,
    show_ylabel: bool,
    show_xlabel: bool,
    title: str,
) -> list:
    handles = []
    data = overview[overview["category"].astype(str).eq(category)].copy()

    for model in MODEL_ORDER:
        sub = data[data["model_name"].astype(str).eq(model)].sort_values("step")
        if sub.empty:
            continue
        line = ax.plot(
            sub["step"],
            sub[metric_col],
            marker="o",
            markersize=3.4,
            linewidth=1.8,
            label=model,
            color=MODEL_COLORS[model],
        )[0]
        handles.append(line)

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
                    color=MODEL_COLORS[model],
                    alpha=0.11,
                    linewidth=0,
                )

    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel(METRIC_TITLES[metric_col], fontsize=AXIS_LABEL_FONTSIZE)
    if show_xlabel:
        ax.set_xlabel("Step", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(range(0, 11))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.28, linewidth=0.6)
    set_metric_limits(ax, metric_col)
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


def save_three_metric_panel(
    overview: pd.DataFrame,
    category: str,
    output_path: Path,
    figure_title: str | None = None,
) -> None:
    plt = require_matplotlib()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(17.0, 4.5), sharex=True)
    handles = []

    for col_idx, metric_col in enumerate(METRICS):
        handles.extend(
            plot_metric_panel(
                ax=axes[col_idx],
                overview=overview,
                category=category,
                metric_col=metric_col,
                show_ylabel=col_idx == 0,
                show_xlabel=True,
                title=METRIC_TITLES[metric_col],
            )
        )

    legend_handles, legend_labels = unique_legend(handles)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=max(len(legend_labels), 1),
        mode="expand",
        bbox_to_anchor=(0.12, 0.01, 0.76, 0.04),
        handlelength=2.6,
        columnspacing=1.0,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    if figure_title:
        fig.suptitle(figure_title, fontsize=15)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95 if figure_title else 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_category_grid(overview: pd.DataFrame, output_path: Path) -> None:
    plt = require_matplotlib()
    fig, axes = plt.subplots(
        len(CATEGORY_ORDER),
        len(METRICS),
        figsize=(17.2, 12.0),
        sharex=True,
    )
    handles = []

    for row_idx, category in enumerate(CATEGORY_ORDER):
        for col_idx, metric_col in enumerate(METRICS):
            handles.extend(
                plot_metric_panel(
                    ax=axes[row_idx, col_idx],
                    overview=overview,
                    category=category,
                    metric_col=metric_col,
                    show_ylabel=col_idx == 0,
                    show_xlabel=row_idx == len(CATEGORY_ORDER) - 1,
                    title=METRIC_TITLES[metric_col] if row_idx == 0 else "",
                )
            )
            if col_idx == 0:
                axes[row_idx, col_idx].text(
                    -0.23,
                    0.5,
                    category,
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
        bbox_to_anchor=(0.12, 0.01, 0.76, 0.04),
        handlelength=2.6,
        columnspacing=1.0,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    base_dir = repo_root() / "data_analysis" / "results_analysis" / "ablation_studies"
    parser = argparse.ArgumentParser(description="Plot Llama 8B Pipeline A ablation comparison.")
    parser.add_argument(
        "--temperature-overview",
        type=Path,
        default=base_dir / "temperature_comparison" / "tables" / "overview_metrics_all_selected_pipelines.csv",
    )
    parser.add_argument(
        "--quant-overview",
        type=Path,
        default=base_dir / "quantization_comparison" / "tables" / "overview_metrics_all_selected_pipelines.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "llama8b_pipeline_a_comparison",
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
    overview = load_llama_pipeline_a_overview(args.temperature_overview, args.quant_overview)

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "combined_figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    overview.to_csv(tables_dir / "overview_metrics_llama8b_pipeline_a.csv", index=False)

    for fmt in args.formats:
        save_three_metric_panel(
            overview=overview,
            category="All Categories",
            output_path=figures_dir / f"all_categories_three_metric_panel.{fmt}",
        )
        save_category_grid(
            overview=overview,
            output_path=figures_dir / f"categories_three_metric_grid.{fmt}",
        )
        for category in CATEGORY_ORDER:
            save_three_metric_panel(
                overview=overview,
                category=category,
                output_path=figures_dir / f"{category.lower().replace('-', '_')}_three_metric_panel.{fmt}",
                figure_title=category,
            )

    print(f"Wrote Llama 8B Pipeline A comparison figures to: {figures_dir}")
    print(f"Wrote filtered overview to: {tables_dir / 'overview_metrics_llama8b_pipeline_a.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
