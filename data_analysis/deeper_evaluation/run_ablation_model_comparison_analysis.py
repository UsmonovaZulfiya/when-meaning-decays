"""Run Semantic Drift ablation analyses.

This script keeps the existing model-comparison evaluator intact and builds
two ablation-specific datasets/results:

1. Temperature comparison:
   Llama 3.1 8B T=0.2 vs Llama 3.1 8B T=0.8,
   Gemma 3 4B T=0.2 vs Gemma 3 4B T=0.8, both pipelines.
2. Quantization comparison:
   Llama 3.1 8B T=0.8 vs Llama 3.1 8B Q4 T=0.8, Pipeline A only.

The metric calculations are delegated to run_model_comparison_analysis.py so
the bootstrap confidence intervals and MPNet semantic-similarity logic stay
consistent with the main paper figures.

Examples:
    python data_analysis/deeper_evaluation/run_ablation_model_comparison_analysis.py
    python data_analysis/deeper_evaluation/run_ablation_model_comparison_analysis.py --study temperature --formats png pdf
    python data_analysis/deeper_evaluation/run_ablation_model_comparison_analysis.py --study quantization --skip-eval
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.basic_evaluation_set import run_model_comparison_analysis as base
from data_analysis.basic_evaluation_set.run_model_comparison_analysis import (
    CATEGORY_ORDER,
    METRIC_TITLES,
    OutputPaths,
    aggregate_all_categories,
    build_output_paths,
    compute_entropy_diagnostics,
    compute_fast_metrics,
    compute_guess_similarity,
    count_csv_rows,
    load_sentence_transformer,
    merge_metric_tables,
    require_matplotlib,
    save_entropy_diagnostics,
    save_word_tables,
    slugify,
)


OUTPUT_COLUMNS = [
    "model_name",
    "category",
    "original_word",
    "instance_id",
    "step",
    "description",
    "guess",
    "prompt_mode",
]

PROMPT_MODE_BY_PIPELINE = {
    "pipeline_a": "pipeline_A_guess_then_describe_guessed_word",
    "pipeline_b": "pipeline_B_paraphrase_description",
}

PIPELINE_LABELS = {
    "pipeline_a": "Pipeline A",
    "pipeline_b": "Pipeline B",
}

PLOT_METRICS = [
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


@dataclass(frozen=True)
class CsvSource:
    name: str
    kind: str
    label: str
    paths: tuple[Path, ...]
    model_filter: str | None = None
    pipelines: tuple[str, ...] = ("pipeline_a", "pipeline_b")
    force_pipeline: str | None = None


@dataclass(frozen=True)
class AblationStudy:
    slug: str
    title: str
    sources: tuple[CsvSource, ...]
    model_order: tuple[str, ...]
    model_colors: dict[str, str]
    pipelines: tuple[str, ...]


def repo_root() -> Path:
    return PROJECT_ROOT


def is_raw_result_csv(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() != ".csv":
        return False
    if "checkpoint" in name:
        return False
    return "results_from" in name or name.startswith("results_")


def discover_raw_results(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*.csv") if is_raw_result_csv(path)))


def infer_pipeline(path: Path, prompt_mode: str | None = None) -> str:
    path_text = str(path).replace("\\", "/").lower()
    name = path.name.lower()
    prompt = (prompt_mode or "").lower()

    if "/pipeline_b/" in path_text or "pipeline_b" in name or "pipelineb" in name:
        return "pipeline_b"
    if "exp-2" in name or "exp_2" in name or prompt.startswith("pipeline_b"):
        return "pipeline_b"
    if "/pipeline_a/" in path_text or "pipeline_a" in name:
        return "pipeline_a"
    if "exp-1" in name or "exp_1" in name or prompt.startswith("pipeline_a"):
        return "pipeline_a"

    raise ValueError(f"Could not infer pipeline for {path}")


def required(row: dict[str, str], column: str, path: Path) -> str:
    if column not in row:
        raise KeyError(f"Missing required column {column!r} in {path}")
    return row[column]


def rows_from_raw_source(source: CsvSource) -> Iterable[dict[str, str]]:
    for path in source.paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                pipeline = source.force_pipeline or infer_pipeline(
                    path,
                    row.get("Prompt_Mode"),
                )
                if pipeline not in source.pipelines:
                    continue
                yield {
                    "model_name": source.label,
                    "category": required(row, "Category", path),
                    "original_word": required(row, "Word", path),
                    "instance_id": required(row, "Instance_ID", path),
                    "step": required(row, "Step", path),
                    "description": required(row, "Description", path),
                    "guess": required(row, "Guess", path),
                    "prompt_mode": PROMPT_MODE_BY_PIPELINE[pipeline],
                }


def rows_from_unified_source(source: CsvSource) -> Iterable[dict[str, str]]:
    for path in source.paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if source.model_filter and row.get("model_name") != source.model_filter:
                    continue
                pipeline = infer_pipeline(path, row.get("prompt_mode"))
                if pipeline not in source.pipelines:
                    continue
                yield {
                    "model_name": source.label,
                    "category": required(row, "category", path),
                    "original_word": required(row, "original_word", path),
                    "instance_id": required(row, "instance_id", path),
                    "step": required(row, "step", path),
                    "description": required(row, "description", path),
                    "guess": required(row, "guess", path),
                    "prompt_mode": PROMPT_MODE_BY_PIPELINE[pipeline],
                }


def iter_source_rows(source: CsvSource) -> Iterable[dict[str, str]]:
    if source.kind == "raw":
        yield from rows_from_raw_source(source)
    elif source.kind == "unified":
        yield from rows_from_unified_source(source)
    else:
        raise ValueError(f"Unknown source kind {source.kind!r}")


def write_unified_dataset(study: AblationStudy, output_csv: Path) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[tuple[str, str, str], dict[str, int]] = {}
    source_rows = []

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for source in study.sources:
            count = 0
            blank_guess_count = 0
            for row in iter_source_rows(source):
                writer.writerow(row)
                count += 1
                is_blank_guess = not str(row["guess"]).strip()
                if is_blank_guess:
                    blank_guess_count += 1
                pipeline = infer_pipeline(output_csv, row["prompt_mode"])
                key = (row["model_name"], pipeline, row["category"])
                if key not in summary:
                    summary[key] = {"rows": 0, "blank_guess_rows": 0}
                summary[key]["rows"] += 1
                if is_blank_guess:
                    summary[key]["blank_guess_rows"] += 1
            source_rows.append(
                {
                    "study": study.slug,
                    "source": source.name,
                    "kind": source.kind,
                    "label": source.label,
                    "rows": count,
                    "blank_guess_rows": blank_guess_count,
                    "paths": "; ".join(str(path) for path in source.paths),
                }
            )

    summary_rows = [
        {
            "study": study.slug,
            "model_name": model,
            "pipeline": pipeline,
            "category": category,
            "rows": counts["rows"],
            "blank_guess_rows": counts["blank_guess_rows"],
            "nonblank_guess_rows": counts["rows"] - counts["blank_guess_rows"],
        }
        for (model, pipeline, category), counts in sorted(summary.items())
    ]
    summary_df = pd.DataFrame(summary_rows)
    pd.DataFrame(source_rows).to_csv(output_csv.with_name(f"{output_csv.stem}_sources.csv"), index=False)
    summary_df.to_csv(output_csv.with_name(f"{output_csv.stem}_summary.csv"), index=False)
    return summary_df


def temperature_study(root: Path) -> AblationStudy:
    outputs = root / "datasets"
    ablation = outputs / "ablation"
    llama_t02 = discover_raw_results(ablation / "llama-3.1-8b")

    return AblationStudy(
        slug="temperature_comparison",
        title="Temperature Ablation",
        sources=(
            CsvSource(
                name="llama_8b_t02",
                kind="raw",
                label="Llama 3.1 8B (T=0.2)",
                paths=llama_t02,
            ),
            CsvSource(
                name="llama_8b_t08",
                kind="unified",
                label="Llama 3.1 8B (T=0.8)",
                paths=(outputs / "unified_semantic_drift_results.csv",),
                model_filter="Llama 3.1 8B Instruct",
            ),
            CsvSource(
                name="gemma_4b_t02",
                kind="raw",
                label="Gemma 3 4B (T=0.2)",
                paths=(
                    ablation / "Results_gemma-3-4b-it-temp-02-exp-1.csv",
                    ablation / "Results_gemma-3-4b-it-temp-02-exp-2.csv",
                ),
            ),
            CsvSource(
                name="gemma_4b_t08",
                kind="unified",
                label="Gemma 3 4B (T=0.8)",
                paths=(outputs / "unified_semantic_drift_results.csv",),
                model_filter="Gemma 3 4B IT",
            ),
        ),
        model_order=(
            "Llama 3.1 8B (T=0.2)",
            "Llama 3.1 8B (T=0.8)",
            "Gemma 3 4B (T=0.2)",
            "Gemma 3 4B (T=0.8)",
        ),
        model_colors={
            "Llama 3.1 8B (T=0.2)": "#1f77b4",
            "Llama 3.1 8B (T=0.8)": "#ff7f0e",
            "Gemma 3 4B (T=0.2)": "#2ca02c",
            "Gemma 3 4B (T=0.8)": "#d62728",
        },
        pipelines=("pipeline_a", "pipeline_b"),
    )


def quantization_study(root: Path) -> AblationStudy:
    outputs = root / "datasets"
    ablation = outputs / "ablation"
    return AblationStudy(
        slug="quantization_comparison",
        title="Quantization Ablation",
        sources=(
            CsvSource(
                name="llama_8b_t08",
                kind="unified",
                label="Llama 3.1 8B (T=0.8)",
                paths=(outputs / "unified_semantic_drift_results.csv",),
                model_filter="Llama 3.1 8B Instruct",
                pipelines=("pipeline_a",),
            ),
            CsvSource(
                name="llama_8b_q4_t08",
                kind="raw",
                label="Llama 3.1 8B Q4 (T=0.8)",
                paths=(ablation / "Results_from_golden_llama-31-8B-Instruct_Q4.csv",),
                pipelines=("pipeline_a",),
                force_pipeline="pipeline_a",
            ),
        ),
        model_order=(
            "Llama 3.1 8B (T=0.8)",
            "Llama 3.1 8B Q4 (T=0.8)",
        ),
        model_colors={
            "Llama 3.1 8B (T=0.8)": "#1f77b4",
            "Llama 3.1 8B Q4 (T=0.8)": "#e377c2",
        },
        pipelines=("pipeline_a",),
    )


def validate_sources(study: AblationStudy) -> None:
    missing = [
        path
        for source in study.sources
        for path in source.paths
        if not path.exists()
    ]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing source files for {study.slug}:\n{formatted}")
    for source in study.sources:
        if not source.paths:
            raise FileNotFoundError(f"No source files found for {study.slug}/{source.name}")


def patch_base_model_metadata(study: AblationStudy) -> None:
    base.MODEL_ORDER = list(study.model_order)
    base.MODEL_GROUPS = {"ablation": list(study.model_order)}
    base.MODEL_SLUGS = {model: slugify(model) for model in study.model_order}
    base.MODEL_COLORS = dict(study.model_colors)


def set_metric_limits(ax, metric_col: str, values: pd.Series) -> None:
    if metric_col in {"exact_accuracy", "word_guess_semantic_similarity", "normalized_entropy"}:
        ax.set_ylim(-0.03, 1.03)
        return
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return
    lower = min(0.0, float(numeric.min()))
    upper = float(numeric.max())
    pad = max((upper - lower) * 0.08, 0.02)
    ax.set_ylim(lower - pad, upper + pad)


def plot_panel(
    ax,
    overview: pd.DataFrame,
    metric_col: str,
    pipeline: str,
    category: str,
    models: Iterable[str],
    colors: dict[str, str],
    show_ylabel: bool,
    show_xlabel: bool,
    title: str,
) -> list:
    lines = []
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
            sub[metric_col],
            marker="o",
            markersize=3.2,
            linewidth=1.5,
            label=model,
            color=colors.get(model),
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
                    color=colors.get(model),
                    alpha=0.10,
                    linewidth=0,
                )

    ax.set_title(title, fontsize=14)
    if show_ylabel:
        ax.set_ylabel(METRIC_SHORT_TITLES.get(metric_col, metric_col), fontsize=14)
    if show_xlabel:
        ax.set_xlabel("Step", fontsize=14)
    ax.set_xticks(range(0, 11))
    ax.tick_params(axis="both", labelsize=12)
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
    study: AblationStudy,
    output_path: Path,
) -> None:
    plt = require_matplotlib()
    rows = len(study.pipelines)
    cols = len(PLOT_METRICS)
    height = 7.2 if rows == 2 else 4.5
    fig, axes = plt.subplots(rows, cols, figsize=(17.0, height), sharex=True, squeeze=False)
    handles = []

    for row_idx, pipeline in enumerate(study.pipelines):
        for col_idx, metric_col in enumerate(PLOT_METRICS):
            title = METRIC_SHORT_TITLES.get(metric_col, metric_col) if row_idx == 0 else ""
            handles.extend(
                plot_panel(
                    ax=axes[row_idx, col_idx],
                    overview=overview,
                    metric_col=metric_col,
                    pipeline=pipeline,
                    category="All Categories",
                    models=study.model_order,
                    colors=study.model_colors,
                    show_ylabel=False,
                    show_xlabel=row_idx == rows - 1,
                    title=title,
                )
            )
            if col_idx == 0:
                if rows > 1:
                    axes[row_idx, col_idx].text(
                        -0.11,
                        0.5,
                        PIPELINE_LABELS[pipeline],
                        transform=axes[row_idx, col_idx].transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=16,
                    )
                else:
                    axes[row_idx, col_idx].set_ylabel(PIPELINE_LABELS[pipeline], fontsize=16)

    legend_handles, legend_labels = unique_legend_handles(handles)
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
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.10 if rows == 1 else 0.08, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_category_metric_figure(
    overview: pd.DataFrame,
    study: AblationStudy,
    metric_col: str,
    output_path: Path,
) -> None:
    plt = require_matplotlib()
    rows = len(study.pipelines)
    cols = len(CATEGORY_ORDER)
    height = 7.2 if rows == 2 else 4.5
    fig, axes = plt.subplots(rows, cols, figsize=(17.2, height), sharex=True, squeeze=False)
    handles = []

    for row_idx, pipeline in enumerate(study.pipelines):
        for col_idx, category in enumerate(CATEGORY_ORDER):
            handles.extend(
                plot_panel(
                    ax=axes[row_idx, col_idx],
                    overview=overview,
                    metric_col=metric_col,
                    pipeline=pipeline,
                    category=category,
                    models=study.model_order,
                    colors=study.model_colors,
                    show_ylabel=col_idx == 0,
                    show_xlabel=row_idx == rows - 1,
                    title=category if row_idx == 0 else "",
                )
            )
            if col_idx == 0 and rows > 1:
                axes[row_idx, col_idx].text(
                    -0.23,
                    0.5,
                    PIPELINE_LABELS[pipeline],
                    transform=axes[row_idx, col_idx].transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=16,
                )

    legend_handles, legend_labels = unique_legend_handles(handles)
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
        fontsize=12,
    )
    fig.tight_layout(rect=[0.02, 0.10 if rows == 1 else 0.08, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_combined_figures(
    overview: pd.DataFrame,
    study: AblationStudy,
    output_dir: Path,
    formats: Iterable[str],
) -> None:
    missing = [metric for metric in PLOT_METRICS if metric not in overview.columns]
    if missing:
        raise ValueError(f"Overview CSV is missing columns needed for plots: {', '.join(missing)}")

    for fmt in formats:
        all_name = "all_categories_six_panel" if len(study.pipelines) == 2 else "all_categories_panel"
        save_all_categories_figure(
            overview=overview,
            study=study,
            output_path=output_dir / f"{all_name}.{fmt}",
        )
        for metric_col in PLOT_METRICS:
            save_category_metric_figure(
                overview=overview,
                study=study,
                metric_col=metric_col,
                output_path=output_dir / f"categories_pipeline_grid_{slugify(metric_col)}.{fmt}",
            )


def write_overview_tables(overview: pd.DataFrame, paths: OutputPaths, pipelines: tuple[str, ...]) -> None:
    for pipeline in pipelines:
        pipeline_overview = overview[overview["pipeline"].astype(str).eq(pipeline)].copy()
        pipeline_overview.to_csv(paths.tables / pipeline / "overview_metrics.csv", index=False)
        if not pipeline_overview.empty:
            final_step = int(pd.to_numeric(pipeline_overview["step"], errors="coerce").max())
            pipeline_overview[pipeline_overview["step"].eq(final_step)].to_csv(
                paths.tables / pipeline / "final_step_metrics.csv",
                index=False,
            )
        for metric_col in PLOT_METRICS:
            all_category = aggregate_all_categories(overview, metric_col)
            if not all_category.empty:
                all_category[all_category["pipeline"].astype(str).eq(pipeline)].to_csv(
                    paths.tables / pipeline / f"{metric_col}_all_categories.csv",
                    index=False,
                )
    overview.to_csv(paths.tables / "overview_metrics_all_selected_pipelines.csv", index=False)


def make_output_paths(base_dir: Path, study: AblationStudy) -> OutputPaths:
    paths = build_output_paths(base_dir / "ablation_studies", study.slug)
    for pipeline in study.pipelines:
        (paths.tables / pipeline).mkdir(parents=True, exist_ok=True)
        (paths.figures / pipeline).mkdir(parents=True, exist_ok=True)
    return paths


def run_study(
    study: AblationStudy,
    args: argparse.Namespace,
    encoder,
    cache: base.EmbeddingCache | None,
) -> None:
    validate_sources(study)
    patch_base_model_metadata(study)

    dataset_path = args.output_dir / "ablation_studies" / "datasets" / f"{study.slug}_unified.csv"
    print(f"\n[{study.title}] Writing unified dataset: {dataset_path}")
    summary = write_unified_dataset(study, dataset_path)
    print(summary.to_string(index=False))

    if args.prepare_only:
        return

    paths = make_output_paths(args.output_dir, study)
    disable_progress = args.no_progress or base.tqdm is None
    total_rows = None
    if not disable_progress and not args.skip_row_count:
        print(f"[{study.title}] Counting rows...")
        total_rows = count_csv_rows(dataset_path)
        print(f"[{study.title}] Rows in unified dataset: {total_rows:,}")

    metric_tables: dict[str, pd.DataFrame] = {}
    word_tables: dict[str, pd.DataFrame] = {}

    if not args.skip_eval:
        print(f"[{study.title}] Computing exact accuracy and entropy...")
        fast_tables, fast_word_tables = compute_fast_metrics(
            input_csv=dataset_path,
            metrics=["exact", "entropy"],
            models=list(study.model_order),
            pipelines=list(study.pipelines),
            chunk_size=args.chunk_size,
            total_rows=total_rows,
            disable_progress=disable_progress,
            bootstrap_reps=args.bootstrap_reps,
            ci_level=args.bootstrap_ci,
            bootstrap_seed=args.bootstrap_seed,
        )
        metric_tables.update(fast_tables)
        word_tables.update(fast_word_tables)

        print(f"[{study.title}] Computing MPNet word-guess semantic similarity...")
        if encoder is None or cache is None:
            raise RuntimeError("Semantic similarity requested but embedding model/cache are unavailable.")
        metric_tables["guess_similarity"], word_tables["guess_similarity"] = compute_guess_similarity(
            input_csv=dataset_path,
            models=list(study.model_order),
            pipelines=list(study.pipelines),
            chunk_size=args.embedding_chunk_size,
            cache=cache,
            encoder=encoder,
            batch_size=args.embedding_batch_size,
            total_rows=total_rows,
            disable_progress=disable_progress,
            bootstrap_reps=args.bootstrap_reps,
            ci_level=args.bootstrap_ci,
            bootstrap_seed=args.bootstrap_seed,
        )

        overview = merge_metric_tables(metric_tables)
        write_overview_tables(overview, paths, study.pipelines)
        save_word_tables(word_tables, paths, list(study.pipelines))

        entropy_diagnostics = compute_entropy_diagnostics(
            word_tables=word_tables,
            entropy_threshold=args.entropy_diagnostic_entropy_threshold,
            accuracy_threshold=args.entropy_diagnostic_accuracy_threshold,
            similarity_threshold=args.entropy_diagnostic_similarity_threshold,
            max_examples=args.entropy_diagnostic_examples,
        )
        if not entropy_diagnostics.empty:
            save_entropy_diagnostics(entropy_diagnostics, paths, list(study.pipelines))
    else:
        overview_path = paths.tables / "overview_metrics_all_selected_pipelines.csv"
        if not overview_path.exists():
            raise FileNotFoundError(f"Cannot skip evaluation; overview not found: {overview_path}")
        overview = pd.read_csv(overview_path)

    if not args.skip_plots:
        print(f"[{study.title}] Writing combined figures...")
        save_combined_figures(
            overview=overview,
            study=study,
            output_dir=paths.root / "combined_figures",
            formats=args.formats,
        )
        print(f"[{study.title}] Figures: {paths.root / 'combined_figures'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Semantic Drift ablation analyses.")
    parser.add_argument(
        "--study",
        choices=["all", "temperature", "quantization"],
        default="all",
        help="Which ablation study to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "data_analysis" / "results_analysis",
        help="Base directory for generated ablation results.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Combined figure formats to write.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only write merged ablation datasets.")
    parser.add_argument("--skip-eval", action="store_true", help="Reuse existing metric tables and only plot.")
    parser.add_argument("--skip-plots", action="store_true", help="Compute tables but do not write combined figures.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--skip-row-count", action="store_true", help="Skip pre-counting rows for progress bars.")
    parser.add_argument("--chunk-size", type=int, default=200_000, help="Rows per chunk for exact/entropy metrics.")
    parser.add_argument(
        "--embedding-chunk-size",
        type=int,
        default=50_000,
        help="Rows per chunk for embedding-heavy metrics.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model for semantic similarity.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=128, help="Embedding batch size.")
    parser.add_argument("--bootstrap-reps", type=int, default=2000, help="Bootstrap replicates for CIs.")
    parser.add_argument("--bootstrap-ci", type=float, default=0.95, help="Bootstrap confidence level.")
    parser.add_argument("--bootstrap-seed", type=int, default=12345, help="Bootstrap seed.")
    parser.add_argument("--entropy-diagnostic-entropy-threshold", type=float, default=0.5)
    parser.add_argument("--entropy-diagnostic-accuracy-threshold", type=float, default=0.5)
    parser.add_argument("--entropy-diagnostic-similarity-threshold", type=float, default=0.5)
    parser.add_argument("--entropy-diagnostic-examples", type=int, default=3)
    return parser


def selected_studies(root: Path, study_name: str) -> list[AblationStudy]:
    studies = {
        "temperature": temperature_study(root),
        "quantization": quantization_study(root),
    }
    if study_name == "all":
        return [studies["temperature"], studies["quantization"]]
    return [studies[study_name]]


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap-reps must be 0 or greater")
    if not 0 < args.bootstrap_ci < 1:
        raise ValueError("--bootstrap-ci must be between 0 and 1")

    root = repo_root()
    studies = selected_studies(root, args.study)
    encoder = None
    cache = None

    try:
        if not args.prepare_only and not args.skip_eval:
            print(f"Loading embedding model: {args.embedding_model}")
            encoder = load_sentence_transformer(args.embedding_model)
            shared_cache_dir = args.output_dir / "ablation_studies" / "cache"
            shared_cache_dir.mkdir(parents=True, exist_ok=True)
            cache = base.EmbeddingCache(
                shared_cache_dir / f"embeddings_{slugify(args.embedding_model)}.sqlite",
                args.embedding_model,
            )

        for study in studies:
            run_study(study, args, encoder=encoder, cache=cache)
    finally:
        if cache is not None:
            cache.close()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
