"""Generate compact word-guess similarity robustness tables.

The tables compare the submitted MPNet semantic-similarity run with reruns
from E5-large-v2 and BGE-large-en-v1.5. Outputs are CSV and Markdown files
under data_analysis/results_analysis/embedding_model_robustness/tables by default.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from .run_model_comparison_analysis import MODEL_ORDER, MODEL_SLUGS
except ImportError:  # Direct execution from this directory is supported.
    from run_model_comparison_analysis import MODEL_ORDER, MODEL_SLUGS


ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = ROOT.parent
METRIC_COL = "word_guess_semantic_similarity"
WEIGHT_COL = "word_guess_similarity_n"
FINAL_CATEGORY = "All Categories"

CATEGORY_SHORT = {
    "High-Freq-Concrete": "HFC",
    "Low-Freq-Concrete": "LFC",
    "High-Freq-Abstract": "HFA",
    "Low-Freq-Abstract": "LFA",
}


@dataclass(frozen=True)
class EmbeddingRun:
    label: str
    overview_csv: Path


def default_runs() -> list[EmbeddingRun]:
    return [
        EmbeddingRun(
            "MPNet / submitted",
            ANALYSIS_ROOT
            / "results_analysis"
            / "all_exact_entropy_guess_similarity"
            / "tables"
            / "overview_metrics_all_selected_pipelines.csv",
        ),
        EmbeddingRun(
            "E5-large-v2",
            ANALYSIS_ROOT
            / "results_analysis"
            / "all_guess_similarity_intfloat_e5_large_v2"
            / "tables"
            / "overview_metrics_all_selected_pipelines.csv",
        ),
        EmbeddingRun(
            "BGE-large-en-v1.5",
            ANALYSIS_ROOT
            / "results_analysis"
            / "all_guess_similarity_baai_bge_large_en_v1_5"
            / "tables"
            / "overview_metrics_all_selected_pipelines.csv",
        ),
    ]


def read_overview(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing overview CSV: {path}")

    overview = pd.read_csv(path)
    required = {"model_name", "pipeline", "category", "step", METRIC_COL}
    missing = required.difference(overview.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    overview["step"] = pd.to_numeric(overview["step"], errors="raise").astype(int)
    overview[METRIC_COL] = pd.to_numeric(overview[METRIC_COL], errors="coerce")
    if WEIGHT_COL in overview.columns:
        overview[WEIGHT_COL] = pd.to_numeric(overview[WEIGHT_COL], errors="coerce")
    return overview


def weighted_mean(rows: pd.DataFrame, value_col: str) -> float:
    values = pd.to_numeric(rows[value_col], errors="coerce")
    valid = rows.loc[values.notna()].copy()
    if valid.empty:
        return math.nan

    if WEIGHT_COL in valid.columns:
        weights = pd.to_numeric(valid[WEIGHT_COL], errors="coerce").fillna(0.0)
        if float(weights.sum()) > 0:
            return float((valid[value_col] * weights).sum() / weights.sum())

    return float(valid[value_col].mean())


def final_rows_for_scope(overview: pd.DataFrame, model_name: str | None) -> pd.DataFrame:
    scoped = overview.copy()
    if model_name is not None:
        scoped = scoped[scoped["model_name"].eq(model_name)]
    if scoped.empty:
        raise ValueError(f"No rows found for model scope: {model_name or 'all models'}")
    final_step = int(scoped["step"].max())
    return scoped[scoped["step"].eq(final_step)].copy()


def pipeline_final_similarity(final_rows: pd.DataFrame, pipeline: str) -> float:
    rows = final_rows[
        final_rows["pipeline"].eq(pipeline)
        & final_rows["category"].eq(FINAL_CATEGORY)
    ]
    return weighted_mean(rows, METRIC_COL)


def lowest_final_category(final_rows: pd.DataFrame) -> str:
    category_rows = final_rows[
        final_rows["category"].ne(FINAL_CATEGORY)
        & final_rows["pipeline"].isin(["pipeline_a", "pipeline_b"])
    ]
    if category_rows.empty:
        return ""

    category_scores = pd.Series(
        {
            category: weighted_mean(rows, METRIC_COL)
            for category, rows in category_rows.groupby("category", observed=True)
        }
    ).dropna()
    if category_scores.empty:
        return ""
    category = str(category_scores.idxmin())
    return CATEGORY_SHORT.get(category, category)


def conclusion_from_gap(gap: float) -> str:
    if math.isnan(gap):
        return ""
    return "preserved" if gap > 0 else "not preserved"


def build_table(runs: list[EmbeddingRun], model_name: str | None) -> pd.DataFrame:
    rows = []
    for run in runs:
        overview = read_overview(run.overview_csv)
        final_rows = final_rows_for_scope(overview, model_name)
        pipeline_a = pipeline_final_similarity(final_rows, "pipeline_a")
        pipeline_b = pipeline_final_similarity(final_rows, "pipeline_b")
        gap = pipeline_a - pipeline_b
        rows.append(
            {
                "Embedding model": run.label,
                "Pipeline A final sim.": pipeline_a,
                "Pipeline B final sim.": pipeline_b,
                "Gap A-B": gap,
                "Lowest category": lowest_final_category(final_rows),
                "Main conclusion": conclusion_from_gap(gap),
            }
        )
    return pd.DataFrame(rows)


def format_markdown_value(column: str, value: object) -> str:
    if column in {"Pipeline A final sim.", "Pipeline B final sim.", "Gap A-B"}:
        return f"{float(value):.3f}" if pd.notna(value) else ""
    return str(value)


def write_markdown(table: pd.DataFrame, output_path: Path, title: str) -> None:
    columns = table.columns.tolist()
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| "
            + " | ".join(format_markdown_value(column, row[column]) for column in columns)
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(table: pd.DataFrame, output_dir: Path, slug: str, title: str) -> None:
    csv_table = table.copy()
    for column in ["Pipeline A final sim.", "Pipeline B final sim.", "Gap A-B"]:
        csv_table[column] = csv_table[column].map(lambda value: f"{float(value):.6f}" if pd.notna(value) else "")
    csv_table.to_csv(output_dir / f"{slug}.csv", index=False)
    write_markdown(table, output_dir / f"{slug}.md", title)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build compact embedding-model robustness tables for word-guess semantic similarity."
    )
    parser.add_argument(
        "--mpnet-overview",
        type=Path,
        default=default_runs()[0].overview_csv,
        help="Submitted MPNet overview_metrics_all_selected_pipelines.csv.",
    )
    parser.add_argument(
        "--e5-overview",
        type=Path,
        default=default_runs()[1].overview_csv,
        help="E5-large-v2 overview_metrics_all_selected_pipelines.csv.",
    )
    parser.add_argument(
        "--bge-overview",
        type=Path,
        default=default_runs()[2].overview_csv,
        help="BGE-large-en-v1.5 overview_metrics_all_selected_pipelines.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_ROOT / "results_analysis" / "embedding_model_robustness" / "tables",
        help="Directory for generated robustness CSV/Markdown tables.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    runs = [
        EmbeddingRun("MPNet / submitted", args.mpnet_overview.resolve()),
        EmbeddingRun("E5-large-v2", args.e5_overview.resolve()),
        EmbeddingRun("BGE-large-en-v1.5", args.bge_overview.resolve()),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scopes: list[tuple[str, str, str | None]] = [
        ("All models", "embedding_robustness_all_models", None),
    ]
    scopes.extend((model, f"embedding_robustness_{MODEL_SLUGS[model]}", model) for model in MODEL_ORDER)

    for title, filename_slug, model_name in scopes:
        table = build_table(runs, model_name)
        write_outputs(table, args.output_dir, filename_slug, title)
        print(args.output_dir / f"{filename_slug}.md")

    combined = []
    for title, _, model_name in scopes:
        table = build_table(runs, model_name)
        table.insert(0, "Scope", title)
        combined.append(table)
    pd.concat(combined, ignore_index=True).to_csv(
        args.output_dir / "embedding_robustness_all_tables.csv",
        index=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
