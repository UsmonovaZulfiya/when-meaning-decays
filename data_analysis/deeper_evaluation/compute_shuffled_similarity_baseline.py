"""Compute calibrated final-step word-guess similarity against a shuffled baseline.

For each embedding model, generation model, and pipeline:

observed similarity = sim(target_word, actual_final_guess)
shuffled similarity = sim(target_word, final_guess_from_another_target_word)
calibrated similarity = (observed - shuffled_mean) / (1 - shuffled_mean)

The shuffled value is computed as the exact expectation over final guesses from
other target words within the same generation-model/pipeline scope, rather than
as one random draw. This makes the result deterministic and equivalent to the
mean over many shuffles.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.basic_evaluation_set.run_model_comparison_analysis import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingCache,
    MODEL_ORDER,
    PIPELINE_LABELS,
    cosine_from_cache,
    normalize_guess_series,
    normalize_word_series,
    prompt_mode_to_pipeline,
    resolve_embedding_model,
    slugify,
)


ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = ROOT.parent
FINAL_STEP = 10


@dataclass(frozen=True)
class EmbeddingRun:
    label: str
    model_name: str
    cache_path: Path


def default_cache_for_model(embedding_model: str) -> Path:
    model_name = resolve_embedding_model(embedding_model)
    if model_name == DEFAULT_EMBEDDING_MODEL:
        run_name = "all_exact_entropy_guess_similarity"
    else:
        run_name = f"all_guess_similarity_{slugify(model_name)}"
    return (
        ANALYSIS_ROOT
        / "results_analysis"
        / run_name
        / "cache"
        / f"embeddings_{slugify(model_name)}.sqlite"
    )


def default_embedding_runs() -> list[EmbeddingRun]:
    specs = [
        ("MPNet / submitted", DEFAULT_EMBEDDING_MODEL),
        ("E5-large-v2", "intfloat/e5-large-v2"),
        ("BGE-large-en-v1.5", "BAAI/bge-large-en-v1.5"),
    ]
    return [
        EmbeddingRun(label, model_name, default_cache_for_model(model_name))
        for label, model_name in specs
    ]


def load_final_step_rows(input_csv: Path, chunk_size: int) -> pd.DataFrame:
    usecols = ["model_name", "category", "original_word", "instance_id", "step", "guess", "prompt_mode"]
    chunks = []
    for chunk in pd.read_csv(input_csv, usecols=usecols, chunksize=chunk_size):
        chunk["step"] = pd.to_numeric(chunk["step"], errors="coerce")
        chunk = chunk[chunk["step"].eq(FINAL_STEP)].copy()
        if chunk.empty:
            continue
        chunk["pipeline"] = prompt_mode_to_pipeline(chunk["prompt_mode"])
        chunk["word_norm"] = normalize_word_series(chunk["original_word"])
        chunk["guess_norm"] = normalize_guess_series(chunk["guess"])
        chunk = chunk[
            chunk["word_norm"].astype(str).str.strip().ne("")
            & chunk["guess_norm"].astype(str).str.strip().ne("")
        ].copy()
        if chunk.empty:
            continue
        chunks.append(
            chunk[
                [
                    "model_name",
                    "pipeline",
                    "category",
                    "original_word",
                    "instance_id",
                    "word_norm",
                    "guess_norm",
                ]
            ]
        )
    if not chunks:
        raise ValueError(f"No final-step rows found in {input_csv}")
    return pd.concat(chunks, ignore_index=True)


def load_embeddings(cache_path: Path, embedding_model: str, texts: list[str]) -> dict[str, np.ndarray]:
    cache = EmbeddingCache(cache_path, embedding_model)
    try:
        embeddings = cache.get_many(texts)
    finally:
        cache.close()

    missing = sorted(set(texts).difference(embeddings))
    if missing:
        examples = ", ".join(repr(text) for text in missing[:5])
        raise ValueError(
            f"{cache_path} is missing {len(missing):,} embeddings for {embedding_model}. "
            f"First missing examples: {examples}"
        )
    return embeddings


def weighted_guess_similarity(word: str, guess_counts: Counter[str], embeddings: dict[str, np.ndarray]) -> float:
    total = sum(guess_counts.values())
    if total <= 0:
        return np.nan
    numerator = 0.0
    for guess, count in guess_counts.items():
        numerator += count * cosine_from_cache(word, guess, embeddings)
    return float(numerator / total)


def summarize_scope(scope: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> dict[str, object]:
    observed = [
        cosine_from_cache(word, guess, embeddings)
        for word, guess in zip(scope["word_norm"], scope["guess_norm"])
    ]
    scope = scope.copy()
    scope["observed_similarity"] = observed

    all_guess_counts = Counter(scope["guess_norm"])
    all_row_n = int(len(scope))
    shuffled_by_word: dict[str, float] = {}

    for word, word_rows in scope.groupby("word_norm", observed=True):
        own_guess_counts = Counter(word_rows["guess_norm"])
        other_guess_counts = all_guess_counts.copy()
        other_guess_counts.subtract(own_guess_counts)
        other_guess_counts = Counter({guess: count for guess, count in other_guess_counts.items() if count > 0})
        shuffled_by_word[str(word)] = weighted_guess_similarity(str(word), other_guess_counts, embeddings)

    scope["shuffled_similarity"] = scope["word_norm"].map(shuffled_by_word)
    observed_mean = float(scope["observed_similarity"].mean())
    shuffled_mean = float(scope["shuffled_similarity"].mean())
    calibrated = float((observed_mean - shuffled_mean) / (1.0 - shuffled_mean))

    return {
        "observed_similarity": observed_mean,
        "shuffled_mean": shuffled_mean,
        "calibrated_similarity": calibrated,
        "row_n": all_row_n,
        "target_word_n": int(scope["word_norm"].nunique()),
        "unique_final_guess_n": int(scope["guess_norm"].nunique()),
    }


def compute_for_embedding(final_rows: pd.DataFrame, run: EmbeddingRun) -> pd.DataFrame:
    if not run.cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found for {run.label}: {run.cache_path}")

    texts = (
        pd.concat([final_rows["word_norm"], final_rows["guess_norm"]])
        .dropna()
        .astype(str)
        .loc[lambda values: values.str.strip().ne("")]
        .unique()
        .tolist()
    )
    embeddings = load_embeddings(run.cache_path, run.model_name, texts)

    rows = []
    for (model_name, pipeline), scope in final_rows.groupby(["model_name", "pipeline"], observed=True):
        summary = summarize_scope(scope, embeddings)
        rows.append(
            {
                "embedding_model": run.label,
                "embedding_model_name": run.model_name,
                "model_name": model_name,
                "pipeline": pipeline,
                "pipeline_label": PIPELINE_LABELS.get(pipeline, pipeline),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def aggregate_all_models(per_model: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (embedding_model, embedding_model_name, pipeline, pipeline_label), scope in per_model.groupby(
        ["embedding_model", "embedding_model_name", "pipeline", "pipeline_label"],
        observed=True,
    ):
        weights = pd.to_numeric(scope["row_n"], errors="coerce")
        row_n = int(weights.sum())
        observed = float(np.average(scope["observed_similarity"], weights=weights))
        shuffled = float(np.average(scope["shuffled_mean"], weights=weights))
        rows.append(
            {
                "embedding_model": embedding_model,
                "embedding_model_name": embedding_model_name,
                "model_name": "All models",
                "pipeline": pipeline,
                "pipeline_label": pipeline_label,
                "observed_similarity": observed,
                "shuffled_mean": shuffled,
                "calibrated_similarity": float((observed - shuffled) / (1.0 - shuffled)),
                "row_n": row_n,
                "target_word_n": int(scope["target_word_n"].sum()),
                "unique_final_guess_n": np.nan,
            }
        )
    return pd.DataFrame(rows)


def sort_summary(table: pd.DataFrame) -> pd.DataFrame:
    embedding_order = {
        "MPNet / submitted": 0,
        "E5-large-v2": 1,
        "BGE-large-en-v1.5": 2,
    }
    model_order = {"All models": -1, **{model: idx for idx, model in enumerate(MODEL_ORDER)}}
    pipeline_order = {"pipeline_a": 0, "pipeline_b": 1}
    return (
        table.assign(
            _embedding_order=table["embedding_model"].map(embedding_order).fillna(999),
            _model_order=table["model_name"].map(model_order).fillna(999),
            _pipeline_order=table["pipeline"].map(pipeline_order).fillna(999),
        )
        .sort_values(["_model_order", "_embedding_order", "_pipeline_order"])
        .drop(columns=["_embedding_order", "_model_order", "_pipeline_order"])
        .reset_index(drop=True)
    )


def write_markdown(table: pd.DataFrame, output_path: Path, include_model: bool) -> None:
    columns = [
        "model_name",
        "embedding_model",
        "pipeline_label",
        "observed_similarity",
        "shuffled_mean",
        "calibrated_similarity",
    ]
    if not include_model:
        columns = [column for column in columns if column != "model_name"]

    headers = {
        "model_name": "Model",
        "embedding_model": "Embedding model",
        "pipeline_label": "Pipeline",
        "observed_similarity": "Observed final sim.",
        "shuffled_mean": "Shuffled mean",
        "calibrated_similarity": "Calibrated sim.",
    }
    lines = [
        "| " + " | ".join(headers[column] for column in columns) + " |",
        "| " + " | ".join("---:" if column.endswith("similarity") or column == "shuffled_mean" else "---" for column in columns) + " |",
    ]
    for _, row in table[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in {"observed_similarity", "shuffled_mean", "calibrated_similarity"}:
                values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute final-step calibrated semantic similarity with a shuffled baseline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "unified_semantic_drift_results.csv",
        help="Unified semantic drift results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_ROOT / "results_analysis" / "shuffled_similarity_baseline",
        help="Directory for CSV and Markdown outputs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="CSV rows per chunk while loading final-step rows.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_rows = load_final_step_rows(args.input, args.chunk_size)
    per_model_parts = []
    for run in default_embedding_runs():
        per_model_parts.append(compute_for_embedding(final_rows, run))

    per_model = sort_summary(pd.concat(per_model_parts, ignore_index=True))
    all_models = sort_summary(aggregate_all_models(per_model))
    combined = sort_summary(pd.concat([all_models, per_model], ignore_index=True))

    all_models.to_csv(args.output_dir / "shuffled_baseline_all_models.csv", index=False)
    per_model.to_csv(args.output_dir / "shuffled_baseline_by_model.csv", index=False)
    combined.to_csv(args.output_dir / "shuffled_baseline_all_and_by_model.csv", index=False)
    write_markdown(all_models, args.output_dir / "shuffled_baseline_all_models.md", include_model=False)
    write_markdown(per_model, args.output_dir / "shuffled_baseline_by_model.md", include_model=True)

    print(args.output_dir / "shuffled_baseline_all_models.md")
    print(args.output_dir / "shuffled_baseline_by_model.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
