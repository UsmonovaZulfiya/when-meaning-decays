"""Extract the highest word-guess similarity chain per model and pipeline.

By default this uses the submitted MPNet embedding cache and ranks each
instance-level chain by its mean original-word-to-guess similarity across
steps 1-10. Step 0 is still included in the exported chain for context.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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


def chain_sort_key(row: dict[str, object]) -> tuple:
    model_order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    pipeline_order = {"pipeline_a": 0, "pipeline_b": 1}
    return (
        model_order.get(str(row["model_name"]), 999),
        pipeline_order.get(str(row["pipeline"]), 999),
    )


def extract_highest_chains(
    input_csv: Path,
    cache_path: Path,
    embedding_model: str,
    chunk_size: int,
    rank_start_step: int,
    rank_end_step: int,
) -> pd.DataFrame:
    cache = EmbeddingCache(cache_path, embedding_model)
    usecols = ["model_name", "category", "original_word", "instance_id", "step", "guess", "prompt_mode"]

    chain_sums: defaultdict[tuple, float] = defaultdict(float)
    chain_counts: defaultdict[tuple, int] = defaultdict(int)
    chain_step_guesses: defaultdict[tuple, dict[int, str]] = defaultdict(dict)
    chain_step_sims: defaultdict[tuple, dict[int, float]] = defaultdict(dict)
    chain_word_norm: dict[tuple, str] = {}

    try:
        for chunk in pd.read_csv(input_csv, usecols=usecols, chunksize=chunk_size):
            chunk["step"] = pd.to_numeric(chunk["step"], errors="coerce")
            chunk = chunk[chunk["step"].between(0, rank_end_step, inclusive="both")].copy()
            if chunk.empty:
                continue

            chunk["step"] = chunk["step"].astype(int)
            chunk["pipeline"] = prompt_mode_to_pipeline(chunk["prompt_mode"])
            chunk["word_norm"] = normalize_word_series(chunk["original_word"])
            chunk["guess_norm"] = normalize_guess_series(chunk["guess"])

            texts = (
                pd.concat([chunk["word_norm"], chunk["guess_norm"]])
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            embeddings = cache.get_many(texts)

            for row in chunk.itertuples(index=False):
                key = (
                    row.model_name,
                    row.pipeline,
                    row.category,
                    row.original_word,
                    str(row.instance_id),
                )
                step = int(row.step)
                word = str(row.word_norm)
                guess = str(row.guess_norm)
                similarity = cosine_from_cache(word, guess, embeddings)

                chain_word_norm[key] = word
                chain_step_guesses[key][step] = guess
                if not np.isnan(similarity):
                    chain_step_sims[key][step] = float(similarity)
                    if rank_start_step <= step <= rank_end_step:
                        chain_sums[key] += float(similarity)
                        chain_counts[key] += 1
    finally:
        cache.close()

    best_by_model_pipeline: dict[tuple[str, str], dict[str, object]] = {}
    for key, count in chain_counts.items():
        if count <= 0:
            continue
        model_name, pipeline, category, original_word, instance_id = key
        score = chain_sums[key] / count
        step_sims = chain_step_sims[key]
        final_similarity = step_sims.get(rank_end_step, np.nan)
        candidate = {
            "model_name": model_name,
            "pipeline": pipeline,
            "category": category,
            "original_word": original_word,
            "original_word_norm": chain_word_norm.get(key, ""),
            "instance_id": instance_id,
            f"mean_similarity_steps_{rank_start_step}_{rank_end_step}": score,
            f"valid_similarity_steps_{rank_start_step}_{rank_end_step}": count,
            f"step{rank_end_step}_similarity": final_similarity,
        }
        for step in range(0, rank_end_step + 1):
            candidate[f"step{step}_guess"] = chain_step_guesses[key].get(step, "")
            candidate[f"step{step}_similarity"] = step_sims.get(step, np.nan)

        group_key = (str(model_name), str(pipeline))
        current = best_by_model_pipeline.get(group_key)
        if current is None:
            best_by_model_pipeline[group_key] = candidate
            continue

        current_score = float(current[f"mean_similarity_steps_{rank_start_step}_{rank_end_step}"])
        current_final = float(current.get(f"step{rank_end_step}_similarity", np.nan))
        if (
            score > current_score
            or (np.isclose(score, current_score) and final_similarity > current_final)
            or (
                np.isclose(score, current_score)
                and np.isclose(final_similarity, current_final, equal_nan=True)
                and str((category, original_word, instance_id)) < str(
                    (current["category"], current["original_word"], current["instance_id"])
                )
            )
        ):
            best_by_model_pipeline[group_key] = candidate

    rows = sorted(best_by_model_pipeline.values(), key=chain_sort_key)
    table = pd.DataFrame(rows)
    table.insert(1, "pipeline_label", table["pipeline"].map(PIPELINE_LABELS).fillna(table["pipeline"]))
    step_cols = [f"step{step}_guess" for step in range(0, rank_end_step + 1)]
    table["chain"] = table[step_cols].fillna("").agg(" -> ".join, axis=1)
    return table


def write_markdown(table: pd.DataFrame, output_path: Path, rank_start_step: int, rank_end_step: int) -> None:
    score_col = f"mean_similarity_steps_{rank_start_step}_{rank_end_step}"
    final_col = f"step{rank_end_step}_similarity"
    columns = [
        "model_name",
        "pipeline_label",
        "category",
        "original_word",
        "instance_id",
        score_col,
        final_col,
        "chain",
    ]

    lines = [
        f"| Model | Pipeline | Category | Original word | Instance | Mean sim. steps {rank_start_step}-{rank_end_step} | Step {rank_end_step} sim. | Chain |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in table[columns].itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.model_name),
                    str(row.pipeline_label),
                    str(row.category),
                    str(row.original_word),
                    str(row.instance_id),
                    f"{float(getattr(row, score_col)):.6f}",
                    f"{float(getattr(row, final_col)):.6f}",
                    str(row.chain),
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the highest semantic-similarity chain per generation model and pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "unified_semantic_drift_results.csv",
        help="Unified semantic drift results CSV.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name or alias used for the cache lookup.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        help="SQLite embedding cache. Defaults to the expected cache for --embedding-model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_ROOT / "results_analysis" / "highest_similarity_chains",
        help="Directory for the extracted chain tables.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="CSV rows per processing chunk.",
    )
    parser.add_argument(
        "--rank-start-step",
        type=int,
        default=1,
        help="First step included in the chain-level mean similarity score.",
    )
    parser.add_argument(
        "--rank-end-step",
        type=int,
        default=10,
        help="Last step included in the chain-level mean similarity score and exported chain.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    embedding_model = resolve_embedding_model(args.embedding_model)
    cache_path = args.embedding_cache or default_cache_for_model(embedding_model)
    if not cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = extract_highest_chains(
        input_csv=args.input,
        cache_path=cache_path,
        embedding_model=embedding_model,
        chunk_size=args.chunk_size,
        rank_start_step=args.rank_start_step,
        rank_end_step=args.rank_end_step,
    )

    output_slug = f"highest_similarity_chains_{slugify(embedding_model)}"
    csv_path = args.output_dir / f"{output_slug}.csv"
    md_path = args.output_dir / f"{output_slug}.md"
    table.to_csv(csv_path, index=False)
    write_markdown(table, md_path, args.rank_start_step, args.rank_end_step)
    print(md_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
