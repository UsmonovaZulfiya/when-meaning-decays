"""Compute conservative WordNet-aware final-step recovery.

This analysis uses the unified Semantic Drift CSV and step 10 only.
It counts a final guess as recovered if it is:

1. an exact normalized match,
2. a noun-lemma match, or
3. a shared noun-synset match in WordNet.

No hypernym or hyponym matches are included.
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.basic_evaluation_set.run_model_comparison_analysis import (
    CATEGORY_ORDER,
    MODEL_ORDER,
    PIPELINE_LABELS,
    prompt_mode_to_pipeline,
)


ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = ROOT.parent
FINAL_STEP = 10
RECOVERED_TYPES = {"exact", "lemma", "wordnet_synonym"}
CATEGORY_SHORT = {
    "High-Freq-Concrete": "HFC",
    "Low-Freq-Concrete": "LFC",
    "High-Freq-Abstract": "HFA",
    "Low-Freq-Abstract": "LFA",
}

wn: Any = None


def normalize_text(value: object) -> str:
    """Strict normalization with common answer-prefix cleanup."""
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = text.splitlines()[0] if text else ""
    text = re.sub(r"^(the word is|answer|guess|my guess is)\s*[:\-]?\s*", "", text)
    text = re.sub(r"[^a-z\s\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_wordnet_form(value: object) -> str:
    return normalize_text(value).replace(" ", "_").replace("-", "_")


def initialize_wordnet(download: bool) -> None:
    """Load WordNet and optionally download its corpus data."""
    global wn

    try:
        import nltk
        from nltk.corpus import wordnet as wordnet
    except ImportError as exc:
        raise RuntimeError(
            "This analysis requires nltk. Install it with: "
            "..\\.venv\\Scripts\\python.exe -m pip install nltk"
        ) from exc

    if download:
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    try:
        wordnet.synsets("dog", pos=wordnet.NOUN)
    except LookupError as exc:
        raise RuntimeError(
            "NLTK WordNet data is missing. Run either:\n"
            "  ..\\.venv\\Scripts\\python.exe -m nltk.downloader wordnet omw-1.4\n"
            "or rerun this script with --download-wordnet."
        ) from exc

    wn = wordnet
    noun_lemma_forms.cache_clear()
    noun_synset_names.cache_clear()


@lru_cache(maxsize=None)
def noun_lemma_forms(value: str) -> frozenset[str]:
    wordnet_value = to_wordnet_form(value)
    forms = {wordnet_value} if wordnet_value else set()
    lemma = wn.morphy(wordnet_value, wn.NOUN) if wordnet_value else None
    if lemma:
        forms.add(lemma)
    return frozenset(forms)


@lru_cache(maxsize=None)
def noun_synset_names(value: str) -> frozenset[str]:
    wordnet_value = to_wordnet_form(value)
    if not wordnet_value:
        return frozenset()
    return frozenset(synset.name() for synset in wn.synsets(wordnet_value, pos=wn.NOUN))


def recovery_type(target: object, guess: object) -> str:
    target_norm = normalize_text(target)
    guess_norm = normalize_text(guess)

    if not target_norm or not guess_norm:
        return "empty_or_invalid"

    if target_norm == guess_norm:
        return "exact"

    if noun_lemma_forms(target_norm) & noun_lemma_forms(guess_norm):
        return "lemma"

    target_synsets = noun_synset_names(target_norm)
    guess_synsets = noun_synset_names(guess_norm)
    if target_synsets and guess_synsets and (target_synsets & guess_synsets):
        return "wordnet_synonym"

    return "not_recovered"


def load_final_step(input_csv: Path, chunk_size: int) -> pd.DataFrame:
    usecols = ["model_name", "category", "original_word", "instance_id", "step", "guess", "prompt_mode"]
    chunks = []
    for chunk in pd.read_csv(input_csv, usecols=usecols, chunksize=chunk_size):
        chunk["step"] = pd.to_numeric(chunk["step"], errors="coerce")
        chunk = chunk[chunk["step"].eq(FINAL_STEP)].copy()
        if chunk.empty:
            continue

        chunk["pipeline"] = prompt_mode_to_pipeline(chunk["prompt_mode"])
        chunk["target_norm"] = chunk["original_word"].map(normalize_text)
        chunk["guess_norm"] = chunk["guess"].map(normalize_text)
        chunk["exact_recovered"] = chunk["target_norm"].eq(chunk["guess_norm"]) & chunk["target_norm"].ne("")
        chunks.append(
            chunk[
                [
                    "model_name",
                    "pipeline",
                    "category",
                    "original_word",
                    "instance_id",
                    "target_norm",
                    "guess_norm",
                    "exact_recovered",
                ]
            ]
        )

    if not chunks:
        raise ValueError(f"No final-step rows found in {input_csv}")
    return pd.concat(chunks, ignore_index=True)


def add_wordnet_recovery(final_step: pd.DataFrame) -> pd.DataFrame:
    final_step = final_step.copy()
    final_step["wn_recovery_type"] = [
        recovery_type(target, guess)
        for target, guess in zip(final_step["original_word"], final_step["guess_norm"])
    ]
    final_step["wordnet_recovered"] = final_step["wn_recovery_type"].isin(RECOVERED_TYPES)
    return final_step


def compute_word_level(final_step: pd.DataFrame) -> pd.DataFrame:
    return (
        final_step.groupby(
            ["model_name", "pipeline", "category", "original_word", "target_norm"],
            as_index=False,
            observed=True,
        )
        .agg(
            exact_acc=("exact_recovered", "mean"),
            wordnet_acc=("wordnet_recovered", "mean"),
            chain_n=("wordnet_recovered", "size"),
        )
    )


def add_delta_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["delta"] = table["wordnet_acc"] - table["exact_acc"]
    table["relative_error_reclassified"] = table["delta"] / (1.0 - table["exact_acc"])
    return table


def summarize_pipeline(word_level: pd.DataFrame) -> pd.DataFrame:
    table = (
        word_level.groupby("pipeline", as_index=False, observed=True)
        .agg(
            exact_acc=("exact_acc", "mean"),
            wordnet_acc=("wordnet_acc", "mean"),
            target_word_n=("target_norm", "nunique"),
            model_word_n=("wordnet_acc", "size"),
            chain_n=("chain_n", "sum"),
        )
    )
    table["pipeline_label"] = table["pipeline"].map(PIPELINE_LABELS).fillna(table["pipeline"])
    return add_delta_columns(table)


def summarize_category(word_level: pd.DataFrame) -> pd.DataFrame:
    table = (
        word_level.groupby(["pipeline", "category"], as_index=False, observed=True)
        .agg(
            exact_acc=("exact_acc", "mean"),
            wordnet_acc=("wordnet_acc", "mean"),
            target_word_n=("target_norm", "nunique"),
            model_word_n=("wordnet_acc", "size"),
            chain_n=("chain_n", "sum"),
        )
    )
    table["pipeline_label"] = table["pipeline"].map(PIPELINE_LABELS).fillna(table["pipeline"])
    return add_delta_columns(table)


def summarize_by_model(word_level: pd.DataFrame) -> pd.DataFrame:
    table = (
        word_level.groupby(["model_name", "pipeline"], as_index=False, observed=True)
        .agg(
            exact_acc=("exact_acc", "mean"),
            wordnet_acc=("wordnet_acc", "mean"),
            target_word_n=("target_norm", "nunique"),
            model_word_n=("wordnet_acc", "size"),
            chain_n=("chain_n", "sum"),
        )
    )
    table["pipeline_label"] = table["pipeline"].map(PIPELINE_LABELS).fillna(table["pipeline"])
    return add_delta_columns(table)


def summarize_type_breakdown(final_step: pd.DataFrame) -> pd.DataFrame:
    table = (
        final_step.groupby(["pipeline", "wn_recovery_type"], as_index=False, observed=True)
        .size()
        .rename(columns={"size": "count"})
    )
    table["pipeline_label"] = table["pipeline"].map(PIPELINE_LABELS).fillna(table["pipeline"])
    table["proportion"] = table["count"] / table.groupby("pipeline")["count"].transform("sum")
    return table


def lowest_category(category_summary: pd.DataFrame, metric_col: str) -> str:
    category_means = category_summary.groupby("category", observed=True)[metric_col].mean()
    if category_means.empty:
        return ""
    category = str(category_means.idxmin())
    return CATEGORY_SHORT.get(category, category)


def build_main_table(pipeline_summary: pd.DataFrame, category_summary: pd.DataFrame) -> pd.DataFrame:
    by_pipeline = pipeline_summary.set_index("pipeline")
    rows = []
    for label, metric_col in [
        ("Exact accuracy", "exact_acc"),
        ("WordNet-aware recovery", "wordnet_acc"),
    ]:
        pipeline_a = float(by_pipeline.loc["pipeline_a", metric_col])
        pipeline_b = float(by_pipeline.loc["pipeline_b", metric_col])
        rows.append(
            {
                "Metric": label,
                "Pipeline A": pipeline_a,
                "Pipeline B": pipeline_b,
                "A-B gap": pipeline_a - pipeline_b,
                "Lowest category": lowest_category(category_summary, metric_col),
            }
        )
    return pd.DataFrame(rows)


def sort_outputs(
    pipeline_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    by_model: pd.DataFrame,
    type_breakdown: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pipeline_order = {"pipeline_a": 0, "pipeline_b": 1}
    category_order = {category: idx for idx, category in enumerate(CATEGORY_ORDER)}
    model_order = {model: idx for idx, model in enumerate(MODEL_ORDER)}

    pipeline_summary = (
        pipeline_summary.assign(_pipeline_order=pipeline_summary["pipeline"].map(pipeline_order).fillna(99))
        .sort_values("_pipeline_order")
        .drop(columns="_pipeline_order")
    )
    category_summary = (
        category_summary.assign(
            _pipeline_order=category_summary["pipeline"].map(pipeline_order).fillna(99),
            _category_order=category_summary["category"].map(category_order).fillna(99),
        )
        .sort_values(["_pipeline_order", "_category_order"])
        .drop(columns=["_pipeline_order", "_category_order"])
    )
    by_model = (
        by_model.assign(
            _model_order=by_model["model_name"].map(model_order).fillna(99),
            _pipeline_order=by_model["pipeline"].map(pipeline_order).fillna(99),
        )
        .sort_values(["_model_order", "_pipeline_order"])
        .drop(columns=["_model_order", "_pipeline_order"])
    )
    type_breakdown = (
        type_breakdown.assign(_pipeline_order=type_breakdown["pipeline"].map(pipeline_order).fillna(99))
        .sort_values(["_pipeline_order", "wn_recovery_type"])
        .drop(columns="_pipeline_order")
    )
    return pipeline_summary, category_summary, by_model, type_breakdown


def write_markdown(table: pd.DataFrame, output_path: Path) -> None:
    columns = table.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute conservative WordNet-aware recovery at the final chain step."
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
        default=ANALYSIS_ROOT / "results_analysis" / "wordnet_recovery_final_step",
        help="Directory for generated tables.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="CSV rows per chunk while loading final-step rows.",
    )
    parser.add_argument(
        "--download-wordnet",
        action="store_true",
        help="Download nltk WordNet/OMW corpora before running.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    initialize_wordnet(download=args.download_wordnet)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_step = add_wordnet_recovery(load_final_step(args.input, args.chunk_size))
    word_level = compute_word_level(final_step)
    pipeline_summary = summarize_pipeline(word_level)
    category_summary = summarize_category(word_level)
    by_model = summarize_by_model(word_level)
    type_breakdown = summarize_type_breakdown(final_step)
    pipeline_summary, category_summary, by_model, type_breakdown = sort_outputs(
        pipeline_summary,
        category_summary,
        by_model,
        type_breakdown,
    )
    main_table = build_main_table(pipeline_summary, category_summary)

    final_step.to_csv(args.output_dir / "wordnet_recovery_final_step_rows.csv", index=False)
    word_level.to_csv(args.output_dir / "wordnet_recovery_word_level.csv", index=False)
    pipeline_summary.to_csv(args.output_dir / "wordnet_recovery_pipeline_summary.csv", index=False)
    category_summary.to_csv(args.output_dir / "wordnet_recovery_category_summary.csv", index=False)
    by_model.to_csv(args.output_dir / "wordnet_recovery_by_model.csv", index=False)
    type_breakdown.to_csv(args.output_dir / "wordnet_recovery_type_breakdown.csv", index=False)
    main_table.to_csv(args.output_dir / "wordnet_recovery_main_table.csv", index=False)
    write_markdown(main_table, args.output_dir / "wordnet_recovery_main_table.md")

    print(args.output_dir / "wordnet_recovery_main_table.md")
    print(args.output_dir / "wordnet_recovery_pipeline_summary.csv")
    print(args.output_dir / "wordnet_recovery_category_summary.csv")
    print(args.output_dir / "wordnet_recovery_type_breakdown.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
