"""Compare Semantic Drift metrics across models.

Default input:
    datasets/unified_semantic_drift_results.csv

Examples:
    python data_analysis/basic_evaluation_set/run_model_comparison_analysis.py --metrics exact entropy --model-group all --include-human-study
    python data_analysis/basic_evaluation_set/run_model_comparison_analysis.py --metrics exact entropy --model-group llama
    python data_analysis/basic_evaluation_set/run_model_comparison_analysis.py --metrics guess_similarity --embedding-model e5-large-v2 --skip-plots
    python data_analysis/basic_evaluation_set/run_model_comparison_analysis.py --metrics description-sim bleu --model-group gemma
    python data_analysis/basic_evaluation_set/run_model_comparison_analysis.py --metrics all --models "Llama 3.1 8B Instruct" "Llama 3.1 70B Instruct"

Outputs are written under data_analysis/results_analysis/<comparison_name>/.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None


MODEL_ORDER = [
    "Llama 3.1 8B Instruct",
    "Llama 3.1 70B Instruct",
    "Gemma 3 4B IT",
    "Gemma 3 12B IT",
    "Gemma 3 27B IT",
]

MODEL_GROUPS = {
    "all": MODEL_ORDER,
    "llama": ["Llama 3.1 8B Instruct", "Llama 3.1 70B Instruct"],
    "gemma": ["Gemma 3 4B IT", "Gemma 3 12B IT", "Gemma 3 27B IT"],
}

MODEL_SLUGS = {
    "Llama 3.1 8B Instruct": "llama_3_1_8b_instruct",
    "Llama 3.1 70B Instruct": "llama_3_1_70b_instruct",
    "Gemma 3 4B IT": "gemma_3_4b_it",
    "Gemma 3 12B IT": "gemma_3_12b_it",
    "Gemma 3 27B IT": "gemma_3_27b_it",
}

PIPELINE_LABELS = {
    "pipeline_a": "Pipeline A",
    "pipeline_b": "Pipeline B",
}

PIPELINE_FROM_PROMPT_MODE = {
    "pipeline_A_guess_then_describe_guessed_word": "pipeline_a",
    "pipeline_B_paraphrase_description": "pipeline_b",
}

METRIC_ALIASES = {
    "exact": "exact",
    "accuracy": "exact",
    "exact-accuracy": "exact",
    "entropy": "entropy",
    "shannon": "entropy",
    "guess-sim": "guess_similarity",
    "guess_similarity": "guess_similarity",
    "word-guess-sim": "guess_similarity",
    "word_guess_similarity": "guess_similarity",
    "description-sim": "description_similarity",
    "description_similarity": "description_similarity",
    "desc-sim": "description_similarity",
    "desc_sim": "description_similarity",
    "bleu": "description_bleu",
    "description-bleu": "description_bleu",
    "description_bleu": "description_bleu",
}

METRIC_COLUMNS = {
    "exact": "exact_accuracy",
    "guess_similarity": "word_guess_semantic_similarity",
    "description_similarity": "description_semantic_similarity_to_step0",
    "description_bleu": "description_bleu_to_step0",
    "entropy": "shannon_entropy",
    "normalized_entropy": "normalized_entropy",
}

METRIC_TITLES = {
    "exact_accuracy": "Exact Accuracy",
    "word_guess_semantic_similarity": "Semantic Similarity: Original Word vs Guess",
    "description_semantic_similarity_to_step0": "Description Similarity to Step 0",
    "description_bleu_to_step0": "Description BLEU to Step 0",
    "shannon_entropy": "Shannon Entropy of Guesses",
    "normalized_entropy": "Normalized Shannon Entropy of Guesses",
}

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

EMBEDDING_MODEL_ALIASES = {
    "all-mpnet-base-v2": DEFAULT_EMBEDDING_MODEL,
    "mpnet": DEFAULT_EMBEDDING_MODEL,
    "mpnet-submitted": DEFAULT_EMBEDDING_MODEL,
    "submitted": DEFAULT_EMBEDDING_MODEL,
    "sentence-transformers/all-mpnet-base-v2": DEFAULT_EMBEDDING_MODEL,
    "e5-large-v2": "intfloat/e5-large-v2",
    "intfloat/e5-large-v2": "intfloat/e5-large-v2",
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "bge": "BAAI/bge-large-en-v1.5",
    "baai/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
}

EMBEDDING_METRICS = {"guess_similarity", "description_similarity"}

CATEGORY_ORDER = [
    "High-Freq-Concrete",
    "Low-Freq-Concrete",
    "High-Freq-Abstract",
    "Low-Freq-Abstract",
]

CATEGORY_OUTPUT_ORDER = ["All Categories", *CATEGORY_ORDER]

MODEL_COLORS = {
    "Llama 3.1 8B Instruct": "#1f77b4",
    "Llama 3.1 70B Instruct": "#ff7f0e",
    "Gemma 3 4B IT": "#2ca02c",
    "Gemma 3 12B IT": "#d62728",
    "Gemma 3 27B IT": "#9467bd",
}


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    tables: Path
    figures: Path
    cache: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def resolve_embedding_model(value: str) -> str:
    normalized = value.strip()
    return EMBEDDING_MODEL_ALIASES.get(normalized.casefold(), normalized)


def parse_metrics(values: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw in values:
        for item in raw.split(","):
            item = item.strip().lower()
            if not item:
                continue
            if item == "all":
                expanded.extend(
                    [
                        "exact",
                        "entropy",
                        "guess_similarity",
                        "description_similarity",
                        "description_bleu",
                    ]
                )
            elif item == "quick":
                expanded.extend(["exact", "entropy"])
            elif item == "semantic":
                expanded.extend(["guess_similarity", "description_similarity"])
            elif item in METRIC_ALIASES:
                expanded.append(METRIC_ALIASES[item])
            else:
                allowed = ", ".join(sorted(set(METRIC_ALIASES) | {"all", "quick", "semantic"}))
                raise ValueError(f"Unknown metric {item!r}. Allowed values: {allowed}")

    seen: set[str] = set()
    unique = []
    for metric in expanded:
        if metric not in seen:
            unique.append(metric)
            seen.add(metric)
    return unique


def select_models(model_group: str, explicit_models: list[str] | None) -> list[str]:
    if explicit_models:
        requested = []
        model_lookup = {m.casefold(): m for m in MODEL_ORDER}
        slug_lookup = {MODEL_SLUGS[m]: m for m in MODEL_ORDER}
        for value in explicit_models:
            key = value.casefold()
            slug = slugify(value)
            if key in model_lookup:
                requested.append(model_lookup[key])
            elif slug in slug_lookup:
                requested.append(slug_lookup[slug])
            else:
                known = ", ".join(MODEL_ORDER)
                raise ValueError(f"Unknown model {value!r}. Known models: {known}")
        return [m for m in MODEL_ORDER if m in requested]

    return MODEL_GROUPS[model_group]


def build_output_paths(base_dir: Path, comparison_name: str) -> OutputPaths:
    root = base_dir / comparison_name
    paths = OutputPaths(
        root=root,
        tables=root / "tables",
        figures=root / "figures",
        cache=root / "cache",
    )
    for path in [paths.tables, paths.figures, paths.cache]:
        path.mkdir(parents=True, exist_ok=True)
    for pipeline in PIPELINE_LABELS:
        (paths.tables / pipeline).mkdir(parents=True, exist_ok=True)
        (paths.figures / pipeline).mkdir(parents=True, exist_ok=True)
    return paths


@contextmanager
def progress_bar(*args, **kwargs):
    if tqdm is None or kwargs.pop("disable", False):
        yield None
    else:
        with tqdm(*args, **kwargs) as bar:
            yield bar


def count_csv_rows(path: Path) -> int:
    """Count data rows for progress bars without loading the CSV into memory."""
    total_lines = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total_lines += block.count(b"\n")
    return max(total_lines - 1, 0)


def normalize_word_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-zA-Z\s\-]", "", regex=True)
        .str.strip()
    )


def normalize_guess_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    normalized = normalized.str.replace(r"^the word is\s+", "", regex=True)
    normalized = normalized.str.replace(r"^answer:\s*", "", regex=True)
    normalized = normalized.str.replace(r"^guess:\s*", "", regex=True)
    normalized = normalized.str.replace(r"[^a-zA-Z\s\-]", "", regex=True)
    return normalized.str.strip()


def prompt_mode_to_pipeline(series: pd.Series) -> pd.Series:
    return series.map(PIPELINE_FROM_PROMPT_MODE).fillna(series.astype(str).str.lower())


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in (base_seed, *parts))
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % (2**32)


def summarize_clustered_mean(
    values: pd.Series,
    bootstrap_reps: int,
    ci_level: float,
    seed: int,
) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(numeric.size)
    if n == 0:
        return {"mean": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    mean = float(np.mean(numeric))
    se = float(np.std(numeric, ddof=1) / math.sqrt(n)) if n > 1 else np.nan
    if bootstrap_reps <= 0 or n == 1:
        return {"mean": mean, "se": se, "ci_low": mean, "ci_high": mean}

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(bootstrap_reps, n))
    boot_means = numeric[indices].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    return {
        "mean": mean,
        "se": se,
        "ci_low": float(np.quantile(boot_means, alpha)),
        "ci_high": float(np.quantile(boot_means, 1.0 - alpha)),
    }


def add_all_categories(word_df: pd.DataFrame) -> pd.DataFrame:
    if word_df.empty:
        return word_df
    all_rows = word_df.copy()
    all_rows["category"] = "All Categories"
    return pd.concat([word_df, all_rows], ignore_index=True)


def aggregate_word_metric_table(
    word_df: pd.DataFrame,
    metric_col: str,
    word_count_col: str,
    row_count_col: str,
    row_count_source_col: str,
    bootstrap_reps: int,
    ci_level: float,
    seed: int,
) -> pd.DataFrame:
    """Summarize target-word means with uncertainty clustered by target word."""
    keys = ["model_name", "pipeline", "category", "step"]
    rows = []
    for key, sub in add_all_categories(word_df).groupby(keys, observed=True):
        values = pd.to_numeric(sub[metric_col], errors="coerce")
        valid = values.notna()
        if not valid.any():
            continue
        stats = summarize_clustered_mean(
            values.loc[valid],
            bootstrap_reps=bootstrap_reps,
            ci_level=ci_level,
            seed=stable_seed(seed, metric_col, *key),
        )
        rows.append(
            {
                "model_name": key[0],
                "pipeline": key[1],
                "category": key[2],
                "step": key[3],
                metric_col: stats["mean"],
                f"{metric_col}_se": stats["se"],
                f"{metric_col}_ci_low": stats["ci_low"],
                f"{metric_col}_ci_high": stats["ci_high"],
                f"{metric_col}_ci_level": ci_level,
                word_count_col: int(valid.sum()),
                row_count_col: int(pd.to_numeric(sub.loc[valid, row_count_source_col], errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def add_group_sums(
    summary: pd.DataFrame,
    word_df: pd.DataFrame,
    sum_cols: list[str],
) -> pd.DataFrame:
    if summary.empty or not sum_cols:
        return summary
    keys = ["model_name", "pipeline", "category", "step"]
    source = add_all_categories(word_df)
    sums = source.groupby(keys, as_index=False, observed=True)[sum_cols].sum()
    return summary.merge(sums, on=keys, how="left")


def word_rows_from_sums(
    sums: defaultdict[tuple, float],
    counts: defaultdict[tuple, int],
    metric_col: str,
    row_count_col: str,
) -> pd.DataFrame:
    rows = []
    for word_key in sorted(counts):
        total = counts[word_key]
        if total:
            rows.append(
                {
                    "model_name": word_key[0],
                    "pipeline": word_key[1],
                    "category": word_key[2],
                    "step": word_key[3],
                    "original_word": word_key[4],
                    metric_col: sums[word_key] / total,
                    row_count_col: total,
                }
            )
    return pd.DataFrame(rows)


def read_filtered_chunks(
    input_csv: Path,
    usecols: list[str],
    models: list[str],
    pipelines: list[str],
    chunk_size: int,
    progress_desc: str,
    total_rows: int | None,
    disable_progress: bool,
) -> Iterable[pd.DataFrame]:
    reader = pd.read_csv(input_csv, usecols=usecols, chunksize=chunk_size, dtype="string")
    with progress_bar(
        total=total_rows,
        desc=progress_desc,
        unit="rows",
        disable=disable_progress,
    ) as bar:
        for chunk in reader:
            if bar is not None:
                bar.update(len(chunk))
            chunk["pipeline"] = prompt_mode_to_pipeline(chunk["prompt_mode"])
            chunk = chunk[
                chunk["model_name"].isin(models)
                & chunk["pipeline"].isin(pipelines)
            ].copy()
            if chunk.empty:
                continue
            chunk["step"] = pd.to_numeric(chunk["step"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["step"].notna()].copy()
            chunk["step"] = chunk["step"].astype(int)
            yield chunk


def add_word_mean_stats(
    sums: defaultdict[tuple, float],
    counts: defaultdict[tuple, int],
    key_columns: pd.DataFrame,
    values: pd.Series,
) -> None:
    """Accumulate row-level metric values under a word-specific key."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    if not valid.any():
        return
    data = key_columns.loc[valid].copy()
    data["_metric_value"] = numeric.loc[valid]
    grouped = data.groupby(
        ["model_name", "pipeline", "category", "step", "original_word"],
        observed=True,
    )["_metric_value"].agg(["sum", "count"])
    for key, row in grouped.iterrows():
        sums[key] += float(row["sum"])
        counts[key] += int(row["count"])


def rows_from_word_means(
    sums: defaultdict[tuple, float],
    counts: defaultdict[tuple, int],
    metric_col: str,
    word_count_col: str,
    row_count_col: str,
) -> pd.DataFrame:
    """Average per-word means into model/pipeline/category/step means."""
    metric_sum = defaultdict(float)
    word_counts = defaultdict(int)
    row_counts = defaultdict(int)

    for word_key in sorted(counts):
        base_key = word_key[:4]
        if counts[word_key]:
            metric_sum[base_key] += sums[word_key] / counts[word_key]
            word_counts[base_key] += 1
            row_counts[base_key] += counts[word_key]

    rows = []
    for key in sorted(word_counts):
        rows.append(
            {
                "model_name": key[0],
                "pipeline": key[1],
                "category": key[2],
                "step": key[3],
                metric_col: metric_sum[key] / word_counts[key],
                word_count_col: word_counts[key],
                row_count_col: row_counts[key],
            }
        )
    return pd.DataFrame(rows)


def compute_fast_metrics(
    input_csv: Path,
    metrics: list[str],
    models: list[str],
    pipelines: list[str],
    chunk_size: int,
    total_rows: int | None,
    disable_progress: bool,
    bootstrap_reps: int,
    ci_level: float,
    bootstrap_seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    usecols = ["model_name", "category", "original_word", "step", "guess", "prompt_mode"]
    exact_word_correct = defaultdict(int)
    exact_word_total = defaultdict(int)
    entropy_word_counts: defaultdict[tuple, Counter[str]] = defaultdict(Counter)

    for chunk in read_filtered_chunks(
        input_csv,
        usecols,
        models,
        pipelines,
        chunk_size,
        progress_desc="Reading CSV for exact/entropy",
        total_rows=total_rows,
        disable_progress=disable_progress,
    ):
        chunk["word_norm"] = normalize_word_series(chunk["original_word"])
        chunk["guess_norm"] = normalize_guess_series(chunk["guess"])
        group_cols = ["model_name", "pipeline", "category", "step"]

        if "exact" in metrics:
            chunk["exact_correct"] = chunk["word_norm"].eq(chunk["guess_norm"])
            grouped = chunk.groupby(group_cols + ["original_word"], observed=True)["exact_correct"].agg(["sum", "count"])
            for key, row in grouped.iterrows():
                exact_word_correct[key] += int(row["sum"])
                exact_word_total[key] += int(row["count"])

        if "entropy" in metrics:
            grouped_counts = (
                chunk.groupby(group_cols + ["original_word", "guess_norm"], observed=True)
                .size()
                .reset_index(name="n")
            )
            for row in grouped_counts.itertuples(index=False):
                key = (
                    row.model_name,
                    row.pipeline,
                    row.category,
                    int(row.step),
                    row.original_word,
                )
                guess = str(row.guess_norm)
                if guess:
                    entropy_word_counts[key][guess] += int(row.n)

    results: dict[str, pd.DataFrame] = {}
    word_tables: dict[str, pd.DataFrame] = {}

    if "exact" in metrics:
        rows = []
        for word_key in sorted(exact_word_total):
            total = exact_word_total[word_key]
            if not total:
                continue
            rows.append(
                {
                    "model_name": word_key[0],
                    "pipeline": word_key[1],
                    "category": word_key[2],
                    "step": word_key[3],
                    "original_word": word_key[4],
                    "exact_accuracy": exact_word_correct[word_key] / total,
                    "exact_correct": exact_word_correct[word_key],
                    "exact_n": total,
                }
            )
        word_tables["exact"] = pd.DataFrame(rows)
        results["exact"] = aggregate_word_metric_table(
            word_tables["exact"],
            metric_col="exact_accuracy",
            word_count_col="exact_word_n",
            row_count_col="exact_n",
            row_count_source_col="exact_n",
            bootstrap_reps=bootstrap_reps,
            ci_level=ci_level,
            seed=bootstrap_seed,
        )
        results["exact"] = add_group_sums(results["exact"], word_tables["exact"], ["exact_correct"])

    if "entropy" in metrics:
        rows = []
        for word_key in sorted(entropy_word_counts):
            counts = entropy_word_counts[word_key]
            total = sum(counts.values())
            if total == 0:
                continue
            probs = [count / total for count in counts.values()]
            shannon = -sum(p * math.log2(p) for p in probs)
            normalized = shannon / math.log2(total) if total > 1 else 0.0
            top_guesses = "; ".join(f"{guess} ({count})" for guess, count in counts.most_common(5))
            rows.append(
                {
                    "model_name": word_key[0],
                    "pipeline": word_key[1],
                    "category": word_key[2],
                    "step": word_key[3],
                    "original_word": word_key[4],
                    "shannon_entropy": shannon,
                    "normalized_entropy": normalized,
                    "unique_guesses": len(counts),
                    "entropy_n": total,
                    "top_guesses": top_guesses,
                }
            )
        word_tables["entropy"] = pd.DataFrame(rows)
        entropy_summary = aggregate_word_metric_table(
            word_tables["entropy"],
            metric_col="shannon_entropy",
            word_count_col="entropy_word_n",
            row_count_col="entropy_n",
            row_count_source_col="entropy_n",
            bootstrap_reps=bootstrap_reps,
            ci_level=ci_level,
            seed=bootstrap_seed,
        )
        normalized_summary = aggregate_word_metric_table(
            word_tables["entropy"],
            metric_col="normalized_entropy",
            word_count_col="entropy_word_n",
            row_count_col="entropy_n",
            row_count_source_col="entropy_n",
            bootstrap_reps=bootstrap_reps,
            ci_level=ci_level,
            seed=bootstrap_seed,
        ).drop(columns=["entropy_word_n", "entropy_n"], errors="ignore")
        unique_summary = aggregate_word_metric_table(
            word_tables["entropy"],
            metric_col="unique_guesses",
            word_count_col="entropy_word_n",
            row_count_col="entropy_n",
            row_count_source_col="entropy_n",
            bootstrap_reps=bootstrap_reps,
            ci_level=ci_level,
            seed=bootstrap_seed,
        ).drop(columns=["entropy_word_n", "entropy_n"], errors="ignore")
        unique_summary = unique_summary.rename(
            columns={
                "unique_guesses": "mean_unique_guesses",
                "unique_guesses_se": "mean_unique_guesses_se",
                "unique_guesses_ci_low": "mean_unique_guesses_ci_low",
                "unique_guesses_ci_high": "mean_unique_guesses_ci_high",
                "unique_guesses_ci_level": "mean_unique_guesses_ci_level",
            }
        )
        results["entropy"] = (
            entropy_summary
            .merge(normalized_summary, on=["model_name", "pipeline", "category", "step"], how="outer")
            .merge(unique_summary, on=["model_name", "pipeline", "category", "step"], how="outer")
        )

    return results, word_tables


class EmbeddingCache:
    def __init__(self, path: Path, embedding_model_name: str):
        self.path = path
        self.embedding_model_name = embedding_model_name
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model_name TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (model_name, text_hash)
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, texts: list[str]) -> dict[str, np.ndarray]:
        unique = list(dict.fromkeys(t for t in texts if t))
        found: dict[str, np.ndarray] = {}
        if not unique:
            return found

        hashes = [(self.text_hash(text), text) for text in unique]
        hash_to_text = {h: text for h, text in hashes}
        hash_values = list(hash_to_text)
        for start in range(0, len(hash_values), 900):
            batch_hashes = hash_values[start : start + 900]
            placeholders = ",".join(["?"] * len(batch_hashes))
            params = [self.embedding_model_name, *batch_hashes]
            rows = self.conn.execute(
                f"""
                SELECT text_hash, text, dim, vector
                FROM embeddings
                WHERE model_name = ? AND text_hash IN ({placeholders})
                """,
                params,
            ).fetchall()
            for text_hash, text, dim, vector in rows:
                expected_text = hash_to_text.get(text_hash)
                if expected_text == text:
                    found[text] = np.frombuffer(vector, dtype=np.float32, count=dim)
        return found

    def put_many(self, embeddings: dict[str, np.ndarray]) -> None:
        rows = []
        for text, vector in embeddings.items():
            array = np.asarray(vector, dtype=np.float32)
            rows.append(
                (
                    self.embedding_model_name,
                    self.text_hash(text),
                    text,
                    int(array.shape[0]),
                    array.tobytes(),
                )
            )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO embeddings
            (model_name, text_hash, text, dim, vector)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def ensure(
        self,
        texts: Iterable[str],
        encoder,
        batch_size: int,
    ) -> dict[str, np.ndarray]:
        unique = list(dict.fromkeys(str(text) for text in texts if isinstance(text, str) and text.strip()))
        found = self.get_many(unique)
        missing = [text for text in unique if text not in found]
        if missing:
            print(f"Encoding {len(missing):,} missing texts for {self.embedding_model_name}...")
            encoded = encoder.encode(
                missing,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            new_embeddings = {
                text: np.asarray(vector, dtype=np.float32)
                for text, vector in zip(missing, encoded)
            }
            self.put_many(new_embeddings)
            found.update(new_embeddings)
        return found

    def close(self) -> None:
        self.conn.close()


def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "The sentence-transformers package is required for embedding metrics. "
            "It can be installed with: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


def cosine_from_cache(text_a: str, text_b: str, embeddings: dict[str, np.ndarray]) -> float:
    if not text_a or not text_b:
        return np.nan
    vec_a = embeddings.get(text_a)
    vec_b = embeddings.get(text_b)
    if vec_a is None or vec_b is None:
        return np.nan
    return float(np.dot(vec_a, vec_b))


def compute_guess_similarity(
    input_csv: Path,
    models: list[str],
    pipelines: list[str],
    chunk_size: int,
    cache: EmbeddingCache,
    encoder,
    batch_size: int,
    total_rows: int | None,
    disable_progress: bool,
    bootstrap_reps: int,
    ci_level: float,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = ["model_name", "category", "original_word", "step", "guess", "prompt_mode"]
    word_sums = defaultdict(float)
    word_counts = defaultdict(int)

    for chunk in read_filtered_chunks(
        input_csv,
        usecols,
        models,
        pipelines,
        chunk_size,
        progress_desc="Reading CSV for word-guess similarity",
        total_rows=total_rows,
        disable_progress=disable_progress,
    ):
        chunk["word_norm"] = normalize_word_series(chunk["original_word"])
        chunk["guess_norm"] = normalize_guess_series(chunk["guess"])
        texts = pd.concat([chunk["word_norm"], chunk["guess_norm"]]).dropna().astype(str).unique().tolist()
        embeddings = cache.ensure(texts, encoder, batch_size)
        sims = [
            cosine_from_cache(word, guess, embeddings)
            for word, guess in zip(chunk["word_norm"], chunk["guess_norm"])
        ]
        chunk["word_guess_semantic_similarity"] = sims
        add_word_mean_stats(
            word_sums,
            word_counts,
            chunk[["model_name", "pipeline", "category", "step", "original_word"]],
            chunk["word_guess_semantic_similarity"],
        )

    word_df = word_rows_from_sums(
        word_sums,
        word_counts,
        metric_col="word_guess_semantic_similarity",
        row_count_col="word_guess_similarity_row_n",
    )
    summary = aggregate_word_metric_table(
        word_df,
        metric_col="word_guess_semantic_similarity",
        word_count_col="word_guess_similarity_n",
        row_count_col="word_guess_similarity_row_n",
        row_count_source_col="word_guess_similarity_row_n",
        bootstrap_reps=bootstrap_reps,
        ci_level=ci_level,
        seed=bootstrap_seed,
    )
    return summary, word_df


def load_step0_reference_map(
    input_csv: Path,
    models: list[str],
    pipelines: list[str],
    chunk_size: int,
    total_rows: int | None,
    disable_progress: bool,
) -> dict[tuple[str, str, str, str, str], str]:
    usecols = [
        "model_name",
        "category",
        "original_word",
        "instance_id",
        "step",
        "description",
        "prompt_mode",
    ]
    refs: dict[tuple[str, str, str, str, str], str] = {}
    for chunk in read_filtered_chunks(
        input_csv,
        usecols,
        models,
        pipelines,
        chunk_size,
        progress_desc="Reading CSV for step-0 descriptions",
        total_rows=total_rows,
        disable_progress=disable_progress,
    ):
        chunk = chunk[chunk["step"].eq(0)]
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            key = (
                row.model_name,
                row.pipeline,
                row.category,
                row.original_word,
                row.instance_id,
            )
            refs[key] = "" if pd.isna(row.description) else str(row.description)
    return refs


def reference_descriptions_for_chunk(
    chunk: pd.DataFrame,
    refs: dict[tuple[str, str, str, str, str], str],
) -> list[str]:
    output = []
    for row in chunk[["model_name", "pipeline", "category", "original_word", "instance_id"]].itertuples(index=False):
        output.append(refs.get(tuple(row), ""))
    return output


def compute_description_similarity(
    input_csv: Path,
    models: list[str],
    pipelines: list[str],
    chunk_size: int,
    cache: EmbeddingCache,
    encoder,
    batch_size: int,
    refs: dict[tuple[str, str, str, str, str], str],
    total_rows: int | None,
    disable_progress: bool,
    bootstrap_reps: int,
    ci_level: float,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "model_name",
        "category",
        "original_word",
        "instance_id",
        "step",
        "description",
        "prompt_mode",
    ]
    word_sums = defaultdict(float)
    word_counts = defaultdict(int)

    for chunk in read_filtered_chunks(
        input_csv,
        usecols,
        models,
        pipelines,
        chunk_size,
        progress_desc="Reading CSV for description similarity",
        total_rows=total_rows,
        disable_progress=disable_progress,
    ):
        chunk["description"] = chunk["description"].fillna("").astype(str)
        chunk["original_description"] = reference_descriptions_for_chunk(chunk, refs)
        texts = pd.concat([chunk["description"], chunk["original_description"]]).dropna().astype(str).unique().tolist()
        embeddings = cache.ensure(texts, encoder, batch_size)
        sims = [
            cosine_from_cache(desc, ref, embeddings)
            for desc, ref in zip(chunk["description"], chunk["original_description"])
        ]
        chunk["description_semantic_similarity_to_step0"] = sims
        add_word_mean_stats(
            word_sums,
            word_counts,
            chunk[["model_name", "pipeline", "category", "step", "original_word"]],
            chunk["description_semantic_similarity_to_step0"],
        )

    word_df = word_rows_from_sums(
        word_sums,
        word_counts,
        metric_col="description_semantic_similarity_to_step0",
        row_count_col="description_similarity_row_n",
    )
    summary = aggregate_word_metric_table(
        word_df,
        metric_col="description_semantic_similarity_to_step0",
        word_count_col="description_similarity_n",
        row_count_col="description_similarity_row_n",
        row_count_source_col="description_similarity_row_n",
        bootstrap_reps=bootstrap_reps,
        ci_level=ci_level,
        seed=bootstrap_seed,
    )
    return summary, word_df


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize_for_bleu(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    ref_tokens = tokenize_for_bleu(reference)
    cand_tokens = tokenize_for_bleu(candidate)
    if not ref_tokens or not cand_tokens:
        return np.nan

    precisions = []
    for n in range(1, max_n + 1):
        cand_counts = ngram_counts(cand_tokens, n)
        ref_counts = ngram_counts(ref_tokens, n)
        if not cand_counts:
            precisions.append(1.0)
            continue
        clipped = sum(min(count, ref_counts[ngram]) for ngram, count in cand_counts.items())
        total = sum(cand_counts.values())
        precisions.append((clipped + 1.0) / (total + 1.0))

    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    bp = 1.0 if cand_len > ref_len else math.exp(1.0 - (ref_len / cand_len))
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n))


def compute_description_bleu(
    input_csv: Path,
    models: list[str],
    pipelines: list[str],
    chunk_size: int,
    refs: dict[tuple[str, str, str, str, str], str],
    total_rows: int | None,
    disable_progress: bool,
    bootstrap_reps: int,
    ci_level: float,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "model_name",
        "category",
        "original_word",
        "instance_id",
        "step",
        "description",
        "prompt_mode",
    ]
    word_sums = defaultdict(float)
    word_counts = defaultdict(int)

    for chunk in read_filtered_chunks(
        input_csv,
        usecols,
        models,
        pipelines,
        chunk_size,
        progress_desc="Reading CSV for description BLEU",
        total_rows=total_rows,
        disable_progress=disable_progress,
    ):
        chunk["description"] = chunk["description"].fillna("").astype(str)
        chunk["original_description"] = reference_descriptions_for_chunk(chunk, refs)
        chunk["description_bleu_to_step0"] = [
            sentence_bleu(ref, desc)
            for desc, ref in zip(chunk["description"], chunk["original_description"])
        ]
        add_word_mean_stats(
            word_sums,
            word_counts,
            chunk[["model_name", "pipeline", "category", "step", "original_word"]],
            chunk["description_bleu_to_step0"],
        )

    word_df = word_rows_from_sums(
        word_sums,
        word_counts,
        metric_col="description_bleu_to_step0",
        row_count_col="description_bleu_row_n",
    )
    summary = aggregate_word_metric_table(
        word_df,
        metric_col="description_bleu_to_step0",
        word_count_col="description_bleu_n",
        row_count_col="description_bleu_row_n",
        row_count_source_col="description_bleu_row_n",
        bootstrap_reps=bootstrap_reps,
        ci_level=ci_level,
        seed=bootstrap_seed,
    )
    return summary, word_df


def merge_metric_tables(metric_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = ["model_name", "pipeline", "category", "step"]
    overview: pd.DataFrame | None = None
    for table in metric_tables.values():
        if table.empty:
            continue
        overview = table if overview is None else overview.merge(table, on=keys, how="outer")
    if overview is None:
        return pd.DataFrame(columns=keys)
    overview["model_name"] = pd.Categorical(overview["model_name"], MODEL_ORDER, ordered=True)
    overview["pipeline"] = pd.Categorical(overview["pipeline"], ["pipeline_a", "pipeline_b"], ordered=True)
    overview["category"] = pd.Categorical(overview["category"], CATEGORY_OUTPUT_ORDER, ordered=True)
    return overview.sort_values(keys).reset_index(drop=True)


def metric_weight_column(metric_col: str) -> str | None:
    return {
        "exact_accuracy": "exact_word_n",
        "shannon_entropy": "entropy_word_n",
        "normalized_entropy": "entropy_word_n",
        "word_guess_semantic_similarity": "word_guess_similarity_n",
        "description_semantic_similarity_to_step0": "description_similarity_n",
        "description_bleu_to_step0": "description_bleu_n",
    }.get(metric_col)


def aggregate_all_categories(overview: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    all_rows = overview[overview["category"].astype(str).eq("All Categories")].copy()
    if not all_rows.empty:
        return all_rows

    weight_col = metric_weight_column(metric_col)
    rows = []
    for key, sub in overview.groupby(["model_name", "pipeline", "step"], observed=True):
        values = pd.to_numeric(sub[metric_col], errors="coerce")
        valid = values.notna()
        if not valid.any():
            continue
        if weight_col and weight_col in sub.columns:
            weights = pd.to_numeric(sub.loc[valid, weight_col], errors="coerce").fillna(0)
            value = np.average(values.loc[valid], weights=weights) if weights.sum() else values.loc[valid].mean()
        else:
            value = values.loc[valid].mean()
        rows.append(
            {
                "model_name": key[0],
                "pipeline": key[1],
                "category": "All Categories",
                "step": key[2],
                metric_col: value,
            }
        )
    return pd.DataFrame(rows)


def load_human_accuracy(human_csv: Path) -> pd.DataFrame:
    human = pd.read_csv(human_csv)
    rows = []
    for row in human.itertuples(index=False):
        category = row.Category
        rows.append({"category": category, "step": 0, "human_exact_accuracy": 1.0})
        for step in range(1, 11):
            col = f"Correct_Until_Gen_{step + 1}"
            if hasattr(row, col):
                rows.append(
                    {
                        "category": category,
                        "step": step,
                        "human_exact_accuracy": float(getattr(row, col)) / 100.0,
                    }
                )
    category_df = pd.DataFrame(rows)
    overall = (
        category_df.groupby("step", as_index=False)["human_exact_accuracy"]
        .mean()
        .assign(category="All Categories")
    )
    return pd.concat([category_df, overall], ignore_index=True)


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "The matplotlib package is required for plotting. "
            "It can be installed with: pip install matplotlib"
        ) from exc
    return plt


def plot_metric(
    overview: pd.DataFrame,
    metric_col: str,
    pipeline: str,
    category: str,
    models: list[str],
    output_path: Path,
    human_accuracy: pd.DataFrame | None = None,
) -> None:
    plt = require_matplotlib()
    data = aggregate_all_categories(overview, metric_col) if category == "All Categories" else overview[overview["category"].eq(category)]
    data = data[data["pipeline"].eq(pipeline)]

    plt.figure(figsize=(11, 6.5))
    for model in models:
        sub = data[data["model_name"].eq(model)].sort_values("step")
        if sub.empty:
            continue
        plt.plot(
            sub["step"],
            sub[metric_col],
            marker="o",
            linewidth=2.0,
            label=model,
            color=MODEL_COLORS.get(model),
        )
        ci_low = f"{metric_col}_ci_low"
        ci_high = f"{metric_col}_ci_high"
        if ci_low in sub.columns and ci_high in sub.columns:
            lower = pd.to_numeric(sub[ci_low], errors="coerce")
            upper = pd.to_numeric(sub[ci_high], errors="coerce")
            if lower.notna().any() and upper.notna().any():
                plt.fill_between(
                    sub["step"],
                    lower,
                    upper,
                    color=MODEL_COLORS.get(model),
                    alpha=0.14,
                    linewidth=0,
                )

    if human_accuracy is not None and pipeline == "pipeline_a" and metric_col == "exact_accuracy":
        human_sub = human_accuracy[human_accuracy["category"].eq(category)].sort_values("step")
        if not human_sub.empty:
            plt.plot(
                human_sub["step"],
                human_sub["human_exact_accuracy"],
                linestyle="--",
                linewidth=2.2,
                color="#666666",
                label="Human Study",
            )

    plt.xlabel("Step")
    plt.ylabel(METRIC_TITLES.get(metric_col, metric_col))
    plt.title(f"{PIPELINE_LABELS[pipeline]} - {METRIC_TITLES.get(metric_col, metric_col)} - {category}")
    plt.xticks(range(0, 11))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_tables_and_plots(
    overview: pd.DataFrame,
    metrics: list[str],
    models: list[str],
    pipelines: list[str],
    paths: OutputPaths,
    include_human_study: bool,
    human_csv: Path,
    skip_plots: bool,
) -> None:
    if overview.empty:
        raise RuntimeError("No metric rows were computed. Check the selected models and pipelines.")

    human_accuracy = load_human_accuracy(human_csv) if include_human_study else None

    for pipeline in pipelines:
        pipeline_overview = overview[overview["pipeline"].eq(pipeline)].copy()
        pipeline_overview.to_csv(paths.tables / pipeline / "overview_metrics.csv", index=False)

        final_step = int(pipeline_overview["step"].max())
        pipeline_overview[pipeline_overview["step"].eq(final_step)].to_csv(
            paths.tables / pipeline / "final_step_metrics.csv",
            index=False,
        )

    overview.to_csv(paths.tables / "overview_metrics_all_selected_pipelines.csv", index=False)
    if skip_plots:
        return

    metric_cols = []
    for metric in metrics:
        if metric == "entropy":
            metric_cols.extend(["shannon_entropy", "normalized_entropy"])
        else:
            metric_cols.append(METRIC_COLUMNS[metric])

    category_values = CATEGORY_OUTPUT_ORDER
    for pipeline in pipelines:
        for metric_col in metric_cols:
            if metric_col not in overview.columns:
                continue
            all_category = aggregate_all_categories(overview, metric_col)
            if not all_category.empty:
                all_category[all_category["pipeline"].eq(pipeline)].to_csv(
                    paths.tables / pipeline / f"{metric_col}_all_categories.csv",
                    index=False,
                )
            for category in category_values:
                filename = f"{metric_col}_{slugify(category)}.png"
                plot_metric(
                    overview=overview,
                    metric_col=metric_col,
                    pipeline=pipeline,
                    category=category,
                    models=models,
                    output_path=paths.figures / pipeline / filename,
                    human_accuracy=human_accuracy,
                )


ENTROPY_DIAGNOSTIC_LABELS = [
    ("low_entropy_high_accuracy", "Low entropy + high accuracy", "stable preservation"),
    ("low_entropy_low_accuracy", "Low entropy + low accuracy", "stable wrong sink"),
    ("high_entropy_high_similarity", "High entropy + high similarity", "diverse synonyms"),
    ("high_entropy_low_similarity", "High entropy + low similarity", "unstable semantic collapse"),
]


def classify_entropy_diagnostic(
    normalized_entropy: float,
    exact_accuracy: float,
    semantic_similarity: float,
    entropy_threshold: float,
    accuracy_threshold: float,
    similarity_threshold: float,
) -> str:
    if normalized_entropy < entropy_threshold:
        if exact_accuracy >= accuracy_threshold:
            return "low_entropy_high_accuracy"
        return "low_entropy_low_accuracy"
    if semantic_similarity >= similarity_threshold:
        return "high_entropy_high_similarity"
    return "high_entropy_low_similarity"


def compute_entropy_diagnostics(
    word_tables: dict[str, pd.DataFrame],
    entropy_threshold: float,
    accuracy_threshold: float,
    similarity_threshold: float,
    max_examples: int,
) -> pd.DataFrame:
    required = {"entropy", "exact", "guess_similarity"}
    if not required.issubset(word_tables):
        return pd.DataFrame()

    keys = ["model_name", "pipeline", "category", "step", "original_word"]
    diagnostics = (
        word_tables["entropy"][keys + ["normalized_entropy", "shannon_entropy", "top_guesses"]]
        .merge(word_tables["exact"][keys + ["exact_accuracy"]], on=keys, how="inner")
        .merge(
            word_tables["guess_similarity"][keys + ["word_guess_semantic_similarity"]],
            on=keys,
            how="inner",
        )
    )
    if diagnostics.empty:
        return diagnostics

    diagnostics["diagnostic_id"] = [
        classify_entropy_diagnostic(
            float(row.normalized_entropy),
            float(row.exact_accuracy),
            float(row.word_guess_semantic_similarity),
            entropy_threshold,
            accuracy_threshold,
            similarity_threshold,
        )
        for row in diagnostics.itertuples(index=False)
    ]

    diagnostic_lookup = {
        diagnostic_id: (label, interpretation)
        for diagnostic_id, label, interpretation in ENTROPY_DIAGNOSTIC_LABELS
    }
    diagnostics["diagnostic_label"] = diagnostics["diagnostic_id"].map(lambda value: diagnostic_lookup[value][0])
    diagnostics["interpretation"] = diagnostics["diagnostic_id"].map(lambda value: diagnostic_lookup[value][1])
    diagnostics = add_all_categories(diagnostics)

    group_keys = ["model_name", "pipeline", "category", "step"]
    rows = []
    for key, sub in diagnostics.groupby(group_keys, observed=True):
        total_words = len(sub)
        for diagnostic_id, label, interpretation in ENTROPY_DIAGNOSTIC_LABELS:
            cell = sub[sub["diagnostic_id"].eq(diagnostic_id)]
            examples = cell.sort_values(
                ["normalized_entropy", "word_guess_semantic_similarity"],
                ascending=[diagnostic_id.startswith("low_entropy"), False],
            ).head(max_examples)
            rows.append(
                {
                    "model_name": key[0],
                    "pipeline": key[1],
                    "category": key[2],
                    "step": key[3],
                    "diagnostic_id": diagnostic_id,
                    "diagnostic_label": label,
                    "interpretation": interpretation,
                    "word_n": int(len(cell)),
                    "word_share": float(len(cell) / total_words) if total_words else np.nan,
                    "mean_normalized_entropy": float(cell["normalized_entropy"].mean()) if not cell.empty else np.nan,
                    "mean_shannon_entropy": float(cell["shannon_entropy"].mean()) if not cell.empty else np.nan,
                    "mean_exact_accuracy": float(cell["exact_accuracy"].mean()) if not cell.empty else np.nan,
                    "mean_word_guess_semantic_similarity": (
                        float(cell["word_guess_semantic_similarity"].mean()) if not cell.empty else np.nan
                    ),
                    "example_words": "; ".join(examples["original_word"].astype(str).tolist()),
                    "example_top_guesses": " | ".join(
                        f"{row.original_word}: {row.top_guesses}"
                        for row in examples[["original_word", "top_guesses"]].itertuples(index=False)
                    ),
                    "entropy_threshold": entropy_threshold,
                    "accuracy_threshold": accuracy_threshold,
                    "similarity_threshold": similarity_threshold,
                }
            )
    return pd.DataFrame(rows)


def save_word_tables(word_tables: dict[str, pd.DataFrame], paths: OutputPaths, pipelines: list[str]) -> None:
    for name, table in word_tables.items():
        if table.empty:
            continue
        table.to_csv(paths.tables / f"word_level_{name}.csv", index=False)
        for pipeline in pipelines:
            table[table["pipeline"].eq(pipeline)].to_csv(
                paths.tables / pipeline / f"word_level_{name}.csv",
                index=False,
            )


def save_entropy_diagnostics(diagnostics: pd.DataFrame, paths: OutputPaths, pipelines: list[str]) -> None:
    if diagnostics.empty:
        return
    diagnostics.to_csv(paths.tables / "entropy_diagnostic_2x2.csv", index=False)
    for pipeline in pipelines:
        diagnostics[diagnostics["pipeline"].eq(pipeline)].to_csv(
            paths.tables / pipeline / "entropy_diagnostic_2x2.csv",
            index=False,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Semantic Drift model-comparison analysis from the unified CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root() / "datasets" / "unified_semantic_drift_results.csv",
        help="Path to the unified CSV.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["quick"],
        help=(
            "Metrics to run: quick, all, exact, entropy, guess-sim, "
            "description-sim, bleu, semantic. Comma-separated values also work."
        ),
    )
    parser.add_argument(
        "--model-group",
        choices=sorted(MODEL_GROUPS),
        default="all",
        help="Predefined model comparison group.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Explicit model names or slugs. Overrides --model-group.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=["pipeline_a", "pipeline_b"],
        default=["pipeline_a", "pipeline_b"],
        help="Pipelines to analyze separately.",
    )
    parser.add_argument(
        "--include-human-study",
        action="store_true",
        help="Add gray dashed human-study line to exact-accuracy Pipeline A plots.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV tables only. Useful for smoke tests or headless metric runs.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--skip-row-count",
        action="store_true",
        help=(
            "Do not pre-count CSV rows for ETA. Progress bars will still show "
            "rows processed, but not total remaining time."
        ),
    )
    parser.add_argument(
        "--human-study-csv",
        type=Path,
        default=repo_root() / "human_study" / "all_chains_grouped_by_category_percentages.csv",
        help="Human-study category percentage CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "data_analysis" / "results_analysis",
        help="Base directory for generated CSVs, figures, and caches.",
    )
    parser.add_argument(
        "--comparison-name",
        help="Output subfolder name. Defaults to model group plus selected metrics.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Rows to process per chunk for non-embedding metrics.",
    )
    parser.add_argument(
        "--embedding-chunk-size",
        type=int,
        default=50_000,
        help="Rows to process per chunk for embedding-heavy metrics.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "SentenceTransformer model used for semantic similarity. "
            "Aliases include submitted/mpnet, e5-large-v2, and bge-large-en-v1.5."
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Batch size for SentenceTransformer encoding.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=2000,
        help=(
            "Bootstrap replicates for target-word clustered confidence intervals. "
            "Use 0 to write point estimates and standard errors only."
        ),
    )
    parser.add_argument(
        "--bootstrap-ci",
        type=float,
        default=0.95,
        help="Confidence level for target-word clustered bootstrap intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=12345,
        help="Random seed for deterministic bootstrap intervals.",
    )
    parser.add_argument(
        "--entropy-diagnostic-entropy-threshold",
        type=float,
        default=0.5,
        help="Normalized entropy threshold separating low from high entropy in the 2x2 diagnostic table.",
    )
    parser.add_argument(
        "--entropy-diagnostic-accuracy-threshold",
        type=float,
        default=0.5,
        help="Exact-accuracy threshold separating stable preservation from stable wrong sinks.",
    )
    parser.add_argument(
        "--entropy-diagnostic-similarity-threshold",
        type=float,
        default=0.5,
        help="Semantic-similarity threshold separating diverse synonyms from semantic collapse.",
    )
    parser.add_argument(
        "--entropy-diagnostic-examples",
        type=int,
        default=3,
        help="Number of example target words to include in each entropy diagnostic cell.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_csv = args.input.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Unified CSV not found: {input_csv}")
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap-reps must be 0 or greater")
    if not 0 < args.bootstrap_ci < 1:
        raise ValueError("--bootstrap-ci must be between 0 and 1")

    metrics = parse_metrics(args.metrics)
    models = select_models(args.model_group, args.models)
    args.embedding_model = resolve_embedding_model(args.embedding_model)
    comparison_name = args.comparison_name
    if not comparison_name:
        metric_part = "_".join(metrics)
        model_part = args.model_group if not args.models else "_".join(MODEL_SLUGS[m] for m in models)
        comparison_name = f"{model_part}_{metric_part}"
        if EMBEDDING_METRICS.intersection(metrics) and args.embedding_model != DEFAULT_EMBEDDING_MODEL:
            comparison_name = f"{comparison_name}_{slugify(args.embedding_model)}"
    paths = build_output_paths(args.output_dir, comparison_name)

    print("Semantic Drift model-comparison analysis")
    print(f"Input: {input_csv}")
    print(f"Models: {', '.join(models)}")
    print(f"Pipelines: {', '.join(args.pipelines)}")
    print(f"Metrics: {', '.join(metrics)}")
    if EMBEDDING_METRICS.intersection(metrics):
        print(f"Embedding model: {args.embedding_model}")
    print(f"Output: {paths.root}")

    disable_progress = args.no_progress or tqdm is None
    total_rows = None
    if not disable_progress and not args.skip_row_count:
        print("\nCounting CSV rows for progress-bar ETA...")
        input_mb = input_csv.stat().st_size / (1024 * 1024)
        with progress_bar(
            total=input_mb,
            desc="Counting CSV rows",
            unit="MB",
            disable=disable_progress,
        ) as bar:
            if bar is None:
                total_rows = count_csv_rows(input_csv)
            else:
                total_lines = 0
                with input_csv.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        total_lines += block.count(b"\n")
                        bar.update(len(block) / (1024 * 1024))
                total_rows = max(total_lines - 1, 0)
        print(f"Rows in input CSV: {total_rows:,}")
    elif tqdm is None and not args.no_progress:
        print("\nNote: tqdm is not installed, so progress bars are disabled. Install it with: pip install tqdm")

    metric_tables: dict[str, pd.DataFrame] = {}
    word_tables: dict[str, pd.DataFrame] = {}
    fast_metrics = [metric for metric in metrics if metric in {"exact", "entropy"}]
    if fast_metrics:
        print("\nComputing streaming metrics:", ", ".join(fast_metrics))
        fast_tables, fast_word_tables = compute_fast_metrics(
            input_csv=input_csv,
            metrics=fast_metrics,
            models=models,
            pipelines=args.pipelines,
            chunk_size=args.chunk_size,
            total_rows=total_rows,
            disable_progress=disable_progress,
            bootstrap_reps=args.bootstrap_reps,
            ci_level=args.bootstrap_ci,
            bootstrap_seed=args.bootstrap_seed,
        )
        metric_tables.update(fast_tables)
        word_tables.update(fast_word_tables)

    needs_reference_map = any(metric in metrics for metric in {"description_similarity", "description_bleu"})
    refs = {}
    if needs_reference_map:
        print("\nLoading step-0 reference descriptions...")
        refs = load_step0_reference_map(
            input_csv=input_csv,
            models=models,
            pipelines=args.pipelines,
            chunk_size=args.chunk_size,
            total_rows=total_rows,
            disable_progress=disable_progress,
        )
        print(f"Loaded {len(refs):,} step-0 description references")

    embedding_metrics = [metric for metric in metrics if metric in EMBEDDING_METRICS]
    encoder = None
    cache = None
    if embedding_metrics:
        print(f"\nLoading embedding model: {args.embedding_model}")
        encoder = load_sentence_transformer(args.embedding_model)
        cache_path = paths.cache / f"embeddings_{slugify(args.embedding_model)}.sqlite"
        cache = EmbeddingCache(cache_path, args.embedding_model)

    try:
        if "guess_similarity" in metrics:
            print("\nComputing word-guess semantic similarity...")
            metric_tables["guess_similarity"], word_tables["guess_similarity"] = compute_guess_similarity(
                input_csv=input_csv,
                models=models,
                pipelines=args.pipelines,
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

        if "description_similarity" in metrics:
            print("\nComputing description semantic similarity to step 0...")
            metric_tables["description_similarity"], word_tables["description_similarity"] = compute_description_similarity(
                input_csv=input_csv,
                models=models,
                pipelines=args.pipelines,
                chunk_size=args.embedding_chunk_size,
                cache=cache,
                encoder=encoder,
                batch_size=args.embedding_batch_size,
                refs=refs,
                total_rows=total_rows,
                disable_progress=disable_progress,
                bootstrap_reps=args.bootstrap_reps,
                ci_level=args.bootstrap_ci,
                bootstrap_seed=args.bootstrap_seed,
            )
    finally:
        if cache is not None:
            cache.close()

    if "description_bleu" in metrics:
        print("\nComputing description BLEU to step 0...")
        metric_tables["description_bleu"], word_tables["description_bleu"] = compute_description_bleu(
            input_csv=input_csv,
            models=models,
            pipelines=args.pipelines,
            chunk_size=args.embedding_chunk_size,
            refs=refs,
            total_rows=total_rows,
            disable_progress=disable_progress,
            bootstrap_reps=args.bootstrap_reps,
            ci_level=args.bootstrap_ci,
            bootstrap_seed=args.bootstrap_seed,
        )

    overview = merge_metric_tables(metric_tables)
    print("\nWriting tables and plots...")
    save_tables_and_plots(
        overview=overview,
        metrics=metrics,
        models=models,
        pipelines=args.pipelines,
        paths=paths,
        include_human_study=args.include_human_study,
        human_csv=args.human_study_csv,
        skip_plots=args.skip_plots,
    )
    save_word_tables(word_tables, paths, args.pipelines)
    entropy_diagnostics = compute_entropy_diagnostics(
        word_tables=word_tables,
        entropy_threshold=args.entropy_diagnostic_entropy_threshold,
        accuracy_threshold=args.entropy_diagnostic_accuracy_threshold,
        similarity_threshold=args.entropy_diagnostic_similarity_threshold,
        max_examples=args.entropy_diagnostic_examples,
    )
    if not entropy_diagnostics.empty:
        save_entropy_diagnostics(entropy_diagnostics, paths, args.pipelines)
        print("Wrote entropy 2x2 diagnostic tables.")
    elif "entropy" in metrics:
        print(
            "Skipped entropy 2x2 diagnostics; include exact and guess-sim with entropy "
            "to classify all four cells."
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
