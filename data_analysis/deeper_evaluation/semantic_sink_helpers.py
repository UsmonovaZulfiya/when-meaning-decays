"""Helpers for chain-level semantic sink analysis."""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_FROM_PROMPT_MODE = {
    "pipeline_A_guess_then_describe_guessed_word": "pipeline_a",
    "pipeline_B_paraphrase_description": "pipeline_b",
}

MODEL_ORDER = [
    "Llama 3.1 8B Instruct",
    "Llama 3.1 70B Instruct",
    "Gemma 3 4B IT",
    "Gemma 3 12B IT",
    "Gemma 3 27B IT",
]

CATEGORY_ORDER = [
    "High-Freq-Concrete",
    "Low-Freq-Concrete",
    "High-Freq-Abstract",
    "Low-Freq-Abstract",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "with", "you", "your", "they", "their", "them",
}

# Small heuristic lexicons for qualitative signals. These are not a substitute
# for a proper sentiment/politeness model; they are interpretable flags.
HEDGE_WORDS = {
    "about", "almost", "approximately", "could", "generally", "kind", "likely",
    "may", "might", "often", "perhaps", "possibly", "seems", "somewhat",
    "typically", "usually",
}

POLITE_WORDS = {
    "please", "polite", "respectful", "kindly", "thanks", "thank", "courteous",
    "appropriate", "considerate",
}

AFFECTIVE_WORDS = {
    "afraid", "anger", "angry", "anxious", "bad", "beautiful", "fear", "good",
    "happy", "hate", "joy", "love", "negative", "pain", "pleasant", "positive",
    "sad", "shame", "suffering", "threat", "upset", "worry",
}

ABSTRACT_GENERAL_WORDS = {
    "belief", "concept", "consciousness", "experience", "feeling", "idea",
    "knowledge", "meaning", "mental", "mind", "notion", "perception",
    "philosophy", "principle", "process", "quality", "sense", "state",
    "thought", "understanding",
}


def infer_project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "datasets").exists():
            return candidate
    raise FileNotFoundError("Could not find a project root containing datasets/.")


def normalize_word(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-zA-Z\s\-]", "", value)
    return value.strip()


def normalize_series(series: pd.Series) -> pd.Series:
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


def tokenize_text(text) -> list[str]:
    return TOKEN_RE.findall("" if pd.isna(text) else str(text).lower())


def prompt_mode_to_pipeline(series: pd.Series) -> pd.Series:
    return series.map(PIPELINE_FROM_PROMPT_MODE).fillna(series.astype(str).str.lower())


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cached_embeddings(
    texts,
    cache_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[dict[str, np.ndarray], list[str]]:
    texts = list(dict.fromkeys(str(text) for text in texts if isinstance(text, str) and text.strip()))
    embeddings: dict[str, np.ndarray] = {}
    if not cache_path.exists():
        return embeddings, texts

    conn = sqlite3.connect(cache_path)
    try:
        for start in range(0, len(texts), 800):
            batch = texts[start : start + 800]
            hashes = [text_hash(text) for text in batch]
            hash_to_text = dict(zip(hashes, batch))
            placeholders = ",".join(["?"] * len(batch))
            rows = conn.execute(
                f"""
                SELECT text_hash, text, dim, vector
                FROM embeddings
                WHERE model_name = ? AND text_hash IN ({placeholders})
                """,
                [model_name, *hashes],
            ).fetchall()
            for text_hash_value, text, dim, vector in rows:
                if hash_to_text.get(text_hash_value) == text:
                    embeddings[text] = np.frombuffer(vector, dtype=np.float32, count=dim)
    finally:
        conn.close()

    missing = [text for text in texts if text not in embeddings]
    return embeddings, missing


def cosine_cached(text_a: str, text_b: str, embeddings: dict[str, np.ndarray]) -> float:
    vec_a = embeddings.get(text_a)
    vec_b = embeddings.get(text_b)
    if vec_a is None or vec_b is None:
        return np.nan
    return float(np.dot(vec_a, vec_b))


def shannon_entropy(values) -> float:
    counts = Counter(v for v in values if isinstance(v, str) and v)
    total = sum(counts.values())
    if total == 0:
        return np.nan
    probs = [count / total for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def load_target_vocabulary(input_csv: Path, chunk_size: int = 250_000, progress=None) -> pd.DataFrame:
    pieces = []
    reader = pd.read_csv(input_csv, usecols=["category", "original_word"], chunksize=chunk_size, dtype="string")
    iterator = progress(reader, desc="Loading vocabulary chunks") if progress else reader
    for chunk in iterator:
        pieces.append(chunk.drop_duplicates(["category", "original_word"]))
    vocab = pd.concat(pieces, ignore_index=True).drop_duplicates(["category", "original_word"])
    vocab["word_norm"] = normalize_series(vocab["original_word"])
    return vocab.sort_values(["category", "word_norm"]).reset_index(drop=True)


def load_chain_rows(
    input_csv: Path,
    word_to_category: dict[str, str],
    source_words=None,
    source_categories=None,
    models=None,
    pipelines=None,
    include_descriptions: bool = False,
    chunk_size: int = 250_000,
    progress=None,
) -> pd.DataFrame:
    source_word_norms = {normalize_word(word) for word in source_words} if source_words is not None else None
    source_categories = set(source_categories) if source_categories is not None else None
    models = set(models) if models is not None else None
    pipelines = set(pipelines) if pipelines is not None else None

    usecols = ["model_name", "category", "original_word", "instance_id", "step", "guess", "prompt_mode"]
    if include_descriptions:
        usecols.append("description")

    pieces = []
    reader = pd.read_csv(input_csv, usecols=usecols, chunksize=chunk_size, dtype="string")
    iterator = progress(reader, desc="Loading chain rows") if progress else reader
    for chunk in iterator:
        chunk["pipeline"] = prompt_mode_to_pipeline(chunk["prompt_mode"])
        if models is not None:
            chunk = chunk[chunk["model_name"].isin(models)]
        if pipelines is not None:
            chunk = chunk[chunk["pipeline"].isin(pipelines)]
        if source_categories is not None:
            chunk = chunk[chunk["category"].isin(source_categories)]
        chunk["word_norm"] = normalize_series(chunk["original_word"])
        if source_word_norms is not None:
            chunk = chunk[chunk["word_norm"].isin(source_word_norms)]
        if chunk.empty:
            continue

        chunk["guess_norm"] = normalize_guess_series(chunk["guess"])
        chunk["step"] = pd.to_numeric(chunk["step"], errors="coerce")
        chunk["instance_id"] = pd.to_numeric(chunk["instance_id"], errors="coerce")
        chunk = chunk[chunk["step"].notna() & chunk["instance_id"].notna()].copy()
        chunk["step"] = chunk["step"].astype(int)
        chunk["instance_id"] = chunk["instance_id"].astype(int)
        chunk["exact_correct"] = chunk["word_norm"].eq(chunk["guess_norm"])
        chunk["guess_category_exact"] = chunk["guess_norm"].map(word_to_category)
        pieces.append(chunk)

    if not pieces:
        return pd.DataFrame(
            columns=usecols
            + [
                "pipeline",
                "word_norm",
                "guess_norm",
                "exact_correct",
                "guess_category_exact",
            ]
        )

    rows = pd.concat(pieces, ignore_index=True)
    rows["model_name"] = pd.Categorical(rows["model_name"], MODEL_ORDER, ordered=True)
    rows["category"] = pd.Categorical(rows["category"], CATEGORY_ORDER, ordered=True)
    rows = rows.sort_values(["model_name", "pipeline", "category", "original_word", "instance_id", "step"])
    return rows.reset_index(drop=True)


def add_word_guess_similarity(rows: pd.DataFrame, cache_path: Path) -> tuple[pd.DataFrame, list[str]]:
    rows = rows.copy()
    texts = pd.concat([rows["word_norm"], rows["guess_norm"]]).dropna().astype(str).unique().tolist()
    embeddings, missing = load_cached_embeddings(texts, cache_path)
    rows["word_to_guess_similarity"] = [
        cosine_cached(word, guess, embeddings)
        for word, guess in zip(rows["word_norm"], rows["guess_norm"])
    ]
    return rows, missing


def parse_number(value) -> float:
    if pd.isna(value):
        return np.nan
    return pd.to_numeric(str(value).replace(",", "."), errors="coerce")


def load_frequency_lookup(project_root: Path) -> pd.DataFrame:
    """Load available frequency/Zipf metadata for source and guessed words."""
    pieces = []
    weird_path = project_root / "datasets" / "words_weird.csv"
    if weird_path.exists():
        weird = pd.read_csv(weird_path, dtype="string")
        weird["word_norm"] = normalize_series(weird["Spelling"])
        keep = ["word_norm"]
        for col in ["FreqCount", "LogFreq(Zipf)", "CD_count", "CD", "DomPoS"]:
            if col in weird.columns:
                if col != "DomPoS":
                    weird[col] = weird[col].map(parse_number)
                keep.append(col)
        pieces.append(weird[keep].drop_duplicates("word_norm"))

    subtlex_path = project_root / "datasets" / "words_with_usage_count.csv"
    if subtlex_path.exists():
        freq = pd.read_csv(subtlex_path, dtype="string")
        freq["word_norm"] = normalize_series(freq["Word"])
        keep = ["word_norm"]
        for col in ["FREQcount", "CDcount", "SUBTLWF", "Lg10WF", "SUBTLCD", "Lg10CD"]:
            if col in freq.columns:
                freq[col] = freq[col].map(parse_number)
                keep.append(col)
        pieces.append(freq[keep].drop_duplicates("word_norm"))

    if not pieces:
        return pd.DataFrame(columns=["word_norm"])

    lookup = pieces[0]
    for piece in pieces[1:]:
        lookup = lookup.merge(piece, on="word_norm", how="outer")
    return lookup.drop_duplicates("word_norm")


def add_guess_frequency_features(rows: pd.DataFrame, frequency_lookup: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    source_freq = frequency_lookup.add_prefix("source_").rename(columns={"source_word_norm": "word_norm"})
    guess_freq = frequency_lookup.add_prefix("guess_").rename(columns={"guess_word_norm": "guess_norm"})
    rows = rows.merge(source_freq, on="word_norm", how="left")
    rows = rows.merge(guess_freq, on="guess_norm", how="left")
    if "source_LogFreq(Zipf)" in rows.columns and "guess_LogFreq(Zipf)" in rows.columns:
        rows["guess_minus_source_zipf"] = rows["guess_LogFreq(Zipf)"] - rows["source_LogFreq(Zipf)"]
    if "source_Lg10WF" in rows.columns and "guess_Lg10WF" in rows.columns:
        rows["guess_minus_source_lg10wf"] = rows["guess_Lg10WF"] - rows["source_Lg10WF"]
    rows["guess_length_chars"] = rows["guess_norm"].fillna("").astype(str).str.replace(" ", "", regex=False).str.len()
    rows["source_length_chars"] = rows["word_norm"].fillna("").astype(str).str.replace(" ", "", regex=False).str.len()
    rows["guess_minus_source_length"] = rows["guess_length_chars"] - rows["source_length_chars"]
    return rows


def add_description_text_features(rows: pd.DataFrame, text_col: str = "description") -> pd.DataFrame:
    rows = rows.copy()
    token_lists = rows[text_col].map(tokenize_text)
    token_counts = token_lists.map(len)
    type_counts = token_lists.map(lambda toks: len(set(toks)))
    stop_counts = token_lists.map(lambda toks: sum(tok in STOPWORDS for tok in toks))
    hedge_counts = token_lists.map(lambda toks: sum(tok in HEDGE_WORDS for tok in toks))
    polite_counts = token_lists.map(lambda toks: sum(tok in POLITE_WORDS for tok in toks))
    affect_counts = token_lists.map(lambda toks: sum(tok in AFFECTIVE_WORDS for tok in toks))
    abstract_counts = token_lists.map(lambda toks: sum(tok in ABSTRACT_GENERAL_WORDS for tok in toks))

    rows["description_char_len"] = rows[text_col].fillna("").astype(str).str.len()
    rows["description_token_count"] = token_counts
    rows["description_type_count"] = type_counts
    rows["description_ttr"] = np.where(token_counts.gt(0), type_counts / token_counts, np.nan)
    rows["description_avg_token_len"] = token_lists.map(
        lambda toks: float(np.mean([len(tok) for tok in toks])) if toks else np.nan
    )
    rows["description_stopword_rate"] = np.where(token_counts.gt(0), stop_counts / token_counts, np.nan)
    rows["description_hedge_rate"] = np.where(token_counts.gt(0), hedge_counts / token_counts, np.nan)
    rows["description_polite_rate"] = np.where(token_counts.gt(0), polite_counts / token_counts, np.nan)
    rows["description_affective_rate"] = np.where(token_counts.gt(0), affect_counts / token_counts, np.nan)
    rows["description_general_abstract_rate"] = np.where(token_counts.gt(0), abstract_counts / token_counts, np.nan)
    return rows


def add_description_similarity_to_step0(rows: pd.DataFrame, cache_path: Path) -> tuple[pd.DataFrame, list[str]]:
    rows = rows.copy()
    if "description" not in rows.columns:
        raise ValueError("rows must contain a description column")
    key_cols = ["model_name", "pipeline", "category", "original_word", "instance_id"]
    refs = (
        rows[rows["step"].eq(0)][key_cols + ["description"]]
        .drop_duplicates(key_cols)
        .rename(columns={"description": "step0_description"})
    )
    rows = rows.merge(refs, on=key_cols, how="left")
    texts = pd.concat([rows["description"], rows["step0_description"]]).dropna().astype(str).unique().tolist()
    embeddings, missing = load_cached_embeddings(texts, cache_path)
    rows["description_similarity_to_step0"] = [
        cosine_cached(str(desc), str(ref), embeddings)
        for desc, ref in zip(rows["description"].fillna(""), rows["step0_description"].fillna(""))
    ]
    return rows, missing


def chain_metric_deltas(
    rows: pd.DataFrame,
    metric_cols: list[str],
    baseline_step: int = 0,
    final_step: int | None = None,
) -> pd.DataFrame:
    key_cols = ["model_name", "pipeline", "category", "original_word", "word_norm", "instance_id"]
    if final_step is None:
        final_step = int(rows["step"].max())
    base = rows[rows["step"].eq(baseline_step)][key_cols + metric_cols].rename(
        columns={col: f"{col}_step{baseline_step}" for col in metric_cols}
    )
    final = rows[rows["step"].eq(final_step)][key_cols + metric_cols].rename(
        columns={col: f"{col}_step{final_step}" for col in metric_cols}
    )
    deltas = base.merge(final, on=key_cols, how="inner")
    for col in metric_cols:
        deltas[f"{col}_delta_{baseline_step}_to_{final_step}"] = (
            deltas[f"{col}_step{final_step}"] - deltas[f"{col}_step{baseline_step}"]
        )
    return deltas


def aggregate_chain_deltas_word_first(deltas: pd.DataFrame, delta_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    word_level = (
        deltas.groupby(["model_name", "pipeline", "category", "original_word", "word_norm"], observed=True)
        .agg(chain_n=("instance_id", "size"), **{col: (col, "mean") for col in delta_cols})
        .reset_index()
    )
    category_level = (
        word_level.groupby(["model_name", "pipeline", "category"], observed=True)
        .agg(source_word_n=("word_norm", "nunique"), **{col: (col, "mean") for col in delta_cols})
        .reset_index()
        .sort_values(["pipeline", "model_name", "category"])
    )
    return word_level, category_level


def summarize_chain_hits(
    rows: pd.DataFrame,
    sink_word_norm: str | None = None,
    sink_category: str | None = None,
    sink_flag_col: str | None = None,
) -> pd.DataFrame:
    group_cols = ["model_name", "pipeline", "category", "original_word", "word_norm", "instance_id"]
    summaries = []
    for key, group in rows.sort_values(group_cols + ["step"]).groupby(group_cols, observed=True):
        group = group.sort_values("step")
        final = group[group["step"].eq(group["step"].max())].iloc[-1]
        non_initial = group[group["step"].gt(0)]
        first_error = non_initial[~non_initial["exact_correct"]]

        if sink_flag_col is not None:
            sink_mask = group[sink_flag_col].fillna(False).astype(bool)
        elif sink_word_norm is not None:
            sink_mask = group["guess_norm"].eq(sink_word_norm)
        elif sink_category is not None:
            sink_mask = group["guess_category_exact"].eq(sink_category)
        else:
            raise ValueError("Provide sink_word_norm, sink_category, or sink_flag_col.")

        sink_hits = group[sink_mask]
        summaries.append(
            {
                "model_name": key[0],
                "pipeline": key[1],
                "category": key[2],
                "original_word": key[3],
                "word_norm": key[4],
                "instance_id": key[5],
                "observed_steps": group["step"].nunique(),
                "first_error_step": int(first_error["step"].min()) if not first_error.empty else np.nan,
                "first_sink_step": int(sink_hits["step"].min()) if not sink_hits.empty else np.nan,
                "ever_sink": bool(sink_mask.any()),
                "final_sink": bool(sink_mask.loc[final.name]),
                "final_step": int(final["step"]),
                "final_guess_norm": final["guess_norm"],
                "final_guess_category_exact": final["guess_category_exact"],
                "final_exact_correct": bool(final["exact_correct"]),
                "final_word_to_guess_similarity": final.get("word_to_guess_similarity", np.nan),
                "unique_guesses_across_chain": group["guess_norm"].nunique(),
                "guess_entropy_across_chain": shannon_entropy(group["guess_norm"]),
            }
        )
    return pd.DataFrame(summaries)


def chain_summary_by_model(chain_summary: pd.DataFrame, sink_label: str) -> pd.DataFrame:
    return (
        chain_summary.groupby(["model_name", "pipeline"], observed=True)
        .agg(
            chain_n=("instance_id", "size"),
            ever_sink_rate=("ever_sink", "mean"),
            final_sink_rate=("final_sink", "mean"),
            median_first_sink_step=("first_sink_step", "median"),
            final_exact_accuracy=("final_exact_correct", "mean"),
            mean_final_similarity=("final_word_to_guess_similarity", "mean"),
            mean_unique_guesses=("unique_guesses_across_chain", "mean"),
            mean_guess_entropy=("guess_entropy_across_chain", "mean"),
        )
        .reset_index()
        .assign(sink_label=sink_label)
        .sort_values(["pipeline", "model_name"])
    )


def step_profile(rows: pd.DataFrame, sink_flag_col: str, similarity_col: str | None = None) -> pd.DataFrame:
    agg_kwargs = {
        "chain_n": ("instance_id", "nunique"),
        "exact_accuracy": ("exact_correct", "mean"),
        "sink_guess_rate": (sink_flag_col, "mean"),
        "unique_guesses": ("guess_norm", "nunique"),
        "guess_entropy": ("guess_norm", shannon_entropy),
    }
    if similarity_col is not None and similarity_col in rows.columns:
        agg_kwargs["mean_word_to_guess_similarity"] = (similarity_col, "mean")
    return (
        rows.groupby(["model_name", "pipeline", "step"], observed=True)
        .agg(**agg_kwargs)
        .reset_index()
        .sort_values(["pipeline", "model_name", "step"])
    )


def word_first_source_sink_rates(chain_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_word_rates = (
        chain_summary.groupby(["model_name", "pipeline", "category", "original_word", "word_norm"], observed=True)
        .agg(
            chain_n=("instance_id", "size"),
            ever_sink_rate=("ever_sink", "mean"),
            final_sink_rate=("final_sink", "mean"),
            final_exact_accuracy=("final_exact_correct", "mean"),
            median_first_sink_step=("first_sink_step", "median"),
            mean_unique_guesses=("unique_guesses_across_chain", "mean"),
            mean_guess_entropy=("guess_entropy_across_chain", "mean"),
        )
        .reset_index()
    )
    model_pipeline_rates = (
        source_word_rates.groupby(["model_name", "pipeline"], observed=True)
        .agg(
            source_word_n=("word_norm", "nunique"),
            mean_ever_sink_rate_across_words=("ever_sink_rate", "mean"),
            mean_final_sink_rate_across_words=("final_sink_rate", "mean"),
            mean_final_exact_accuracy_across_words=("final_exact_accuracy", "mean"),
            median_first_sink_step_across_words=("median_first_sink_step", "median"),
            mean_unique_guesses=("mean_unique_guesses", "mean"),
            mean_guess_entropy=("mean_guess_entropy", "mean"),
        )
        .reset_index()
        .sort_values(["pipeline", "model_name"])
    )
    return source_word_rates, model_pipeline_rates


def word_step_sink_rates(rows: pd.DataFrame, sink_flag_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    word_step = (
        rows.groupby(["model_name", "pipeline", "category", "original_word", "word_norm", "step"], observed=True)
        .agg(
            chain_n=("instance_id", "nunique"),
            exact_accuracy=("exact_correct", "mean"),
            sink_guess_rate=(sink_flag_col, "mean"),
            known_vocab_guess_rate=("guess_category_exact", lambda s: s.notna().mean()),
            unique_guesses=("guess_norm", "nunique"),
            guess_entropy=("guess_norm", shannon_entropy),
        )
        .reset_index()
    )
    category_step = (
        word_step.groupby(["model_name", "pipeline", "category", "step"], observed=True)
        .agg(
            source_word_n=("word_norm", "nunique"),
            mean_exact_accuracy_across_words=("exact_accuracy", "mean"),
            mean_sink_guess_rate_across_words=("sink_guess_rate", "mean"),
            mean_known_vocab_guess_rate_across_words=("known_vocab_guess_rate", "mean"),
            mean_unique_guesses_across_words=("unique_guesses", "mean"),
            mean_guess_entropy_across_words=("guess_entropy", "mean"),
        )
        .reset_index()
        .sort_values(["pipeline", "model_name", "step"])
    )
    return word_step, category_step


def exact_sink_rank_by_step(rows: pd.DataFrame, sink_category: str, final_step: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    rows["is_cross_word_guess"] = rows["guess_norm"].ne(rows["word_norm"])
    hits = rows[rows["guess_category_exact"].eq(sink_category) & rows["is_cross_word_guess"]].copy()
    source_step_totals = (
        rows.groupby(["model_name", "pipeline", "step", "original_word", "word_norm"], observed=True)["instance_id"]
        .nunique()
        .reset_index(name="chain_n")
    )
    counts = (
        hits.groupby(["model_name", "pipeline", "step", "original_word", "word_norm", "guess_norm"], observed=True)
        .size()
        .reset_index(name="hit_count")
        .merge(source_step_totals, on=["model_name", "pipeline", "step", "original_word", "word_norm"], how="left")
    )
    counts["source_word_sink_rate"] = counts["hit_count"] / counts["chain_n"]
    ranks = (
        counts.groupby(["model_name", "pipeline", "step", "guess_norm"], observed=True)
        .agg(
            mean_source_word_sink_rate=("source_word_sink_rate", "mean"),
            source_word_n=("word_norm", "nunique"),
            total_hit_count=("hit_count", "sum"),
        )
        .reset_index()
        .sort_values(
            ["pipeline", "model_name", "step", "mean_source_word_sink_rate", "source_word_n", "total_hit_count"],
            ascending=[True, True, True, False, False, False],
        )
    )
    top = ranks.groupby(["model_name", "pipeline", "step"], observed=True).head(20).reset_index(drop=True)
    final = ranks[ranks["step"].eq(final_step)].copy()
    return counts, top, final


def nearest_category_basin_map(
    guesses: list[str],
    sink_vocab: list[str],
    cache_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    texts = sorted(set([g for g in guesses if g] + [w for w in sink_vocab if w]))
    embeddings, missing = load_cached_embeddings(texts, cache_path)
    sink_words = [word for word in sink_vocab if word in embeddings]
    sink_matrix = np.vstack([embeddings[word] for word in sink_words]) if sink_words else np.empty((0, 0))
    rows = []
    for guess in sorted(set(g for g in guesses if g)):
        vec = embeddings.get(guess)
        if vec is None or sink_matrix.size == 0:
            rows.append(
                {
                    "guess_norm": guess,
                    "nearest_sink_word": np.nan,
                    "nearest_sink_similarity": np.nan,
                    "missing_embedding": True,
                }
            )
            continue
        sims = sink_matrix @ vec
        best_idx = int(np.argmax(sims))
        rows.append(
            {
                "guess_norm": guess,
                "nearest_sink_word": sink_words[best_idx],
                "nearest_sink_similarity": float(sims[best_idx]),
                "missing_embedding": False,
            }
        )
    return pd.DataFrame(rows), missing


def basin_rank_by_step(
    rows: pd.DataFrame,
    basin_map: pd.DataFrame,
    threshold: float,
    final_step: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapped = rows.merge(basin_map, on="guess_norm", how="left")
    mapped["is_embedding_basin"] = mapped["nearest_sink_similarity"].ge(threshold).fillna(False)
    source_step_totals = (
        mapped.groupby(["model_name", "pipeline", "step", "original_word", "word_norm"], observed=True)["instance_id"]
        .nunique()
        .reset_index(name="chain_n")
    )
    counts = (
        mapped[mapped["is_embedding_basin"]]
        .groupby(["model_name", "pipeline", "step", "original_word", "word_norm", "nearest_sink_word"], observed=True)
        .agg(
            hit_count=("instance_id", "size"),
            mean_nearest_sink_similarity=("nearest_sink_similarity", "mean"),
        )
        .reset_index()
        .merge(source_step_totals, on=["model_name", "pipeline", "step", "original_word", "word_norm"], how="left")
    )
    counts["source_word_basin_rate"] = counts["hit_count"] / counts["chain_n"]
    ranks = (
        counts.groupby(["model_name", "pipeline", "step", "nearest_sink_word"], observed=True)
        .agg(
            mean_source_word_basin_rate=("source_word_basin_rate", "mean"),
            source_word_n=("word_norm", "nunique"),
            total_hit_count=("hit_count", "sum"),
            mean_nearest_sink_similarity=("mean_nearest_sink_similarity", "mean"),
        )
        .reset_index()
        .sort_values(
            ["pipeline", "model_name", "step", "mean_source_word_basin_rate", "source_word_n", "total_hit_count"],
            ascending=[True, True, True, False, False, False],
        )
    )
    top = ranks.groupby(["model_name", "pipeline", "step"], observed=True).head(20).reset_index(drop=True)
    final = ranks[ranks["step"].eq(final_step)].copy()
    return mapped, counts, top, final


def top_guesses_by_step(rows: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    counts = (
        rows.groupby(["model_name", "pipeline", "step", "guess_norm", "guess_category_exact"], dropna=False, observed=True)
        .size()
        .reset_index(name="count")
    )
    totals = (
        counts.groupby(["model_name", "pipeline", "step"], observed=True)["count"]
        .sum()
        .reset_index(name="total_count")
    )
    counts = counts.merge(totals, on=["model_name", "pipeline", "step"], how="left")
    counts["rate"] = counts["count"] / counts["total_count"]
    counts = counts.sort_values(
        ["pipeline", "model_name", "step", "count", "guess_norm"],
        ascending=[True, True, True, False, True],
    )
    return counts.groupby(["model_name", "pipeline", "step"], observed=True).head(top_n).reset_index(drop=True)


def qualitative_trace(rows: pd.DataFrame, chain_summary: pd.DataFrame) -> pd.DataFrame:
    candidates = chain_summary.sort_values(
        ["final_sink", "ever_sink", "first_sink_step", "model_name", "pipeline", "instance_id"],
        ascending=[False, False, True, True, True, True],
    )
    if candidates.empty:
        return pd.DataFrame()
    example = candidates.iloc[0]
    trace = rows[
        rows["model_name"].astype(str).eq(str(example["model_name"]))
        & rows["pipeline"].eq(example["pipeline"])
        & rows["instance_id"].eq(int(example["instance_id"]))
    ].copy().sort_values("step")
    if "description" in trace.columns:
        trace["description_short"] = (
            trace["description"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 260)
        )
    return trace


def guess_transitions(rows: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    transitions = rows.sort_values(["model_name", "pipeline", "instance_id", "step"]).copy()
    group_cols = ["model_name", "pipeline", "instance_id"]
    transitions["next_guess_norm"] = transitions.groupby(group_cols, observed=True)["guess_norm"].shift(-1)
    transitions["next_step"] = transitions.groupby(group_cols, observed=True)["step"].shift(-1)
    transitions = transitions[transitions["next_step"].eq(transitions["step"] + 1)].copy()
    transitions["transition"] = transitions["guess_norm"] + " -> " + transitions["next_guess_norm"]
    counts = (
        transitions.groupby(["model_name", "pipeline", "step", "transition"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values(["pipeline", "model_name", "step", "count", "transition"], ascending=[True, True, True, False, True])
    )
    return counts.groupby(["model_name", "pipeline", "step"], observed=True).head(top_n).reset_index(drop=True)
