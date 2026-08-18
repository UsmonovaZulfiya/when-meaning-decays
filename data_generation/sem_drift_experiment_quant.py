import os
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from models_local_quant import BACKEND_LABEL, LocalChatModel


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# The default server root can be overridden through the SLURM configuration.
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", "/mnt/h200_raid5/zulfiyausmonova")
)

# The golden-description path can be overridden with GOLDEN_DESCRIPTIONS_FILE.
DEFAULT_GOLDEN_FILE = (
    DEFAULT_PROJECT_ROOT
    / "outputs"
    / "semantic_drift_from_golden"
    / "golden_descriptions_gpt_oss.csv"
)

DEFAULT_HF_CACHE = DEFAULT_PROJECT_ROOT / "hf_cache"


# ============================================================
# CONFIG FROM ENVIRONMENT VARIABLES
# ============================================================

PIPELINE = os.environ.get("PIPELINE", "a").strip().lower()
if PIPELINE not in {"a", "b"}:
    raise ValueError("PIPELINE must be either 'a' or 'b'.")

PROMPT_MODE = (
    "pipeline_A_guess_then_describe_guessed_word"
    if PIPELINE == "a"
    else "pipeline_B_paraphrase_description"
)
GENERATED_SOURCE = (
    "local_model_generated_from_guess"
    if PIPELINE == "a"
    else "local_model_paraphrased_description"
)

MODEL_KEY = os.environ.get("MODEL_KEY", "llama-3.1-8b-instruct").strip()

LOCAL_MODELS = {
    "llama-3.1-8b-instruct": str(DEFAULT_PROJECT_ROOT / "models" / "llama-3.1-8b-instruct"),
    "gemma-3-4b-it": str(DEFAULT_PROJECT_ROOT / "models" / "gemma-3-4b-it"),
    "gemma-3-12b-it": str(DEFAULT_PROJECT_ROOT / "models" / "gemma-3-12b-it"),
    "gemma-3-27b-it": str(DEFAULT_PROJECT_ROOT / "models" / "gemma-3-27b-it"),
}

if MODEL_KEY not in LOCAL_MODELS:
    raise ValueError(
        f"Unknown MODEL_KEY={MODEL_KEY!r}. Available keys: {list(LOCAL_MODELS.keys())}"
    )

MODEL_PATH = Path(os.environ.get("MODEL_PATH", LOCAL_MODELS[MODEL_KEY]))

MODEL_LABEL = os.environ.get("MODEL_LABEL", MODEL_KEY).strip()

GOLDEN_DESCRIPTIONS_FILE = Path(
    os.environ.get("GOLDEN_DESCRIPTIONS_FILE", str(DEFAULT_GOLDEN_FILE))
)

OUTPUT_ROOT = Path(
    os.environ.get(
        "OUTPUT_ROOT",
        str(
            DEFAULT_PROJECT_ROOT
            / "outputs"
            / "semantic_drift_from_golden"
            / ("pipelinea" if PIPELINE == "a" else "")
        ),
    )
)

# Output files are stored in the following format:
# /mnt/h200_raid5/zulfiyausmonova/outputs/semantic_drift_from_golden/model_name/*.csv
MODEL_OUTPUT_DIRNAME = re.sub(r"[^A-Za-z0-9_\-]+", "", MODEL_LABEL.replace(" ", "_"))

OUTPUT_BASE = Path(
    os.environ.get("OUTPUT_BASE", str(OUTPUT_ROOT / MODEL_OUTPUT_DIRNAME))
)

NUM_STEPS = int(os.environ.get("NUM_STEPS", "10"))
NUM_INSTANCES = int(os.environ.get("NUM_INSTANCES", "100"))

MAX_WORDS_PER_CATEGORY = int(os.environ.get("MAX_WORDS_PER_CATEGORY", "0"))
WORD_OFFSET_PER_CATEGORY = int(os.environ.get("WORD_OFFSET_PER_CATEGORY", "0"))

ONLY_CATEGORIES = os.environ.get("ONLY_CATEGORIES", "").strip()

GUESS_TEMPERATURE = float(os.environ.get("GUESS_TEMPERATURE", "0.0"))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.8"))

GUESS_MAX_TOKENS = int(os.environ.get("GUESS_MAX_TOKENS", "16"))
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", "140"))

# API concurrency is replaced by local batching. If memory errors occur, the
# batch size should be reduced to 4, 2, or 1.
LOCAL_BATCH_SIZE = int(os.environ.get("LOCAL_BATCH_SIZE", "8"))

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

global_model: LocalChatModel | None = None

OUTPUT_COLUMNS = [
    "model_name",
    "model_id",
    "category",
    "original_word",
    "instance_id",
    "step",
    "description",
    "guess",
    "source",
    "prompt_mode",
    "guess_temperature",
    "generation_temperature",
]

LEGACY_OUTPUT_COLUMN_MAP = {
    "Model": "model_name",
    "Model_ID": "model_id",
    "Category": "category",
    "Word": "original_word",
    "Instance_ID": "instance_id",
    "Step": "step",
    "Description": "description",
    "Guess": "guess",
    "Source": "source",
    "Prompt_Mode": "prompt_mode",
    "Guess_Temperature": "guess_temperature",
    "Generation_Temperature": "generation_temperature",
}


# ============================================================
# HELPERS
# ============================================================

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "", str(s).strip().replace(" ", "_"))


def clean_guess(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = text.split("\n")[0].strip()
    text = text.strip().strip(".:,;!?\"'“”‘’`")
    text = re.sub(
        r"^(the answer is|answer:|guess:)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip().strip(".:,;!?\"'“”‘’`")
    text = text.lower()

    return text


# ============================================================
# DATA LOADING
# ============================================================

def load_golden_descriptions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Golden descriptions file not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"Category", "Word", "Instance_ID", "Description"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Golden descriptions file is missing columns: {missing_cols}")

    df = df.copy()
    df["Word"] = df["Word"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Instance_ID"] = df["Instance_ID"].astype(int)
    df["Description"] = df["Description"].astype(str).str.strip()

    df = df[df["Word"] != ""]
    df = df[df["Description"] != ""]

    df = (
        df.sort_values(["Category", "Word", "Instance_ID"])
        .groupby(["Category", "Word"], as_index=False)
        .head(NUM_INSTANCES)
        .reset_index(drop=True)
    )

    return df


def normalize_checkpoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy checkpoints are normalized to the current lowercase output schema."""
    rename_map = {
        old_name: new_name
        for old_name, new_name in LEGACY_OUTPUT_COLUMN_MAP.items()
        if old_name in df.columns and new_name not in df.columns
    }
    normalized_df = df.rename(columns=rename_map).drop(
        columns=["Provider", "Base_URL", "provider", "base_url"],
        errors="ignore",
    )

    missing_columns = set(OUTPUT_COLUMNS) - set(normalized_df.columns)
    if missing_columns:
        raise ValueError(
            f"Checkpoint is missing required columns: {sorted(missing_columns)}"
        )

    return normalized_df[OUTPUT_COLUMNS].copy()


def get_completed_word_category_pairs_from_checkpoint(checkpoint_path: Path) -> set[tuple[str, str]]:
    """
    A category-word pair is treated as completed if it has at least one row at Step == NUM_STEPS.
    This is safer than checking only Word, because the same word could theoretically appear in multiple categories.
    """
    if not checkpoint_path.exists():
        return set()

    try:
        df = normalize_checkpoint_columns(pd.read_csv(checkpoint_path))

        completed_df = df[df["step"] == NUM_STEPS][
            ["category", "original_word"]
        ].dropna()

        return {
            (str(row["category"]), str(row["original_word"]))
            for _, row in completed_df.iterrows()
        }

    except Exception:
        return set()


# ============================================================
# PROMPTS: PIPELINES A AND B
# ============================================================

def build_guess_prompt(description: str) -> str:
    return f"""
Read this description:

\"{description}\"

Guess the single English word being described.

Rules:
- Reply with ONLY the word.
- Do not explain.
- Do not use punctuation.
- If unsure, provide your best single-word guess.
""".strip()


def build_description_from_guess_prompt(guess: str) -> str:
    return f"""
Target word: {guess}

Generate a new description of this word.

Rules:
- Write exactly 2 sentences.
- Do NOT mention the target word itself.
- Do NOT use obvious morphological variants of the target word.
- The description must be complete enough that another model could guess the word.
- Use plain English.
- Output only the description.
""".strip()


def build_paraphrase_prompt(description: str) -> str:
    return f"""
Read this description of a word:

\"{description}\"

Paraphrase it.

Rules:
- Write exactly 2 sentences.
- Do NOT make the description exactly the same.
- Do NOT mention the original word if it is obvious from the description.
- Use plain English.
- Output only the paraphrased description.
""".strip()


def process_step_local(current_descriptions: list[str]) -> tuple[list[str], list[str]]:
    """
    One semantic-drift step. Pipeline A regenerates from the guessed word;
    Pipeline B paraphrases the current description. Both retain the guess.
    """
    if global_model is None:
        raise RuntimeError("global_model is not initialized.")

    guess_prompts = [
        build_guess_prompt(desc)
        for desc in current_descriptions
    ]

    raw_guess_responses = global_model.invoke_batch(
        guess_prompts,
        temperature=GUESS_TEMPERATURE,
        max_new_tokens=GUESS_MAX_TOKENS,
    )

    guesses = [clean_guess(resp.content) for resp in raw_guess_responses]
    if PIPELINE == "a":
        safe_guesses = [g if g else "unknown concept" for g in guesses]
        regen_prompts = [
            build_description_from_guess_prompt(guess)
            for guess in safe_guesses
        ]
    else:
        regen_prompts = [
            build_paraphrase_prompt(description)
            for description in current_descriptions
        ]

    regen_responses = global_model.invoke_batch(
        regen_prompts,
        temperature=GEN_TEMPERATURE,
        max_new_tokens=GEN_MAX_TOKENS,
    )

    next_descriptions = [
        str(resp.content).strip()
        for resp in regen_responses
    ]

    return guesses, next_descriptions


# ============================================================
# EXPERIMENT
# ============================================================

def run_category_experiment(
    category_name: str,
    category_df: pd.DataFrame,
    checkpoint_path: Path,
) -> pd.DataFrame:

    records = []

    completed_pairs = get_completed_word_category_pairs_from_checkpoint(checkpoint_path)

    if checkpoint_path.exists():
        old_df = normalize_checkpoint_columns(pd.read_csv(checkpoint_path))
        records = old_df.to_dict("records")
        old_df.to_csv(checkpoint_path, index=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Completed category-word pairs found: {len(completed_pairs)}")

    words = category_df["Word"].dropna().astype(str).unique().tolist()

    for word in tqdm(words, desc=f"{MODEL_LABEL} | {category_name}"):

        pair_key = (category_name, word)

        if pair_key in completed_pairs:
            continue

        word_df = (
            category_df[category_df["Word"] == word]
            .sort_values("Instance_ID")
            .head(NUM_INSTANCES)
        )

        if len(word_df) == 0:
            continue

        current_descs = word_df["Description"].tolist()
        instance_ids = word_df["Instance_ID"].astype(int).tolist()

        # Step 0 is populated with GPT-OSS golden descriptions.
        for instance_id, desc in zip(instance_ids, current_descs):
            records.append({
                "model_name": MODEL_LABEL,
                "model_id": str(MODEL_PATH),
                "category": category_name,
                "original_word": word,
                "instance_id": instance_id,
                "step": 0,
                "description": desc,
                "guess": word,
                "source": "gpt_oss_golden_description",
                "prompt_mode": PROMPT_MODE,
                "guess_temperature": GUESS_TEMPERATURE,
                "generation_temperature": GEN_TEMPERATURE,
            })

        # Steps 1 through NUM_STEPS are populated with model-generated descriptions.
        for step_num in range(1, NUM_STEPS + 1):
            guesses, next_descs = process_step_local(current_descs)

            for instance_id, guess, desc in zip(instance_ids, guesses, next_descs):
                records.append({
                    "model_name": MODEL_LABEL,
                    "model_id": str(MODEL_PATH),
                    "category": category_name,
                    "original_word": word,
                    "instance_id": instance_id,
                    "step": step_num,
                    "description": desc,
                    "guess": guess,
                    "source": GENERATED_SOURCE,
                    "prompt_mode": PROMPT_MODE,
                    "guess_temperature": GUESS_TEMPERATURE,
                    "generation_temperature": GEN_TEMPERATURE,
                })

            current_descs = next_descs

        # A global checkpoint is saved after each completed word.
        pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def select_categories(df: pd.DataFrame) -> list[str]:
    categories = df["Category"].dropna().astype(str).unique().tolist()

    if ONLY_CATEGORIES:
        allowed = {c.strip() for c in ONLY_CATEGORIES.split(",") if c.strip()}
        categories = [c for c in categories if c in allowed]

    return categories


def apply_word_limits(category_df: pd.DataFrame) -> pd.DataFrame:
    unique_words = (
        category_df["Word"]
        .drop_duplicates()
        .tolist()
    )

    if WORD_OFFSET_PER_CATEGORY > 0:
        unique_words = unique_words[WORD_OFFSET_PER_CATEGORY:]

    if MAX_WORDS_PER_CATEGORY > 0:
        unique_words = unique_words[:MAX_WORDS_PER_CATEGORY]

    return category_df[category_df["Word"].isin(unique_words)].copy()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print(f"Local {BACKEND_LABEL} Semantic Drift Run - Pipeline {PIPELINE.upper()}")
    print("==============================")
    print(f"SCRIPT_DIR: {SCRIPT_DIR}")
    print(f"GOLDEN_DESCRIPTIONS_FILE: {GOLDEN_DESCRIPTIONS_FILE}")
    print(f"OUTPUT_BASE: {OUTPUT_BASE}")
    print(f"MODEL_KEY: {MODEL_KEY}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"MODEL_LABEL: {MODEL_LABEL}")
    print(f"PIPELINE: {PIPELINE}")
    print(f"NUM_STEPS: {NUM_STEPS}")
    print(f"NUM_INSTANCES: {NUM_INSTANCES}")
    print(f"MAX_WORDS_PER_CATEGORY: {MAX_WORDS_PER_CATEGORY}")
    print(f"WORD_OFFSET_PER_CATEGORY: {WORD_OFFSET_PER_CATEGORY}")
    print(f"ONLY_CATEGORIES: {ONLY_CATEGORIES}")
    print(f"GUESS_TEMPERATURE: {GUESS_TEMPERATURE}")
    print(f"GEN_TEMPERATURE: {GEN_TEMPERATURE}")
    print(f"GUESS_MAX_TOKENS: {GUESS_MAX_TOKENS}")
    print(f"GEN_MAX_TOKENS: {GEN_MAX_TOKENS}")
    print(f"LOCAL_BATCH_SIZE: {LOCAL_BATCH_SIZE}")
    print("==============================\n")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {MODEL_PATH}")

    golden_df = load_golden_descriptions(GOLDEN_DESCRIPTIONS_FILE)

    print(f"Loaded golden rows: {len(golden_df)}")
    print(f"Unique words: {golden_df['Word'].nunique()}")
    print(f"Categories: {golden_df['Category'].unique().tolist()}")

    print(f"\nLoading local {BACKEND_LABEL} model...")
    global_model = LocalChatModel(
        model_path=str(MODEL_PATH),
        max_batch_size=LOCAL_BATCH_SIZE,
        default_temperature=GEN_TEMPERATURE,
        default_max_new_tokens=GEN_MAX_TOKENS,
    )
    print("Model loaded.\n")

    categories = select_categories(golden_df)

    filename_prefix = "from_golden" if PIPELINE == "a" else "from_golden_exp2"
    checkpoint_path = OUTPUT_BASE / (
        f"checkpoint_{filename_prefix}_{sanitize(MODEL_LABEL)}_ALL_CATEGORIES.csv"
    )

    final_save_path = OUTPUT_BASE / (
        f"Results_{filename_prefix}_{sanitize(MODEL_LABEL)}_ALL_CATEGORIES_{RUN_TIMESTAMP}.csv"
    )

    for category in categories:
        category_df = golden_df[golden_df["Category"] == category].copy()
        category_df = apply_word_limits(category_df)

        print(
            f"\nCategory: {category} | "
            f"Words: {category_df['Word'].nunique()} | "
            f"Rows: {len(category_df)}"
        )

        if len(category_df) == 0:
            print(f"Skipping empty category: {category}")
            continue

        run_category_experiment(
            category_name=category,
            category_df=category_df,
            checkpoint_path=checkpoint_path,
        )

    # The checkpoint is copied to a timestamped result file after all categories finish.
    if checkpoint_path.exists():
        final_df = normalize_checkpoint_columns(pd.read_csv(checkpoint_path))
        final_df.to_csv(checkpoint_path, index=False)
        final_df.to_csv(final_save_path, index=False)

        print("\nDone.")
        print(f"Global checkpoint file: {checkpoint_path}")
        print(f"Final result file: {final_save_path}")
        print(f"Final rows: {len(final_df)}")
    else:
        print("\nNo checkpoint/result file was created.")
