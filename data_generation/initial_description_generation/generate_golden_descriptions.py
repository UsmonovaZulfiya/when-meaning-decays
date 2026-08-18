import os, re, time, json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


# =========================
# CONFIG
# =========================

load_dotenv()  # Load environment variables from .env file if it exists

API_KEY = os.getenv("GPT_OSS_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("GPT_OSS_BASE_URL", "url-here")

MODEL_ID = "gpt-oss:120b"

INPUT_WORDS_CSV = "word_categories_short.csv"
# OUTPUT_CSV = "golden_descriptions_gpt_oss_120b.csv"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_CSV = f"golden_descriptions_gpt_oss_120b_{RUN_TIMESTAMP}.csv"

ERROR_CSV = "golden_descriptions_gpt_oss_120b_errors.csv"

DESCRIPTIONS_PER_WORD = 10

MAX_RETRIES = 5
SLEEP_BETWEEN_CALLS = 0

N_CONCURRENT_REQUESTS = 5

# Set to True if you want to run the validation test for every generated description.
# This doubles the number of API calls, but gives you a useful quality-control column.
RUN_DESCRIPTION_CHECK = False


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# =========================
# PROMPTS
# =========================

DESCRIPTION_SYSTEM_PROMPT = """
You are generating high-quality descriptions for a controlled semantic drift experiment.

Your task is to describe a target English word clearly enough that another language model could guess it.

Rules:
- Write exactly 2 sentences.
- Do NOT mention the target word itself.
- Do NOT use obvious morphological variants of the target word.
- Do NOT use quotation marks.
- The description must be specific enough for guessing the word.
- Avoid being too vague or poetic.
- Avoid listing many possible meanings.
- Use plain English.
- Output only the description, nothing else.
""".strip()


def build_description_prompt(word: str) -> str:
    return f"""
Target word: {word}

Generate one clear, complete, guessable description of this word.
Remember: do not mention the target word itself.
""".strip()


CHECK_SYSTEM_PROMPT = """
You are validating descriptions for a word-guessing experiment.

Given a target word and a description, decide whether the description is specific enough that the target word would be a reasonable guess.

Return only valid JSON with this schema:
{
  "is_good_description": true or false,
  "likely_guess": "single guessed word or short phrase",
  "reason": "brief explanation"
}
""".strip()


def build_check_prompt(word: str, description: str) -> str:
    return f"""
Target word: {word}

Description:
{description}

Question:
Is this description specific enough that the target word is a reasonable answer?
""".strip()


# =========================
# HELPERS
# =========================

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_target_word(description: str, word: str) -> bool:
    """
    Conservative check: catches exact target word occurrence.
    It does not catch all morphological variants, but catches the main leakage case.
    """
    desc = description.lower()
    target = word.lower()

    pattern = r"\b" + re.escape(target) + r"\b"
    return re.search(pattern, desc) is not None


def call_chat_completion(system_prompt: str, user_prompt: str, temperature: float = 0.8) -> str:
    """
    Generic API call with retry logic.
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
            )
            return normalize_text(res.choices[0].message.content)

        except Exception as e:
            last_error = e
            wait_time = min(2 ** attempt, 30)
            print(f"[Retry {attempt}/{MAX_RETRIES}] API error: {e}. Waiting {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(f"API call failed after {MAX_RETRIES} retries. Last error: {last_error}")


def generate_description(word: str) -> str:
    return call_chat_completion(
        system_prompt=DESCRIPTION_SYSTEM_PROMPT,
        user_prompt=build_description_prompt(word),
        temperature=0.8
    )


def check_description(word: str, description: str) -> tuple[bool, str]:
    """
    Returns:
        is_good_description, raw_response
    """
    raw = call_chat_completion(
        system_prompt=CHECK_SYSTEM_PROMPT,
        user_prompt=build_check_prompt(word, description),
        temperature=0.0
    )

    try:
        parsed = json.loads(raw)
        return bool(parsed.get("is_good_description", False)), raw
    except Exception:
        # If parsing fails, keep raw response for later inspection.
        return False, raw


def load_words_long_format(path: str) -> pd.DataFrame:
    """
    Converts your wide word_categories.csv format:

    High-Freq Concrete,Low-Freq Concrete,High-Freq Abstract,Low-Freq Abstract
    Apple,Abacus,Ability,Aberration
    Baby,Accordion,Action,Acrimony

    into long format:

    Category, Word
    High-Freq Concrete, Apple
    Low-Freq Concrete, Abacus
    ...
    """
    df_wide = pd.read_csv(path)

    rows = []
    for category in df_wide.columns:
        for word in df_wide[category].dropna().astype(str):
            word = word.strip()
            if word:
                rows.append({
                    "Category": category,
                    "Word": word
                })

    df_long = pd.DataFrame(rows).drop_duplicates()
    return df_long


def load_existing_output(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=[
        "Category",
        "Word",
        "Instance_ID",
        "Description",
        "Description_Word_Check",
        "Raw_Check_Response"
    ])


def already_done(existing_df: pd.DataFrame, word: str, instance_id: int) -> bool:
    if existing_df.empty:
        return False

    mask = (
        (existing_df["Word"].astype(str).str.lower() == word.lower()) &
        (existing_df["Instance_ID"].astype(int) == int(instance_id))
    )
    return mask.any()


def save_error(category: str, word: str, instance_id: int, stage: str, error: Exception):
    error_row = {
        "Category": category,
        "Word": word,
        "Instance_ID": instance_id,
        "Stage": stage,
        "Error": str(error),
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    temp_df = pd.DataFrame([error_row])
    file_exists = os.path.exists(ERROR_CSV)

    temp_df.to_csv(
        ERROR_CSV,
        mode="a",
        header=not file_exists,
        index=False
    )


def process_one_description(category: str, word: str, instance_id: int) -> dict:
    """
    Generates one golden description and optionally validates it.
    Returns a dictionary with either:
      {"status": "ok", "row": new_row}
    or:
      {"status": "error", ...}
    """

    try:
        description = generate_description(word)

        # If the model accidentally mentions the target word, retry a few times.
        leakage_attempts = 0
        while contains_target_word(description, word) and leakage_attempts < 3:
            leakage_attempts += 1
            print(f"[Leakage detected] Word '{word}' appeared in description. Retrying...")
            description = generate_description(word)

        # If after retries it still contains the word, keep it but mark it.
        leakage_detected = contains_target_word(description, word)

        if RUN_DESCRIPTION_CHECK:
            try:
                is_good, raw_check = check_description(word, description)
            except Exception as e:
                is_good = None
                raw_check = f"CHECK_FAILED: {e}"
        else:
            is_good, raw_check = None, ""

        new_row = {
            "Category": category,
            "Word": word,
            "Instance_ID": instance_id,
            "Description": description,
            "Description_Word_Check": is_good,
            "Raw_Check_Response": raw_check,
            "Leakage_Detected": leakage_detected
        }

        return {
            "status": "ok",
            "row": new_row
        }

    except Exception as e:
        return {
            "status": "error",
            "category": category,
            "word": word,
            "instance_id": instance_id,
            "stage": "generation",
            "error": e
        }


# =========================
# MAIN
# =========================

def main():
    words_df = load_words_long_format(INPUT_WORDS_CSV)
    existing_df = load_existing_output(OUTPUT_CSV)

    print(f"Loaded {len(words_df)} unique words.")
    print(f"Existing generated rows: {len(existing_df)}")

    total_expected = len(words_df) * DESCRIPTIONS_PER_WORD
    print(f"Expected final rows: {total_expected}")
    print(f"Concurrent requests: {N_CONCURRENT_REQUESTS}")

    tasks = []

    for _, row in words_df.iterrows():
        category = row["Category"]
        word = row["Word"]

        for instance_id in range(1, DESCRIPTIONS_PER_WORD + 1):
            if already_done(existing_df, word, instance_id):
                continue

            tasks.append((category, word, instance_id))

    print(f"Remaining tasks to generate: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_task = {
            executor.submit(process_one_description, category, word, instance_id): (category, word, instance_id)
            for category, word, instance_id in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Generating descriptions"):
            category, word, instance_id = future_to_task[future]

            try:
                result = future.result()
            except Exception as e:
                print(f"[UNEXPECTED ERROR] Word='{word}', Instance={instance_id}. Error: {e}")
                save_error(category, word, instance_id, stage="unexpected", error=e)
                continue

            if result["status"] == "ok":
                new_row = result["row"]

                temp_df = pd.DataFrame([new_row])
                file_exists = os.path.exists(OUTPUT_CSV)

                temp_df.to_csv(
                    OUTPUT_CSV,
                    mode="a",
                    header=not file_exists,
                    index=False
                )

                existing_df = pd.concat([existing_df, temp_df], ignore_index=True)

            else:
                print(
                    f"[FAILED] Word='{result['word']}', "
                    f"Instance={result['instance_id']}. Error: {result['error']}"
                )

                save_error(
                    result["category"],
                    result["word"],
                    result["instance_id"],
                    stage=result["stage"],
                    error=result["error"]
                )

            if SLEEP_BETWEEN_CALLS > 0:
                time.sleep(SLEEP_BETWEEN_CALLS)

    print("Generation completed.")

    if os.path.exists(OUTPUT_CSV):
        final_df = pd.read_csv(OUTPUT_CSV)
        print(f"Final rows saved: {len(final_df)}")
        print(f"Saved to: {OUTPUT_CSV}")
    else:
        print("No output file was created.")


if __name__ == "__main__":
    main()