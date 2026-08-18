import os, re, time, json, string
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


# =========================
# CONFIG
# =========================

load_dotenv()

API_KEY = os.getenv("GPT_OSS_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("GPT_OSS_BASE_URL", "url-here")

MODEL_ID = "gpt-oss:120b"

# Put your generated descriptions file here
INPUT_DESCRIPTIONS_CSV = "/mnt/h200_raid5/zulfiyausmonova/outputs/semantic_drift/golden_descriptions_gpt_oss.csv"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_CSV = f"/mnt/h200_raid5/zulfiyausmonova/outputs/semantic_drift/golden_descriptions_guess_validation_{RUN_TIMESTAMP}.csv"
SUMMARY_CSV = f"/mnt/h200_raid5/zulfiyausmonova/outputs/semantic_drift/golden_descriptions_guess_validation_summary_{RUN_TIMESTAMP}.csv"
ERROR_CSV = f"/mnt/h200_raid5/zulfiyausmonova/logs/golden_descriptions_validation_errors_{RUN_TIMESTAMP}.csv"

MAX_RETRIES = 5
SLEEP_BETWEEN_CALLS = 0

N_CONCURRENT_REQUESTS = 5


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# =========================
# PROMPTS
# =========================

GUESS_SYSTEM_PROMPT = """

You will receive a description of one English word.
Your task is to guess the single most likely target word from this description.

Rules:
- Output only one word.
- Do not explain your answer.
- Do not use punctuation.
- Do not output a sentence.
- If unsure, still provide your best single-word guess.

""".strip()


def build_guess_prompt(description: str) -> str:
    return f"""
Description:
{description}

What is the single English word being described?
""".strip()


# =========================
# HELPERS
# =========================

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_match(text: str) -> str:
    """
    Normalizes model guess and target word for strict-ish matching.
    This removes punctuation, lowercases, and strips common leading articles.
    """
    text = normalize_text(text).lower()

    # Remove wrapping quotes
    text = text.strip("\"'“”‘’`")

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove common leading articles if model says "an apple"
    text = re.sub(r"^(a|an|the)\s+", "", text)

    # Keep only first line and trim again
    text = text.split("\n")[0].strip()

    return text


def call_chat_completion(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
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


def guess_word_from_description(description: str) -> str:
    return call_chat_completion(
        system_prompt=GUESS_SYSTEM_PROMPT,
        user_prompt=build_guess_prompt(description),
        temperature=0.0
    )


def save_error(category, word, instance_id, stage, error):
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


def load_existing_output(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame(columns=[
        "Category",
        "Word",
        "Instance_ID",
        "Description",
        "GPT_OSS_Guess",
        "Normalized_Word",
        "Normalized_Guess",
        "Guess_Correct",
        "Raw_Guess_Response"
    ])


def already_done(existing_df: pd.DataFrame, word: str, instance_id: int) -> bool:
    if existing_df.empty:
        return False

    mask = (
        (existing_df["Word"].astype(str).str.lower() == str(word).lower()) &
        (existing_df["Instance_ID"].astype(int) == int(instance_id))
    )
    return mask.any()


def process_one_validation(row: dict) -> dict:
    category = row["Category"]
    word = row["Word"]
    instance_id = int(row["Instance_ID"])
    description = row["Description"]

    try:
        raw_guess = guess_word_from_description(description)

        normalized_word = normalize_for_match(word)
        normalized_guess = normalize_for_match(raw_guess)

        is_correct = normalized_guess == normalized_word

        new_row = {
            "Category": category,
            "Word": word,
            "Instance_ID": instance_id,
            "Description": description,
            "GPT_OSS_Guess": normalized_guess,
            "Normalized_Word": normalized_word,
            "Normalized_Guess": normalized_guess,
            "Guess_Correct": is_correct,
            "Raw_Guess_Response": raw_guess
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
            "stage": "guess_validation",
            "error": e
        }


def save_summary(output_csv: str, summary_csv: str):
    df = pd.read_csv(output_csv)

    total = len(df)
    correct = int(df["Guess_Correct"].sum())
    overall_accuracy = correct / total if total > 0 else 0

    print("\n=========================")
    print("VALIDATION SUMMARY")
    print("=========================")
    print(f"Total validated descriptions: {total}")
    print(f"Correct guesses: {correct}")
    print(f"Overall accuracy: {overall_accuracy:.4f} = {overall_accuracy * 100:.2f}%")

    by_category = (
        df.groupby("Category")["Guess_Correct"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={
            "count": "Total",
            "sum": "Correct",
            "mean": "Accuracy"
        })
    )

    by_category["Accuracy_Percent"] = by_category["Accuracy"] * 100

    print("\nAccuracy by category:")
    print(by_category)

    by_word = (
        df.groupby(["Category", "Word"])["Guess_Correct"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={
            "count": "Total",
            "sum": "Correct",
            "mean": "Accuracy"
        })
    )

    by_word["Accuracy_Percent"] = by_word["Accuracy"] * 100

    # Save both category and word-level summaries in one CSV with a type column
    category_summary = by_category.copy()
    category_summary.insert(0, "Summary_Level", "category")
    category_summary.insert(2, "Word", "")

    word_summary = by_word.copy()
    word_summary.insert(0, "Summary_Level", "word")

    final_summary = pd.concat([category_summary, word_summary], ignore_index=True)
    final_summary.to_csv(summary_csv, index=False)

    print(f"\nSummary saved to: {summary_csv}")


# =========================
# MAIN
# =========================

def main():
    input_df = pd.read_csv(INPUT_DESCRIPTIONS_CSV)
    existing_df = load_existing_output(OUTPUT_CSV)

    required_cols = {"Category", "Word", "Instance_ID", "Description"}
    missing_cols = required_cols - set(input_df.columns)

    if missing_cols:
        raise ValueError(f"Input file is missing columns: {missing_cols}")

    print(f"Loaded descriptions: {len(input_df)}")
    print(f"Existing validated rows: {len(existing_df)}")
    print(f"Concurrent requests: {N_CONCURRENT_REQUESTS}")

    tasks = []

    for _, row in input_df.iterrows():
        word = row["Word"]
        instance_id = int(row["Instance_ID"])

        if already_done(existing_df, word, instance_id):
            continue

        tasks.append(row.to_dict())

    print(f"Remaining descriptions to validate: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_task = {
            executor.submit(process_one_validation, row): row
            for row in tasks
        }

        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Validating descriptions by guessing"
        ):
            original_row = future_to_task[future]

            category = original_row["Category"]
            word = original_row["Word"]
            instance_id = int(original_row["Instance_ID"])

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

    print("\nValidation completed.")

    if os.path.exists(OUTPUT_CSV):
        print(f"Validated rows saved to: {OUTPUT_CSV}")
        save_summary(OUTPUT_CSV, SUMMARY_CSV)
    else:
        print("No validation output file was created.")


if __name__ == "__main__":
    main()
