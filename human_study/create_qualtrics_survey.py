#!/usr/bin/env python3
import csv
from pathlib import Path
import argparse
import html


REQUIRED_COLUMNS = {
    "Guessed word",
    "Guessed description",
    "Type",
}

AGREEMENT_STATEMENT = """[[Question:MC:SingleAnswer]]
[[ID:AGREEMENT]]
You are invited to take part in a short language study.<br />
<br />
In this study, you will complete two types of word tasks.<br />
<br />
In one task, you will see an English word.<br />
You will write a two-sentence description of that word.<br />
The description should help another person guess the word.<br />
<br />
In another task, you will see a description of an English word.<br />
You will guess which single English word the description refers to.<br />
<br />
USE OF AI IS NOT ALLOWED.
<br />
Your participation is voluntary and your responses will be stored and used for research purposes.<br />
<br />
Do you agree to participate?

[[Choices]]
Yes, I agree

No, I do not agree

[[PageBreak]]"""

TYPE0_PROMPT = """Describe the word below in two sentences in a way that another person could guess it. &nbsp;<br />"""

TYPE1_PROMPT = """Guess the single most likely English word that matches the description below. &nbsp;<br />"""


TYPE0_INSTRUCTIONS = """[[Question:DB]]
[[ID:TYPE0_INSTRUCTIONS]]
You will receive one English word.<br />
Your first task is to describe a this English word clearly enough so that another person could guess it.<br />
<br />
<strong>Rules:</strong><br />
- Write exactly 2 sentences.<br />
- Do NOT mention the target word itself.<br />
- Do NOT use obvious morphological variants of the target word.<br />
- Do NOT use AI (for example, ChatGPT) to generate this description.<br />
- You are allowed to search for the meaning of a word, but the description should be written in your own words.<br />
- The description must be specific enough for guessing the word.<br />
<br />
<strong>For example:</strong><br />
<strong>Given word:</strong> Apple<br />
<br />
<strong>Valid description:</strong> It is a round, edible fruit with smooth skin that comes in shades of red, green, or yellow and contains a single core of small seeds. The same name is used by a major technology corporation whose logo features a stylized version of this fruit."""


TYPE1_INSTRUCTIONS = """[[Question:DB]]
[[ID:TYPE1_INSTRUCTIONS]]
You will receive a description of one English word.<br />
Your task is to guess the single most likely English word from this description.<br />
<br />
<strong>Rules:</strong><br />
- Write only one word.<br />
- Do not explain your answer.<br />
- Do not use punctuation.<br />
- If unsure, still provide your best single-word guess.<br />
<br />
<strong>For example:</strong><br />
<strong>Given description:</strong> It is a round, typically red or green fruit that grows on trees and is often eaten raw, baked in pies, or pressed into cider. The same name is also used by a well-known technology corporation that designs smartphones, tablets, and computers.<br />
<br />
<strong>Guess word:</strong> Apple"""


def clean(value: str) -> str:
    """Clean cell values for Qualtrics TXT import."""
    if value is None:
        return ""
    value = str(value).strip()
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value


def qhtml(value: str) -> str:
    """
    Escape CSV values before inserting them into HTML-ish Qualtrics text.
    This prevents characters like &, <, and > from breaking the import.
    """
    return html.escape(clean(value), quote=False)


def safe_tag(value: str) -> str:
    """
    Make a value safe for a Qualtrics question ID/export tag.
    Keeps only letters, digits, and underscores.
    """
    cleaned = []
    for char in value:
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


def make_type0_question(index: int, word: str) -> str:
    """
    Type 0:
    Show the word.
    Ask participant to describe it.
    Expected answer: block / essay text.
    """
    qid = f"T0_Q{index:03d}"
    word = qhtml(word)

    return f"""[[Question:TE:Essay]]
[[ID:{qid}]]
{TYPE0_PROMPT}  

<br />

Word: <u><strong>{word}</strong></u>"""


def make_type1_question(index: int, description: str) -> str:
    """
    Type 1:
    Show the description.
    Ask participant to guess the word.
    Expected answer: single word / single-line text.
    """
    qid = f"T1_Q{index:03d}"
    description = qhtml(description)

    return f"""[[Question:TE:SingleLine]]
[[ID:{qid}]]
{TYPE1_PROMPT}

<br />

Description: <u><strong>{description}</strong></u>"""


def generate_qualtrics_txt(input_csv: str, output_txt: str) -> None:
    input_csv = Path(input_csv)
    output_txt = Path(output_txt)

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])

        missing = REQUIRED_COLUMNS - headers
        if missing:
            raise ValueError(
                f"Missing required column(s): {', '.join(sorted(missing))}"
            )

        rows = list(reader)

    output_parts = ["[[AdvancedFormat]]"]

    type0_description = ["[[Block:Type 0 - Describe Words]]"]

    type0_description.append(TYPE0_INSTRUCTIONS)
    type0_description.append("[[PageBreak]]")

    type1_description = ["[[Block:Type 1 - Guess Words from Descriptions]]"]
    
    type1_description.append(TYPE1_INSTRUCTIONS)
    type1_description.append("[[PageBreak]]")



    type0_parts = ["[[Block:Type 0 - Describe Words]]"]
    type1_parts = ["[[Block:Type 1 - Guess Words from Descriptions]]"]

    # Add instruction pages at the beginning of each block.

    type0_count = 0
    type1_count = 0

    for row_number, row in enumerate(rows, start=2):
        word = clean(row["Guessed word"])
        description = clean(row["Guessed description"])
        row_type = clean(row["Type"])

        if row_type == "0":
            if not word:
                raise ValueError(
                    f"Row {row_number}: Type is 0 but Word_Gen_1 is empty."
                )

            type0_count += 1
            type0_parts.append(
                make_type0_question(
                    index=type0_count,
                    word=word,
                )
            )
            type0_parts.append("[[PageBreak]]")

        elif row_type == "1":
            if not description:
                raise ValueError(
                    f"Row {row_number}: Type is 1 but Description_Gen_1 is empty."
                )

            type1_count += 1
            type1_parts.append(
                make_type1_question(
                    index=type1_count,
                    description=description,
                )
            )
            type1_parts.append("[[PageBreak]]")

        else:
            raise ValueError(
                f"Row {row_number}: Invalid Type value '{row_type}'. "
                "Expected 0 or 1."
            )

    #output_parts.append(AGREEMENT_STATEMENT)
    #output_parts.extend(type0_description)
    output_parts.extend(type0_parts)
    #output_parts.extend(type1_description)
    output_parts.extend(type1_parts)

    # Qualtrics expects questions separated by blank lines.
    txt_content = "\n\n".join(output_parts).strip() + "\n"

    output_txt.write_text(txt_content, encoding="utf-8")

    print(f"Created Qualtrics import file: {output_txt}")
    print(f"Type 0 essay questions: {type0_count}")
    print(f"Type 1 single-line questions: {type1_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Create Qualtrics Advanced TXT surveys from all CSV files in a folder."
    )

    parser.add_argument(
        "input_folder",
        nargs="?",
        default=".",
        help="Folder containing input CSV files. Default: current folder.",
    )

    args = parser.parse_args()

    input_folder = Path(args.input_folder)

    for input_csv in sorted(input_folder.glob("*.csv")):
        output_txt = input_folder / f"qualtrics_import_{input_csv.stem}.txt"

        generate_qualtrics_txt(
            str(input_csv),
            str(output_txt)
        )

        print(f"Created: {output_txt}")

if __name__ == "__main__":
    main()