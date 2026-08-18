"""Generate the compact step-10 overview table.

Reads the latest combined result CSV from
data_analysis/results_analysis/all_exact_entropy_guess_similarity/tables and
writes two new overview files in data_analysis/.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = ROOT.parent
INPUT_CSV = (
    ANALYSIS_ROOT
    / "results_analysis"
    / "all_exact_entropy_guess_similarity"
    / "tables"
    / "overview_metrics_all_selected_pipelines.csv"
)
OUTPUT_CSV = ANALYSIS_ROOT / "step10_overview_table.csv"
OUTPUT_MD = ANALYSIS_ROOT / "step10_overview_table.md"

METRICS = [
    ("Step-10 exact accuracy", "exact_accuracy"),
    ("Step-10 semantic similarity", "word_guess_semantic_similarity"),
    ("Step-10 normalized entropy", "normalized_entropy"),
]

CONDITIONS = [
    (
        "Pipeline A, all models avg.",
        lambda row: row["pipeline"] == "pipeline_a"
        and row["category"] == "All Categories",
    ),
    (
        "Pipeline B, all models avg.",
        lambda row: row["pipeline"] == "pipeline_b"
        and row["category"] == "All Categories",
    ),
    ("HFC", lambda row: row["category"] == "High-Freq-Concrete"),
    ("HFA", lambda row: row["category"] == "High-Freq-Abstract"),
    ("LFC", lambda row: row["category"] == "Low-Freq-Concrete"),
    ("LFA", lambda row: row["category"] == "Low-Freq-Abstract"),
]


def load_step10_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["step"] == "10"]


def mean_metric(rows: list[dict[str, str]], metric: str) -> float:
    return fmean(float(row[metric]) for row in rows)


def build_overview(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    overview: list[dict[str, str]] = []

    for label, predicate in CONDITIONS:
        selected = [row for row in rows if predicate(row)]
        if not selected:
            raise ValueError(f"No rows matched condition: {label}")

        out_row = {"Condition": label}
        for heading, metric in METRICS:
            out_row[heading] = f"{mean_metric(selected, metric):.6f}"
        overview.append(out_row)

    return overview


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = ["Condition", *[heading for heading, _ in METRICS]]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]]) -> None:
    fieldnames = ["Condition", *[heading for heading, _ in METRICS]]
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]

    for row in rows:
        values = [row["Condition"]]
        values.extend(f"{float(row[heading]):.3f}" for heading, _ in METRICS)
        lines.append("| " + " | ".join(values) + " |")

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_overview(load_step10_rows())
    write_csv(rows)
    write_markdown(rows)
    print(OUTPUT_MD)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
