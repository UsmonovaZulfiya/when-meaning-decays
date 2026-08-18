# Data analysis

Analysis code is grouped by purpose while all generated artifacts continue to
use the shared `data_analysis/results_analysis` directory.

```text
data_analysis/
|-- basic_evaluation_set/
|-- predicting_semantic_stability/
|-- deeper_evaluation/
|-- requirements.txt
`-- results_analysis/                 # created when analyses run
```

- `basic_evaluation_set` contains exact accuracy, word- and description-level
  semantic similarity, Shannon entropy, robustness tables, and primary plots.
- `predicting_semantic_stability` contains regression/model-search work.
- `deeper_evaluation` contains semantic sinks, ablations, shuffled calibration,
  WordNet recovery, and chain extraction.

The main input is `datasets/unified_semantic_drift_results.csv`. Regression and
semantic-sink analyses also reference `golden_descriptions_gpt_oss.csv`,
`word_categories_latest.xlsx`, `words_with_usage_count.csv`, and
`words_weird.csv` in `datasets`.

## Typical workflow

```bash
python -m pip install -r data_analysis/requirements.txt
python data_analysis/basic_evaluation_set/run_model_comparison_analysis.py \
  --metrics all \
  --model-group all \
  --include-human-study
```

`--help` can be used on individual scripts. The WordNet corpora are additionally
required for WordNet recovery; `--download-wordnet` can be passed on its first
run.
