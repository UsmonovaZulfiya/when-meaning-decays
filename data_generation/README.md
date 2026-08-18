# Data generation

The full-precision and 4-bit drivers implement the same two experiments. Their
only direct code difference is the loader import:

- `sem_drift_experiment.py` uses `models_local.py`.
- `sem_drift_experiment_quant.py` uses `models_local_quant.py`.

Pipeline A performs `description -> guess -> description generated from the
guess`. Pipeline B performs `description -> guess (recorded) -> paraphrase of
the current description`.

## Server layout

The default project root contains:

```text
final_submission_code/data_generation/
hf_cache/
models/<model-key>/
outputs/semantic_drift_from_golden/golden_descriptions_gpt_oss.csv
venv/
```

`PROJECT_ROOT` or `CODE_DIR` can be overridden at submission time when needed.

## SLURM arrays

Both batch files use the same eight tasks:

| Task | Pipeline | Model |
|---:|:---:|---|
| 0 | A | `llama-3.1-8b-instruct` |
| 1 | A | `gemma-3-4b-it` |
| 2 | A | `gemma-3-12b-it` |
| 3 | A | `gemma-3-27b-it` |
| 4 | B | `llama-3.1-8b-instruct` |
| 5 | B | `gemma-3-4b-it` |
| 6 | B | `gemma-3-12b-it` |
| 7 | B | `gemma-3-27b-it` |

Every model and pipeline can be run for both loading modes:

```bash
sbatch run_original.sbatch
sbatch run_quant.sbatch
```

Selected configurations can be run with, for example,
`sbatch --array=2,6 run_original.sbatch` or
`sbatch --array=0,4 run_quant.sbatch`.

Generated CSVs use the following column schema:

```text
model_name, model_id, category, original_word, instance_id, step,
description, guess, source, prompt_mode, guess_temperature,
generation_temperature
```