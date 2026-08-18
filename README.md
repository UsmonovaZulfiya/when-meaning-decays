# When Meaning Decays: Semantic Drift in LLM Transmission Chains

This repository accompanies the paper **When Meaning Decays: Semantic Drift in
LLM Transmission Chains**. It contains the code used to generate repeated
word-description transmission chains and to reproduce the main evaluation,
robustness, ablation, and semantic-stability analyses.

## Overview

The project studies whether a target concept remains recoverable as its
description is repeatedly transformed by a language model. The controlled
experiment covers 400 English target concepts, 100 initial descriptions per
concept, 10 transmission steps, and five instruction-tuned Llama and Gemma
models.

Two transmission mechanisms are compared:

- **Pipeline A — lexically re-anchored guess–describe chain:** the model guesses
  the concept from the current description, and that guess is used to generate
  the next description.
- **Pipeline B — description-only paraphrase chain:** the current description is
  paraphrased, while a separate guess is recorded for evaluation but is not
  propagated to the next step.

The analysis code measures exact recovery, word–guess and description-level
semantic similarity, Shannon entropy, semantic sinks, WordNet-aware recovery,
embedding-model robustness, lexical predictors of final stability, and
temperature/quantization ablations.

## Repository structure

```text
.
├── data_generation/
│   ├── sem_drift_experiment.py          # full-precision transmission chains
│   ├── sem_drift_experiment_quant.py    # 4-bit transmission chains
│   ├── run_original.sbatch              # full-precision SLURM array
│   ├── run_quant.sbatch                 # quantized SLURM array
│   └── initial_description_generation/  # GPT-OSS preparation utilities
├── data_analysis/
│   ├── basic_evaluation_set/            # primary metrics, tables, and figures
│   ├── deeper_evaluation/               # sinks, WordNet, ablations, trajectories
│   └── predicting_semantic_stability/   # regression notebook and model search
├── datasets/                            # analysis data downloaded from Figshare
├── human_study/                         # human-chain preparation and summaries
├── requirements.txt
└── README.md
```

Shorter component-specific notes are provided in
[`data_generation/README.md`](data_generation/README.md) and
[`data_analysis/README.md`](data_analysis/README.md).

## Data

> **Download the research data from Figshare:**
> **https://figshare.com/s/d33601d454e63a767321**

The large experimental outputs are hosted separately due to large file size. The GitHub
repository contains the code; Figshare contains the larger research artifacts
needed to reproduce the reported analyses.

Place the downloaded files under `datasets/` using the paths expected by the
analysis code:

```text
datasets/
├── unified_semantic_drift_results.csv
├── golden_descriptions_gpt_oss.csv
├── word_categories_latest.xlsx
├── words_with_usage_count.csv
├── words_weird.csv
└── ablation/
    └── ...                              # released temperature/quantization runs
```

The unified CSV is the main input for the primary metrics, WordNet recovery,
calibrated similarity, trajectory extraction, and model-comparison figures.
The golden descriptions and lexical metadata are additionally used by the
semantic-sink and semantic-stability analyses. The ablation runner expects the
released ablation files under `datasets/ablation/`.

## Installation

Python 3.10 or newer is required by the source syntax. From the repository root:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The root requirements file covers data generation, notebooks, and analysis.
For analysis-only environments, the smaller dependency set can be installed
instead:

```bash
python -m pip install -r data_analysis/requirements.txt
```

Model generation additionally requires a compatible PyTorch/CUDA installation,
local model checkpoints, and sufficient accelerator memory. The 4-bit driver
uses `bitsandbytes`. Analysis can be run independently on the downloaded data;
CUDA is useful for embedding-heavy metrics but is not required for the exact
recovery and entropy calculations.

The semantic-similarity analyses use Sentence Transformers and may download the
selected embedding checkpoint on first use. WordNet recovery requires the NLTK
WordNet and Open Multilingual WordNet corpora; the corresponding script can
download them with `--download-wordnet`.

## Models and main configuration

The released analysis dataset contains results for:

- Llama 3.1 8B Instruct
- Llama 3.1 70B Instruct
- Gemma 3 4B IT
- Gemma 3 12B IT
- Gemma 3 27B IT

GPT-OSS 120B was used during initial-description generation and validation. The
current local-generation drivers expose model keys for Llama 3.1 8B Instruct
and the three Gemma models; the supplied SLURM arrays use those four keys.
Checkpoints are loaded from local directories rather than downloaded by the
experiment driver.

The default experimental settings encoded in the drivers are:

| Setting | Value |
|---|---:|
| Target concepts | 400 |
| Initial descriptions per concept | 100 |
| Transmission steps | 10 |
| Guessing temperature | 0.0 |
| Description/paraphrase temperature | 0.8 |
| Sampling top-p | 0.95 |
| Local batch size | 8 |

Generation settings can be overridden through the environment variables in
[`sem_drift_experiment.py`](data_generation/sem_drift_experiment.py) and
[`sem_drift_experiment_quant.py`](data_generation/sem_drift_experiment_quant.py).

## Reproducing the transmission experiments

The released `golden_descriptions_gpt_oss.csv` can be used as the starting point
without repeating the API-based preparation stage. The preparation utilities
under `data_generation/initial_description_generation/` require an
OpenAI-compatible GPT-OSS endpoint, `GPT_OSS_API_KEY`, `GPT_OSS_BASE_URL`, and
user-configured input/output paths.

### Direct Python execution

The following example runs Llama 3.1 8B from a local checkpoint. It is intended
for a POSIX shell launched from the repository root:

```bash
export PROJECT_ROOT="$PWD"
export MODEL_KEY="llama-3.1-8b-instruct"
export MODEL_PATH="$PWD/models/llama-3.1-8b-instruct"
export MODEL_LABEL="Llama 3.1 8B Instruct"
export GOLDEN_DESCRIPTIONS_FILE="$PWD/datasets/golden_descriptions_gpt_oss.csv"

# Pipeline A, full precision
export PIPELINE="a"
export OUTPUT_ROOT="$PWD/outputs/full_precision/pipeline_a"
python data_generation/sem_drift_experiment.py

# Pipeline B, full precision
export PIPELINE="b"
export OUTPUT_ROOT="$PWD/outputs/full_precision/pipeline_b"
python data_generation/sem_drift_experiment.py
```

The same configuration can be run with 4-bit model loading by selecting a
separate output directory and invoking the quantized driver:

```bash
export PIPELINE="a"  # change to "b" for Pipeline B
export OUTPUT_ROOT="$PWD/outputs/quantized/pipeline_a"
python data_generation/sem_drift_experiment_quant.py
```

`MODEL_KEY`, `MODEL_PATH`, and `MODEL_LABEL` should be changed together when a
different configured checkpoint is used. `MAX_WORDS_PER_CATEGORY` and
`WORD_OFFSET_PER_CATEGORY` support small test runs before a full submission.

### SLURM execution

[`run_original.sbatch`](data_generation/run_original.sbatch) and
[`run_quant.sbatch`](data_generation/run_quant.sbatch) are eight-task arrays.
Tasks 0–3 run Pipeline A and tasks 4–7 run Pipeline B over Llama 3.1 8B and the
three Gemma checkpoints. Scheduler directives, model placement, environment
activation, and log paths must be adapted to the target cluster before use.
After configuration, the arrays are submitted with:

```bash
sbatch data_generation/run_original.sbatch
sbatch data_generation/run_quant.sbatch
```

Running the complete generation study is computationally intensive. The
Figshare release is the recommended starting point when only the paper's
analyses and figures need to be reproduced.

## Reproducing the analysis

All commands below are run from the repository root and use
`datasets/unified_semantic_drift_results.csv` by default.

### Primary evaluation

A faster pass computes exact recovery and entropy for every model and pipeline:

```bash
python data_analysis/basic_evaluation_set/run_model_comparison_analysis.py \
  --metrics quick \
  --model-group all \
  --include-human-study
```

The complete evaluator adds word–guess semantic similarity,
description-to-initial-description semantic similarity, and description BLEU:

```bash
python data_analysis/basic_evaluation_set/run_model_comparison_analysis.py \
  --metrics all \
  --model-group all \
  --comparison-name all_exact_entropy_guess_similarity \
  --include-human-study
```

The evaluator processes the large CSV in chunks and writes tables, diagnostic
exports, embedding caches, and figures. `--models`, `--pipelines`, and
`--skip-plots` can be used for narrower or headless runs. Alternative embedding
models can be selected with `--embedding-model`.

### Embedding-model robustness

The complete evaluation above creates the default MPNet cache and overview.
The two robustness variants and their comparison tables are produced with:

```bash
python data_analysis/basic_evaluation_set/run_model_comparison_analysis.py \
  --metrics guess-sim \
  --model-group all \
  --embedding-model e5-large-v2 \
  --comparison-name all_guess_similarity_intfloat_e5_large_v2 \
  --skip-plots

python data_analysis/basic_evaluation_set/run_model_comparison_analysis.py \
  --metrics guess-sim \
  --model-group all \
  --embedding-model bge-large-en-v1.5 \
  --comparison-name all_guess_similarity_baai_bge_large_en_v1_5 \
  --skip-plots

python data_analysis/basic_evaluation_set/generate_embedding_robustness_tables.py
```

### Secondary and robustness analyses

```bash
# Calibrated final-step similarity using the three embedding caches above
python data_analysis/deeper_evaluation/compute_shuffled_similarity_baseline.py

# Conservative final-step WordNet-aware recovery
python data_analysis/deeper_evaluation/compute_wordnet_recovery_final_step.py \
  --download-wordnet

# Temperature and quantization ablations
python data_analysis/deeper_evaluation/run_ablation_model_comparison_analysis.py \
  --study all
```

Representative high-similarity trajectories can be extracted after an
embedding cache has been produced:

```bash
python data_analysis/deeper_evaluation/extract_highest_similarity_chains.py
```

The following notebooks contain the interactive analyses that are not exposed
as a single command-line workflow:

- [`semantic_sink_chain_analysis.ipynb`](data_analysis/deeper_evaluation/semantic_sink_chain_analysis.ipynb):
  semantic sinks, lexical/category shifts, and chain-level diagnostics.
- [`regression_semantic_stability.ipynb`](data_analysis/predicting_semantic_stability/regression_semantic_stability.ipynb):
  grouped regression and prediction of final semantic stability.
- [`word_stability_by_model.ipynb`](data_analysis/basic_evaluation_set/word_stability_by_model.ipynb):
  word-level stability views and human/model comparisons.

They can be opened with:

```bash
jupyter lab
```

## Outputs

Transmission drivers write a global checkpoint and a timestamped result CSV
under `OUTPUT_ROOT/<model-label>/`. Checkpoints allow completed category–word
pairs to be skipped when a run is resumed.

Analysis scripts write generated tables, figures, caches, and diagnostics under
`data_analysis/results_analysis/` unless `--output-dir` is supplied. Ablation
results are grouped under `data_analysis/results_analysis/ablation_studies/`.

## Computational requirements

Full model inference involves millions of chained generations and is intended
for GPU/HPC execution. Full-precision and 4-bit loading have different memory
requirements, and the suitable batch size depends on the checkpoint and
accelerator. No specific hardware configuration is assumed by this README.

Reproducing exact-recovery and entropy tables from the Figshare dataset is much
less demanding. Embedding-based analyses remain compute-intensive but avoid
rerunning the language-model transmission chains and reuse SQLite embedding
caches when available.

## Citation

Definitive proceedings metadata is not yet included in this repository. A
provisional citation is:

```bibtex
@unpublished{usmonova_when_meaning_decays,
  title  = {When Meaning Decays: Semantic Drift in LLM Transmission Chains},
  author = {Usmonova, Zulfiya and Patil, Viren Pankaj and Bueno, Ivo and Kasneci, Enkelejda}
}
```

## License and third-party resources

The code in this repository is released under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). External
models, datasets, lexical resources, and generated research artifacts remain
subject to their respective terms; the repository's software license does not
override those terms.

## Contact

**Zulfiya Usmonova**  
Technical University of Munich  
[zulfiya.usmonova@tum.de](mailto:zulfiya.usmonova@tum.de)
