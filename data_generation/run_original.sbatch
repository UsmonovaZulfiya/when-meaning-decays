#!/bin/bash
#SBATCH -J semantic_drift_original
#SBATCH -p a100_only
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH --array=0-7
#SBATCH -o /logs/%x_%A_%a.out
#SBATCH -e /logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/}"
CODE_DIR="${CODE_DIR:-$PROJECT_ROOT/final_submission_code/data_generation}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Pipeline A is run by tasks 0-3; Pipeline B is run by tasks 4-7.
MODELS=(
    "llama-3.1-8b-instruct"
    "gemma-3-4b-it"
    "gemma-3-12b-it"
    "gemma-3-27b-it"
)

if (( TASK_ID < 0 || TASK_ID >= 8 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 7." >&2
    exit 2
fi

MODEL_INDEX=$((TASK_ID % ${#MODELS[@]}))
DEFAULT_PIPELINE="a"
if (( TASK_ID >= ${#MODELS[@]} )); then
    DEFAULT_PIPELINE="b"
fi

export PROJECT_ROOT
export MODEL_KEY="${MODEL_KEY:-${MODELS[$MODEL_INDEX]}}"
export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/$MODEL_KEY}"
export MODEL_LABEL="${MODEL_LABEL:-$MODEL_KEY}"
export PIPELINE="${PIPELINE:-$DEFAULT_PIPELINE}"
export GOLDEN_DESCRIPTIONS_FILE="${GOLDEN_DESCRIPTIONS_FILE:-$PROJECT_ROOT/outputs/semantic_drift_from_golden/golden_descriptions_gpt_oss.csv}"

if [[ "$PIPELINE" == "a" ]]; then
    DEFAULT_OUTPUT_ROOT="$PROJECT_ROOT/outputs/semantic_drift_from_golden/full_precision/pipelinea"
elif [[ "$PIPELINE" == "b" ]]; then
    DEFAULT_OUTPUT_ROOT="$PROJECT_ROOT/outputs/semantic_drift_from_golden/full_precision/pipelineb"
else
    echo "PIPELINE must be 'a' or 'b', received: $PIPELINE" >&2
    exit 2
fi
export OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"

export NUM_STEPS="${NUM_STEPS:-10}"
export NUM_INSTANCES="${NUM_INSTANCES:-100}"
export MAX_WORDS_PER_CATEGORY="${MAX_WORDS_PER_CATEGORY:-0}"
export WORD_OFFSET_PER_CATEGORY="${WORD_OFFSET_PER_CATEGORY:-0}"
export ONLY_CATEGORIES="${ONLY_CATEGORIES:-High-Freq-Concrete,Low-Freq-Concrete,High-Freq-Abstract,Low-Freq-Abstract}"
export GUESS_TEMPERATURE="${GUESS_TEMPERATURE:-0.0}"
export GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.8}"
export GUESS_MAX_TOKENS="${GUESS_MAX_TOKENS:-16}"
export GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-140}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-8}"

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "$PROJECT_ROOT/logs" "$OUTPUT_ROOT" "$HF_HOME"
source "$PROJECT_ROOT/venv/bin/activate"

echo "Model: $MODEL_KEY"
echo "Pipeline: $PIPELINE"
echo "Backend: full precision"
echo "Code: $CODE_DIR/sem_drift_experiment.py"
echo "Output root: $OUTPUT_ROOT"

python "$CODE_DIR/sem_drift_experiment.py"
