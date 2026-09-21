#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=10
#SBATCH -p dev
#SBATCH --time=04:00:00
#SBATCH -o out/slurm/%x_%j.out
#SBATCH -e out/slurm/%x_%j.err
#SBATCH -J gemini_instructtts

# LALM-as-a-judge for InstructTTSEval: scores generated speech for consistency
# with the style instruction (APS / DSD / RP). This is the judge reported in the
# paper (Gemini-2.5-Pro). Needs GENAI_API_KEY.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "FATAL: Gemini evaluation must run inside a Slurm allocation." >&2
    echo "Submit this file with sbatch, or call it with bash from a compute job." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p out/slurm

module load miniconda3 2>/dev/null || true
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-step}"
elif [[ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-step}"
else
    echo "FATAL: conda was not found; load miniconda3 or set CONDA_PATH." >&2
    exit 1
fi

EVAL_DIR="InstructTTSEval/eval"

# ====== User Configurable Variables ======
INPUT_JSONL="${INPUT_JSONL:-${EVAL_DIR}/example_en.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${EVAL_DIR}/example_en_score.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-${EVAL_DIR}/eval_prompt.txt}"
API_KEY="${API_KEY:-${GENAI_API_KEY:-}}"
MODEL_NAME="${MODEL_NAME:-models/gemini-2.5-pro}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-ALL}"   # ALL | APS | DSD | RP
NUM_WORKERS="${NUM_WORKERS:-10}"
EXPECTED_PER_TASK="${EXPECTED_PER_TASK:-1000}"

: "${API_KEY:?Set GENAI_API_KEY (or API_KEY) before running the Gemini judge}"
[[ -f "${EVAL_DIR}/gemini_eval.py" ]] || {
    echo "Missing ${EVAL_DIR}/gemini_eval.py; clone the official InstructTTSEval repo first." >&2
    exit 1
}
[[ -f "$INPUT_JSONL" ]] || { echo "Missing input JSONL: $INPUT_JSONL" >&2; exit 1; }
[[ -f "$PROMPT_FILE" ]] || { echo "Missing Gemini prompt: $PROMPT_FILE" >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT_JSONL")"

if [[ "${ALLOW_INCOMPLETE:-0}" != "1" ]]; then
    VALIDATE_TASKS=(APS DSD RP)
    [[ "$INSTRUCTION_TYPE" != "ALL" ]] && VALIDATE_TASKS=("$INSTRUCTION_TYPE")
    python scripts/eval/validate_results.py \
        --input_jsonl "$INPUT_JSONL" \
        --tasks "${VALIDATE_TASKS[@]}" \
        --expected_per_task "$EXPECTED_PER_TASK"
fi

# ====== Run Evaluation ======
python "${EVAL_DIR}/gemini_eval.py" \
    --input_jsonl "$INPUT_JSONL" \
    --output_jsonl "$OUTPUT_JSONL" \
    --prompt_file "$PROMPT_FILE" \
    --api_key "$API_KEY" \
    --model_name "$MODEL_NAME" \
    --instruction_type "$INSTRUCTION_TYPE" \
    --num_workers "$NUM_WORKERS"

if [[ "${ALLOW_INCOMPLETE:-0}" != "1" ]]; then
    python scripts/eval/validate_results.py \
        --input_jsonl "$OUTPUT_JSONL" \
        --tasks "${VALIDATE_TASKS[@]}" \
        --expected_per_task "$EXPECTED_PER_TASK" \
        --require_gemini_score
fi

echo "Evaluation completed! Results saved in $OUTPUT_JSONL"
