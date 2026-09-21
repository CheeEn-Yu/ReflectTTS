#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -p dev
#SBATCH --time=04:00:00
#SBATCH -o out/slurm/%x_%j.out
#SBATCH -e out/slurm/%x_%j.err
#SBATCH -J score_instructtts

# CLSP score (instruction<->speech cosine sim + WER) on InstructTTSEval-schema
# result JSONL(s). It can run standalone or be called by an inference job to
# score its v1/v2 outputs before the allocation exits.
#
# Two ways to pick inputs:
#   1. OUT_DIR=<dir>            -> scores <dir>/v1.jsonl and v2.jsonl
#   2. INPUT_JSONLS="a b ..."   -> scores exactly those files
# Each <x>.jsonl is scored to <x>_clsp.jsonl (override pattern via SUFFIX).
#
# Run on the server (CUDA + CLSP + whisper). Local box only for editing.
#
# Optional env:
#   CLSP_MODEL=yfyeung/CLSP
#   SUFFIX=_clsp            # output = ${input%.jsonl}${SUFFIX}.jsonl
#   SKIP_WER=1              # CLSP only, no ASR/WER
#   LIMIT=N                 # score first N rows
#   TASKS="APS DSD RP"      # subset of tasks
#   RESULT_DIR=<dir>        # also write <dir>/final_result.txt (per-task + overall)
#   BATCH_SIZE=N            # CLSP/ASR batch size (default 8)
#   REQUIRE_COMPLETE=0       # allow failed/missing generations (paper runs default to strict)
#   EXPECTED_PER_TASK=1000   # enforce the paper's full English benchmark size
#
# Submit a standalone score job:
#   INPUT_JSONLS="out/en/results.jsonl" EXPECTED_PER_TASK=1000 \
#     sbatch -A <account> scripts/eval/run_clsp_score.sh
# Or, from an existing compute-node allocation/job:
#   OUT_DIR=out/en_twohop_vllm_full bash scripts/eval/run_clsp_score.sh
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "FATAL: CLSP/Whisper scoring must run inside a Slurm allocation." >&2
    echo "Submit this file with sbatch, or call it with bash from a compute job." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3}"
module load miniconda3 2>/dev/null || true
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CLSP_MODEL="${CLSP_MODEL:-yfyeung/CLSP}"
SUFFIX="${SUFFIX:-_clsp}"

# Resolve inputs.
if [[ -n "${INPUT_JSONLS:-}" ]]; then
    read -r -a INPUTS <<< "$INPUT_JSONLS"
elif [[ -n "${OUT_DIR:-}" ]]; then
    INPUTS=()
    for f in v1 v2; do
        [[ -f "$OUT_DIR/$f.jsonl" ]] && INPUTS+=("$OUT_DIR/$f.jsonl")
    done
    [[ ${#INPUTS[@]} -gt 0 ]] || { echo "No v1.jsonl or v2.jsonl in $OUT_DIR" >&2; exit 1; }
else
    echo "Set OUT_DIR=<dir> or INPUT_JSONLS=\"a.jsonl b.jsonl\"" >&2; exit 1
fi

EXTRA=()
[[ -n "${SKIP_WER:-}" ]]   && EXTRA+=(--skip_wer)
[[ -n "${RESULT_DIR:-}" ]] && EXTRA+=(--result_dir "$RESULT_DIR")
[[ -n "${BATCH_SIZE:-}" ]] && EXTRA+=(--batch_size "$BATCH_SIZE")
[[ -n "${LIMIT:-}" ]]    && EXTRA+=(--limit "$LIMIT")
[[ -n "${TASKS:-}" ]]    && EXTRA+=(--tasks $TASKS)
[[ "${REQUIRE_COMPLETE:-1}" == "1" ]] && EXTRA+=(--require_complete)
[[ -n "${EXPECTED_PER_TASK:-}" ]] && EXTRA+=(--expected_per_task "$EXPECTED_PER_TASK")

for in_jsonl in "${INPUTS[@]}"; do
    out_jsonl="${in_jsonl%.jsonl}${SUFFIX}.jsonl"
    echo "──────── CLSP: $in_jsonl -> $out_jsonl ────────"
    python clsp_eval/clsp_eval.py \
        --model_id     "$CLSP_MODEL" \
        --input_jsonl  "$in_jsonl" \
        --output_jsonl "$out_jsonl" \
        "${EXTRA[@]}"
done

echo "Done. Per-task means printed above."
