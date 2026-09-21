#!/bin/bash
# Download the InstructTTSEval reference WAVs on a compute node.
# Optional: SPLIT=en OUT_DIR=out/eval/gt LIMIT=20 SKIP_EXISTING=0.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -p dev
#SBATCH --time=01:00:00
#SBATCH -o out/slurm/%x_%j.out
#SBATCH -e out/slurm/%x_%j.err
#SBATCH -J download_instructtts_gt

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p out/slurm

module load miniconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-step}"

SPLIT="${SPLIT:-en}"
OUT_DIR="${OUT_DIR:-out/eval/gt}"
ARGS=(
  --split "$SPLIT"
  --audio_dir "$OUT_DIR/wav"
  --output_jsonl "$OUT_DIR/results.jsonl"
)
[[ "${SKIP_EXISTING:-1}" == "1" ]] && ARGS+=(--skip_existing)
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "$LIMIT")

exec python scripts/eval/download_gt_audio.py "${ARGS[@]}"
