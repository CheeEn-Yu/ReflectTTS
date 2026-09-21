#!/bin/bash
# Materialize the rollout-only Qwen2 checkpoint view on a compute node. This is
# CPU work but holds the pruned checkpoint in RAM while writing it.
#
# Example:
#   sbatch -A <account> scripts/train/build_qwen2_view.sbatch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=64G
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH -o out/slurm/%x_%j.out
#SBATCH -e out/slurm/%x_%j.err
#SBATCH -J build_qwen2_view
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

module load miniconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "${STEP_ENV:-step}"

python -m train.rl_vllm.make_qwen2_view \
  --src "${MODEL_DIR:-Step-Audio2/Step-Audio-2-mini}" \
  --out "${VIEW:-out/qwen2_view/step-audio-2-mini-qwen2}"
