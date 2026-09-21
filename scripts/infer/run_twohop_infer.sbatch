#!/bin/bash
# TWO-HOP inference (the paper's main method) on InstructTTSEval, on the file-RPC
# vLLM sidecar (no singularity): v1 -> self-critique -> v2, all three stages with
# the RL adapter. Self-contained — it generates its own v1 and critique, so no
# precomputed data is needed. Scoring is delegated to scripts/eval/run_clsp_score.sh.
#
# One 2-GPU job:
#   [1] merge_view.py            adapter -> merged Qwen2-view  (step env, GPU A)
#   [2] vllm_rollout_server.py   sidecar serves the view       (vllm env, GPU B)
#   [3] infer_full_pipeline_vllm v1 -> critique -> v2          (step env, GPU A)
#   [4] scripts/eval/run_clsp_score.sh   CLSP + WER for v1 and v2
#
# Smoke:  ADAPTER=<run>/ckpt-stepNNN/adapter TAG=smoke LIMIT=20 \
#         sbatch -A <account> -p dev -t 00:50:00 scripts/infer/run_twohop_infer.sbatch
# Full:   ADAPTER=<run>/ckpt-stepNNN/adapter TAG=myrun \
#         sbatch -A <account> scripts/infer/run_twohop_infer.sbatch
# Interactive allocation (already on a compute node with two GPUs):
#   ADAPTER=... TAG=myrun bash scripts/infer/run_twohop_infer.sbatch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node 2
# NOTE: pass your Slurm account on the command line:  sbatch -A <account> ...
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH -o out/slurm/%x_%j.out
#SBATCH -e out/slurm/%x_%j.err
#SBATCH -J eval_full_pipeline
#SBATCH --requeue
set -uo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "FATAL: this launcher loads GPU models and must run inside a Slurm allocation." >&2
  echo "Acquire a compute shell first, for example:" >&2
  echo "  srun -A <account> -p dev -t 00:50:00 --gpus-per-node=2 --pty bash" >&2
  exit 1
fi

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p out/slurm

ADAPTER="${ADAPTER:?must set ADAPTER (e.g. out/rl/vllm_twohop/run_XXXX/ckpt-stepNNN/adapter)}"
TAG="${TAG:-$(basename "$(dirname "$ADAPTER")")_full}"
OUT_DIR="${OUT_DIR:-out/full_eval_vllm/${TAG}}"
VIEW="${VIEW:-out/qwen2_view/${TAG}_merged}"
BASE="${BASE:-Step-Audio-2-mini}"
PROMPT_WAV="${PROMPT_WAV:-assets/default_male.wav}"
SPLIT="${SPLIT:-en}"
INPUT_JSONL="${INPUT_JSONL:-}"          # local JSONL overrides the HF split
LIMIT="${LIMIT:-}"                       # set for a smoke run
EXPECTED_PER_TASK="${EXPECTED_PER_TASK:-${LIMIT:-1000}}"
CLSP_MODEL="${CLSP_MODEL:-yfyeung/CLSP}"
# SPEAKER_RAG=1 -> pick the token2wav reference per-instruction (gender filter +
# CLSP, speaker_rag pkg) for BOTH v1 and v2, instead of the fixed PROMPT_WAV.
SPEAKER_RAG="${SPEAKER_RAG:-1}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-256}"
# Single H200 (141GB) fits the sidecar co-resident with the step-env driver during
# gen; the sidecar is freed before scoring so CLSP+Whisper get the GPU.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
SCORE_BATCH="${SCORE_BATCH:-16}"
STEP_ENV="${STEP_ENV:-step}"
VLLM_ENV="${VLLM_ENV:-vllm}"

WD="$OUT_DIR/rpc_${SLURM_JOB_ID:-local}_${SLURM_RESTART_COUNT:-0}"
mkdir -p "$OUT_DIR" "$WD"

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_DRV="${GPUS[0]}"
GPU_SIDE="${GPUS[1]:-${GPUS[0]}}"
echo "START $(date '+%F %T')  adapter=$ADAPTER tag=$TAG gpu_drv=$GPU_DRV gpu_side=$GPU_SIDE view=$VIEW"

# ---------------------------------------------------------------------------
# [1/4] merge LoRA -> merged Qwen2-view. Skip if already built.
# ---------------------------------------------------------------------------
if [ ! -f "$VIEW/model.safetensors" ]; then
  echo "[1/4] merge_view -> $VIEW"
  (
    module load miniconda3 2>/dev/null || true; eval "$(conda shell.bash hook)"; conda activate "$STEP_ENV"
    cd "$REPO"; export PYTHONPATH="$PWD"; export CUDA_VISIBLE_DEVICES="$GPU_DRV"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -m train.rl_vllm.merge_view --adapter "$ADAPTER" --base "$BASE" --out "$VIEW"
  ) || { echo "merge_view failed"; exit 1; }
else
  echo "[1/4] reuse existing merged view $VIEW"
fi

# ---------------------------------------------------------------------------
# [2/4] launch vLLM sidecar (vllm env) on GPU B with the merged view.
# ---------------------------------------------------------------------------
echo "[2/4] launch vLLM sidecar (GPU $GPU_SIDE)"
SIDECAR_LOG="out/slurm/sidecar_${SLURM_JOB_ID:-local}_${SLURM_RESTART_COUNT:-0}.log"
(
  module load miniconda3 2>/dev/null || true; eval "$(conda shell.bash hook)"; conda activate "$VLLM_ENV"
  cd "$REPO"; export PYTHONPATH="$PWD"; export CUDA_VISIBLE_DEVICES="$GPU_SIDE"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec python -m train.rl_vllm.vllm_rollout_server \
      --view_dir "$VIEW" --workdir "$WD" --gpu_mem_util "$GPU_MEM_UTIL" --max_model_len 8192
) > "$SIDECAR_LOG" 2>&1 &
SIDE_PID=$!
cleanup() { kill "$SIDE_PID" 2>/dev/null; wait "$SIDE_PID" 2>/dev/null; }
trap cleanup EXIT

SIDECAR_READY_TIMEOUT="${SIDECAR_READY_TIMEOUT:-600}"
READY_DEADLINE=$((SECONDS + SIDECAR_READY_TIMEOUT))
while [ ! -f "$WD/ready" ]; do
  if ! kill -0 "$SIDE_PID" 2>/dev/null; then
    echo "FATAL: vLLM sidecar exited before becoming ready. Last log lines:"
    tail -n 80 "$SIDECAR_LOG" || true
    wait "$SIDE_PID"
    exit 1
  fi
  if (( SECONDS >= READY_DEADLINE )); then
    echo "FATAL: vLLM sidecar was not ready after ${SIDECAR_READY_TIMEOUT}s. Last log lines:"
    tail -n 80 "$SIDECAR_LOG" || true
    exit 1
  fi
  sleep 2
done

# ---------------------------------------------------------------------------
# [3/4] full 3-stage pipeline: v1 -> critique -> v2 (all with adapter).
# ---------------------------------------------------------------------------
echo "[3/4] full pipeline (v1 + critique + v2) -> $OUT_DIR  speaker_rag=$SPEAKER_RAG"
(
  module load miniconda3 2>/dev/null || true; eval "$(conda shell.bash hook)"; conda activate "$STEP_ENV"
  cd "$REPO"; export PYTHONPATH="$PWD"; export CUDA_VISIBLE_DEVICES="$GPU_DRV"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  SPKRAG_FLAG=()
  [[ "$SPEAKER_RAG" == "1" ]] && SPKRAG_FLAG=(--speaker_rag --clsp_model "$CLSP_MODEL")
  python infer/infer_full_pipeline_vllm.py \
      --workdir "$WD" --output_dir "$OUT_DIR" \
      --base "$BASE" --prompt_wav "$PROMPT_WAV" \
      --split "$SPLIT" ${INPUT_JSONL:+--input_jsonl "$INPUT_JSONL"} \
      --temperature "$TEMPERATURE" --top_p "$TOP_P" --top_k "$TOP_K" \
      --repetition_penalty "$REPETITION_PENALTY" --max_new_tokens "$MAX_NEW_TOKENS" \
      --max_think_tokens "$MAX_THINK_TOKENS" \
      ${V2_LEN_CAP:+--v2_len_cap "$V2_LEN_CAP" --v2_max_retries "${V2_MAX_RETRIES:-4}"} \
      "${SPKRAG_FLAG[@]}" \
      --no_stop ${LIMIT:+--limit "$LIMIT"}
) || { echo "pipeline failed"; exit 1; }
cleanup; trap - EXIT   # sidecar no longer needed; free GPU B before scoring
sleep 3

# ---------------------------------------------------------------------------
# [4/4] CLSP + WER scoring for both v1 and v2.
# ---------------------------------------------------------------------------
echo "[4/4] CLSP + WER scoring"
(
  module load miniconda3 2>/dev/null || true; eval "$(conda shell.bash hook)"; conda activate "$STEP_ENV"
  cd "$REPO"; export PYTHONPATH="$PWD"; export CUDA_VISIBLE_DEVICES="$GPU_DRV"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  V1_JSONL="$OUT_DIR/v1.jsonl"
  V2_JSONL="$OUT_DIR/v2.jsonl"
  V1_SCORED="$OUT_DIR/v1_scored.jsonl"
  V2_SCORED="$OUT_DIR/v2_scored.jsonl"
  V1_RESULT="$OUT_DIR/v1_result"
  V2_RESULT="$OUT_DIR/v2_result"

  echo "  scoring v1..."
  INPUT_JSONLS="$V1_JSONL" SUFFIX=_scored RESULT_DIR="$V1_RESULT" \
    EXPECTED_PER_TASK="$EXPECTED_PER_TASK" \
    BATCH_SIZE="$SCORE_BATCH" CLSP_MODEL="$CLSP_MODEL" \
    bash scripts/eval/run_clsp_score.sh

  echo "  scoring v2..."
  INPUT_JSONLS="$V2_JSONL" SUFFIX=_scored RESULT_DIR="$V2_RESULT" \
    EXPECTED_PER_TASK="$EXPECTED_PER_TASK" \
    BATCH_SIZE="$SCORE_BATCH" CLSP_MODEL="$CLSP_MODEL" \
    bash scripts/eval/run_clsp_score.sh

)

echo ""
echo "=== Results ==="
echo "v1: $(cat "$OUT_DIR/v1_result/final_result.txt" 2>/dev/null || echo 'N/A')"
echo "v2: $(cat "$OUT_DIR/v2_result/final_result.txt" 2>/dev/null || echo 'N/A')"
echo "END $(date '+%F %T')"
