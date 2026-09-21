#!/bin/bash
# Critique-stage ABLATION on the BASE (pre-RL) Step-Audio-2-mini.
# v1 + critique + v2 generated with the BASE model (NO adapter); only the turn-2
# critique varies by CRITIQUE_MODE. Then CLSP+WER score v1 and v2. One 2-GPU job:
#   [1] make_qwen2_view.py       BASE ckpt -> Qwen2-view  (step env, GPU A)  [shared across runs]
#   [2] vllm_rollout_server.py   sidecar serves the view  (vllm env, GPU B)
#   [3] infer_ablation_vllm.py   v1 -> critique(MODE) -> v2 (step env, GPU A)
#   [4] clsp_eval/clsp_eval.py   CLSP + WER for v1 and v2
#
# Three ablations; submit one job per mode. Build the shared view with one job
# first; after that finishes, the remaining modes can run in parallel:
#   CRITIQUE_MODE=none   exp.1  v1 + v2, empty <think>      "is critique needed?"
#   CRITIQUE_MODE=fixed  exp.2  v1 + fixed neutral + v2     "does critique CONTENT matter?"
#   CRITIQUE_MODE=self   exp.3  v1 + self-critique + v2     "pre-RL self-critique performance"
#
# Smoke:  CRITIQUE_MODE=none TAG=smoke LIMIT=20 sbatch -p dev -t 00:50:00 ablation/run_ablation_base.sbatch
# Full:   CRITIQUE_MODE=none sbatch -p dev ablation/run_ablation_base.sbatch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node 2
# NOTE: pass your Slurm account on the command line:  sbatch -A <account> ...
#SBATCH -p dev
#SBATCH --time=04:00:00
#SBATCH -o out/slurm/%x_%j.out
#SBATCH -e out/slurm/%x_%j.err
#SBATCH -J abl_critique
#SBATCH --requeue
set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p out/slurm

CRITIQUE_MODE="${CRITIQUE_MODE:?must set CRITIQUE_MODE (none|fixed|self)}"
case "$CRITIQUE_MODE" in none|fixed|self) ;; *) echo "bad CRITIQUE_MODE=$CRITIQUE_MODE"; exit 1;; esac
TAG="${TAG:-base_${CRITIQUE_MODE}}"
OUT_DIR="${OUT_DIR:-out/ablation/${TAG}}"
BASE="${BASE:-Step-Audio-2-mini}"
# BASE Qwen2-view is identical for all three modes -> build once, share across runs.
VIEW="${VIEW:-out/qwen2_view/${BASE}_base_qwen2}"
PROMPT_WAV="${PROMPT_WAV:-assets/default_male.wav}"
SPLIT="${SPLIT:-en}"
INPUT_JSONL="${INPUT_JSONL:-}"          # local JSONL overrides the HF split
LIMIT="${LIMIT:-}"                       # set for a smoke run
EXPECTED_PER_TASK="${EXPECTED_PER_TASK:-${LIMIT:-1000}}"
CLSP_MODEL="${CLSP_MODEL:-yfyeung/CLSP}"
# Table VII keeps the retrieved reference speaker fixed across critique modes.
SPEAKER_RAG="${SPEAKER_RAG:-1}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
SCORE_BATCH="${SCORE_BATCH:-16}"
STEP_ENV="${STEP_ENV:-step}"
VLLM_ENV="${VLLM_ENV:-vllm}"

WD="$OUT_DIR/rpc_${SLURM_JOB_ID:-local}_${SLURM_RESTART_COUNT:-0}"
mkdir -p "$OUT_DIR" "$WD"

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_DRV="${GPUS[0]}"
GPU_SIDE="${GPUS[1]:-${GPUS[0]}}"
echo "START $(date '+%F %T')  mode=$CRITIQUE_MODE tag=$TAG gpu_drv=$GPU_DRV gpu_side=$GPU_SIDE view=$VIEW"

# ---------------------------------------------------------------------------
# [1/4] BASE Qwen2-view (no adapter). Shared across the three modes; build once
# before submitting multiple modes, so concurrent jobs do not write it together.
# ---------------------------------------------------------------------------
if [ ! -f "$VIEW/model.safetensors" ]; then
  echo "[1/4] make_qwen2_view (BASE, no adapter) -> $VIEW"
  (
    module load miniconda3 2>/dev/null || true; eval "$(conda shell.bash hook)"; conda activate "$STEP_ENV"
    cd "$REPO"; export PYTHONPATH="$PWD"; export CUDA_VISIBLE_DEVICES="$GPU_DRV"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -m train.rl_vllm.make_qwen2_view --src "$(python -c "from train.rl.train_rl import _resolve_step; print(_resolve_step('$BASE'))")" --out "$VIEW"
  ) || { echo "make_qwen2_view failed"; exit 1; }
else
  echo "[1/4] reuse existing base view $VIEW"
fi

# ---------------------------------------------------------------------------
# [2/4] launch vLLM sidecar (vllm env) on GPU B with the base view.
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
# [3/4] ablation pipeline: v1 -> critique(MODE) -> v2 (all base model).
# ---------------------------------------------------------------------------
echo "[3/4] ablation pipeline (mode=$CRITIQUE_MODE) -> $OUT_DIR  speaker_rag=$SPEAKER_RAG"
(
  module load miniconda3 2>/dev/null || true; eval "$(conda shell.bash hook)"; conda activate "$STEP_ENV"
  cd "$REPO"; export PYTHONPATH="$PWD"; export CUDA_VISIBLE_DEVICES="$GPU_DRV"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  SPKRAG_FLAG=()
  [[ "$SPEAKER_RAG" == "1" ]] && SPKRAG_FLAG=(--speaker_rag --clsp_model "$CLSP_MODEL")
  python ablation/infer_ablation_vllm.py \
      --workdir "$WD" --output_dir "$OUT_DIR" --critique_mode "$CRITIQUE_MODE" \
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
  python clsp_eval/clsp_eval.py --model_id "$CLSP_MODEL" \
      --input_jsonl "$V1_JSONL" --output_jsonl "$V1_SCORED" --result_dir "$V1_RESULT" \
      --batch_size "$SCORE_BATCH" --require_complete \
      --expected_per_task "$EXPECTED_PER_TASK"

  echo "  scoring v2..."
  python clsp_eval/clsp_eval.py --model_id "$CLSP_MODEL" \
      --input_jsonl "$V2_JSONL" --output_jsonl "$V2_SCORED" --result_dir "$V2_RESULT" \
      --batch_size "$SCORE_BATCH" --require_complete \
      --expected_per_task "$EXPECTED_PER_TASK"

)

echo ""
echo "=== Results (mode=$CRITIQUE_MODE) ==="
echo "v1: $(cat "$OUT_DIR/v1_result/final_result.txt" 2>/dev/null || echo 'N/A')"
echo "v2: $(cat "$OUT_DIR/v2_result/final_result.txt" 2>/dev/null || echo 'N/A')"
echo "END $(date '+%F %T')"
