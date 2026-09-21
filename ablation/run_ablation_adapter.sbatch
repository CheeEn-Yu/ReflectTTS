#!/bin/bash
# Critique ablation of the full 3-stage two-hop pipeline on the file-RPC vLLM
# sidecar. v1 + critique + v2 are ALL generated with the SAME RL adapter; only
# Stage 2 (the critique) varies by $CRITIQUE_MODE:
#   self   model-generated critique (the paper inference pipeline)
#   none   empty <think>  -> v1 -> v2 directly, NO critique
#   fixed  inject a rule-based "this is already good" critique
#   random deterministic word-salad control
#   trained training-matched structured critique path
# Then CLSP + WER scored for both v1 and v2 (InstructTTSEval).
#
# One 2-GPU job:
#   [1] merge_view.py            adapter -> merged Qwen2-view  (step env, GPU A)  [reused if VIEW exists]
#   [2] vllm_rollout_server.py   sidecar serves the view       (vllm env, GPU B)
#   [3] infer_ablation_pipeline  v1 -> critique($MODE) -> v2   (step env, GPU A)
#   [4] clsp_eval/clsp_eval.py   CLSP + WER for v1 and v2
#
# All three modes share ONE adapter, so its merged VIEW is built once and reused.
# Submit one job per CRITIQUE_MODE; build the shared VIEW once beforehand.
#
# Submit one mode manually (dev = 4h max on this cluster; `normal` is gone):
#   CRITIQUE_MODE=none sbatch -p dev -t 04:00:00 ablation/run_ablation_adapter.sbatch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node 2
# NOTE: pass your Slurm account on the command line:  sbatch -A <account> ...
#SBATCH -p dev
#SBATCH --time=04:00:00
#SBATCH -o out/slurm/%x_%j.out
#SBATCH -e out/slurm/%x_%j.err
#SBATCH -J ablate_critique
#SBATCH --requeue
set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p out/slurm

CRITIQUE_MODE="${CRITIQUE_MODE:?must set CRITIQUE_MODE (self|none|fixed|random|trained)}"
ADAPTER="${ADAPTER:?must set ADAPTER (e.g. out/rl/vllm_twohop/<run>/ckpt-stepNNN/adapter)}"
ADAPTER_TAG="${ADAPTER_TAG:-$(basename "$(dirname "$ADAPTER")")}"   # e.g. ckpt-step300
RUN_TAG="${RUN_TAG:-static_spkrag_g16_mb4_${ADAPTER_TAG}}"
TAG="${TAG:-ablate_${CRITIQUE_MODE}_${RUN_TAG}}"
OUT_DIR="${OUT_DIR:-out/ablation/${TAG}}"
# The merged view is per-ADAPTER (NOT per-mode) so all three modes reuse it.
VIEW="${VIEW:-out/qwen2_view/${RUN_TAG}_merged}"
BASE="${BASE:-Step-Audio-2-mini}"
PROMPT_WAV="${PROMPT_WAV:-assets/default_male.wav}"
SPLIT="${SPLIT:-en}"
INPUT_JSONL="${INPUT_JSONL:-}"          # local JSONL overrides the HF split
LIMIT="${LIMIT:-}"                       # set for a smoke run
EXPECTED_PER_TASK="${EXPECTED_PER_TASK:-${LIMIT:-1000}}"
CLSP_MODEL="${CLSP_MODEL:-yfyeung/CLSP}"
# SPEAKER_RAG=1 -> pick the token2wav reference per-instruction (gender filter +
# CLSP) for BOTH v1 and v2. The default adapter was trained/eval'd with it ON.
SPEAKER_RAG="${SPEAKER_RAG:-1}"
SPEAKER_EMB_CACHE="${SPEAKER_EMB_CACHE:-speaker_rag/references.emb.pt}"
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
echo "START $(date '+%F %T')  mode=$CRITIQUE_MODE adapter=$ADAPTER tag=$TAG gpu_drv=$GPU_DRV gpu_side=$GPU_SIDE view=$VIEW"

# ---------------------------------------------------------------------------
# [1/4] merge LoRA -> merged Qwen2-view. Skip if already built (shared across modes).
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
# [3/4] ablation pipeline: v1 -> critique($CRITIQUE_MODE) -> v2 (all with adapter).
# ---------------------------------------------------------------------------
echo "[3/4] ablation pipeline (mode=$CRITIQUE_MODE) -> $OUT_DIR  speaker_rag=$SPEAKER_RAG"
(
  module load miniconda3 2>/dev/null || true; eval "$(conda shell.bash hook)"; conda activate "$STEP_ENV"
  cd "$REPO"; export PYTHONPATH="$PWD"; export CUDA_VISIBLE_DEVICES="$GPU_DRV"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  SPKRAG_FLAG=()
  [[ "$SPEAKER_RAG" == "1" ]] && SPKRAG_FLAG=(--speaker_rag --clsp_model "$CLSP_MODEL" \
      --speaker_emb_cache "$SPEAKER_EMB_CACHE")
  python ablation/infer_ablation_pipeline_vllm.py \
      --workdir "$WD" --output_dir "$OUT_DIR" \
      --critique_mode "$CRITIQUE_MODE" \
      --base "$BASE" --prompt_wav "$PROMPT_WAV" \
      --split "$SPLIT" ${INPUT_JSONL:+--input_jsonl "$INPUT_JSONL"} \
      --temperature "$TEMPERATURE" --top_p "$TOP_P" --top_k "$TOP_K" \
      --repetition_penalty "$REPETITION_PENALTY" --max_new_tokens "$MAX_NEW_TOKENS" \
      --max_think_tokens "$MAX_THINK_TOKENS" \
      ${V2_LEN_CAP:+--v2_len_cap "$V2_LEN_CAP" --v2_max_retries "${V2_MAX_RETRIES:-4}"} \
      ${NUM_SHARDS:+--num_shards "$NUM_SHARDS" --shard_id "${SHARD_ID:-0}"} \
      ${FIXED_CRITIQUE_EN:+--fixed_critique "$FIXED_CRITIQUE_EN"} \
      ${FIXED_V1_TEXT:+--fixed_v1_text "$FIXED_V1_TEXT"} \
      ${FIXED_V1_MAX_CODES:+--fixed_v1_max_codes "$FIXED_V1_MAX_CODES"} \
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
