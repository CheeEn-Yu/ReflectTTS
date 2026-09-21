#!/bin/bash
# RL BASELINE: one-hop (single-pass) GRPO for Step-Audio-2-mini with an absolute
# CLSP+WER reward — the comparison point for the two-hop refine paradigm.
#
# Differs from run_rl_twohop.sbatch only in the reward/rollout regime:
#   - --critique_mode none : instruction+text -> audio directly (no v1/critique/refine)
#   - absolute reward       : beta_clsp*CLSP - alpha_wer*min(WER, wer_cap)
#                             (no relative-improve term)
#   - SAME training data     : data/rl/paraspeechcaps_en_twohop/train.jsonl
#                              (the v1_tokens/critique/clsp_v1/wer_v1 fields are ignored)
#
# Same 2-GPU / two-process (trainer + vLLM sidecar) layout as run_rl_twohop.sbatch.
#
# Smoke:  STEPS=3 LIMIT=8 G=4 sbatch -p dev -t 00:50:00 scripts/train/run_rl_onehop_baseline.sbatch
# Full:   sbatch scripts/train/run_rl_onehop_baseline.sbatch
# Speaker RAG in the reward loop (per-instruction prompt_wav instead of fixed
# default_male.wav):  SPEAKER_RAG=1 sbatch scripts/train/run_rl_onehop_baseline.sbatch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node 2
# NOTE: pass your Slurm account on the command line:  sbatch -A <account> ...
#SBATCH -p normal
#SBATCH --time=36:00:00
#SBATCH -o out/slurm/%x_%j.out
#SBATCH -e out/slurm/%x_%j.err
#SBATCH -J rl_onehop_baseline
#SBATCH --requeue
set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p out/slurm

# vLLM "view" of the policy (Qwen2-shaped Step-Audio-2-mini) the sidecar loads.
VIEW="${VIEW:-out/qwen2_view/step-audio-2-mini-qwen2}"
# Same data as two-hop; one-hop mode ignores the v1/critique fields it carries.
TRAIN_JSONL="${TRAIN_JSONL:-data/rl/paraspeechcaps_en_twohop/train.jsonl}"
STEP_ENV="${STEP_ENV:-step}"
VLLM_ENV="${VLLM_ENV:-vllm}"

# ---- preflight: fail early with a clear message instead of mid-run ----
[ -d "$VIEW" ]        || { echo "FATAL: vLLM view missing: $VIEW (build via make_qwen2_view or symlink it)"; exit 1; }
[ -f "$TRAIN_JSONL" ] || { echo "FATAL: train data missing: $TRAIN_JSONL"; exit 1; }
[ -d "Step-Audio2/Step-Audio-2-mini" ] || { echo "FATAL: model missing: Step-Audio2/Step-Audio-2-mini"; exit 1; }

# Per-attempt workdir so a requeued leg starts with a clean file-RPC channel;
# the trainer auto-resumes from the latest ckpt in OUT_DIR.
WD="out/rl/vllm_onehop/rpc_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}"
rm -rf "$WD"; mkdir -p "$WD"

# Split the two allocated GPUs: trainer on the first, sidecar on the second.
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_TRAIN="${GPUS[0]}"
GPU_SIDE="${GPUS[1]:-${GPUS[0]}}"
SYNC="${SYNC:-nccl}"
NCCL_PORT=$((29500 + RANDOM % 3000))
echo "START $(date '+%F %T')  ONE-HOP baseline  trainer GPU=$GPU_TRAIN  sidecar GPU=$GPU_SIDE  sync=$SYNC port=$NCCL_PORT  WD=$WD"

# ---- launch vLLM sidecar (vllm env) ----
(
  module load miniconda3 2>/dev/null || true
  eval "$(conda shell.bash hook)"
  conda activate "$VLLM_ENV" || exit 1
  cd "$REPO"; export PYTHONPATH="$PWD"; export CUDA_VISIBLE_DEVICES="$GPU_SIDE"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python -m train.rl_vllm.vllm_rollout_server \
      --view_dir "$VIEW" --workdir "$WD" \
      --gpu_mem_util "${GPU_MEM_UTIL:-0.85}" --max_model_len "${MAX_MODEL_LEN:-8192}"
) > "out/slurm/sidecar_${SLURM_JOB_ID}.log" 2>&1 &
SIDE_PID=$!

cleanup() { kill "$SIDE_PID" 2>/dev/null; wait "$SIDE_PID" 2>/dev/null; }
trap cleanup EXIT

# Fail quickly when the sidecar environment or model view is broken instead of
# leaving the trainer in its full RPC timeout.
SIDECAR_READY_TIMEOUT="${SIDECAR_READY_TIMEOUT:-600}"
READY_DEADLINE=$((SECONDS + SIDECAR_READY_TIMEOUT))
while [ ! -f "$WD/ready" ]; do
  if ! kill -0 "$SIDE_PID" 2>/dev/null; then
    echo "FATAL: vLLM sidecar exited before becoming ready. Last log lines:"
    tail -n 80 "out/slurm/sidecar_${SLURM_JOB_ID}.log"
    wait "$SIDE_PID"
    exit 1
  fi
  if (( SECONDS >= READY_DEADLINE )); then
    echo "FATAL: vLLM sidecar was not ready after ${SIDECAR_READY_TIMEOUT}s. Last log lines:"
    tail -n 80 "out/slurm/sidecar_${SLURM_JOB_ID}.log"
    exit 1
  fi
  sleep 2
done

enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

TRAINER_ARGS=()
[[ -n "${LIMIT:-}" ]] && TRAINER_ARGS+=(--limit "$LIMIT")
if enabled "${SPEAKER_RAG:-0}"; then
  TRAINER_ARGS+=(--speaker_rag
                 --speaker_emb_cache "${SPEAKER_EMB_CACHE:-out/rl/speaker_emb_cache.pt}"
                 --clsp_model "${CLSP_MODEL:-yfyeung/CLSP}")
fi
enabled "${VERIFY_SYNC:-0}" && TRAINER_ARGS+=(--verify_sync)

# ---- run trainer (step env) ----
# Absolute CLSP+WER reward: reward = BETA_CLSP*CLSP - ALPHA_WER*min(WER, WER_CAP).
# Monotone, capped WER penalty (lower WER always better, bounded variance) — replaces
# the old max(0,1-WER) floor that went flat for WER>=1 and let the policy collapse.
# (lambda_improve / improve_fn / wer_penalty are two-hop-only and unused here.)
(
  module load miniconda3 2>/dev/null || true
  eval "$(conda shell.bash hook)"
  conda activate "$STEP_ENV" || exit 1
  cd "$REPO"; export PYTHONPATH="$PWD"; export CUDA_VISIBLE_DEVICES="$GPU_TRAIN"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python -m train.rl_vllm.train_rl_vllm \
      --workdir "$WD" \
      --output_dir "${OUT_DIR:-out/rl/vllm_onehop/run_${SLURM_JOB_ID}_${SYNC}}" \
      --train_jsonl "$TRAIN_JSONL" \
      --policy_path "${POLICY_PATH:-Step-Audio-2-mini}" \
      --steps "${STEPS:-5000}" \
      --G "${G:-16}" \
      --micro_batch "${MICRO_BATCH:-1}" --grad_accum "${GRAD_ACCUM:-1}" \
      --lr "${LR:-1e-4}" --warmup_steps "${WARMUP:-20}" --lr_min_ratio "${LR_MIN_RATIO:-0.1}" \
      --kl_coef "${KL:-0.05}" --ckpt_every "${CKPT_EVERY:-100}" \
      --grad_clip "${GRAD_CLIP:-1.0}" --clip_advantage "${CLIP_ADV:-5.0}" \
      --temperature "${TEMPERATURE:-0.9}" --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
      --lora_r "${LORA_R:-16}" --lora_alpha "${LORA_ALPHA:-32}" \
      --scorer_device cuda:0 \
      --critique_mode none \
      --alpha_wer "${ALPHA_WER:-0.5}" --beta_clsp "${BETA_CLSP:-0.5}" --wer_cap "${WER_CAP:-0.5}" \
      --sync "$SYNC" --nccl_port "$NCCL_PORT" --seed "${SEED:-0}" \
      "${TRAINER_ARGS[@]}"
)
RC=$?
echo "END $(date '+%F %T')  trainer rc=$RC"
exit $RC
