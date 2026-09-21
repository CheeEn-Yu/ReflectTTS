#!/bin/bash
# ONE-HOP (single-pass) inference with an RL one-hop adapter on InstructTTSEval,
# end to end:
#   [1] resolve best ckpt from the training OUT_DIR (best_ckpt.json, else latest ckpt-step*)
#   [2] launch vLLM server (Singularity) with that adapter as a LoRA module
#   [3] synth the InstructTTSEval split WITH speaker RAG (instructtseval_vllm.py)
#   [4] score CLSP + WER (scripts/eval/run_clsp_score.sh -> final_result.txt)
#
# Meant to be chained after training:
#   sbatch --dependency=afterok:<train_jobid> scripts/infer/run_onehop_infer_rl.sbatch
#
# Env overrides:
#   TRAIN_OUT=out/rl/vllm_onehop/onehop_spkrag_main   # where ckpts live
#   EVAL_OUT=out/eval/onehop_spkrag_main              # synth + scores land here
#   LORA_NAME=onehop_spkrag  SPLIT=en  LIMIT=  CONCURRENCY=16  CLSP_MODEL=yfyeung/CLSP
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node 1
# NOTE: pass your Slurm account on the command line:  sbatch -A <account> ...
#SBATCH -p dev
#SBATCH -t 240
#SBATCH -o out/slurm/%x_%j.out
#SBATCH -e out/slurm/%x_%j.err
#SBATCH -J onehop_spkrag_eval

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p out/slurm
export PYTHONPATH="$REPO_ROOT"

module load miniconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-step}"
echo "[eval] python: $(which python)"

TRAIN_OUT="${TRAIN_OUT:-out/rl/vllm_onehop/onehop_spkrag_main}"
EVAL_OUT="${EVAL_OUT:-out/eval/onehop_spkrag_main}"
LORA_NAME="${LORA_NAME:-onehop_spkrag}"
SPLIT="${SPLIT:-en}"
LIMIT="${LIMIT:-}"
EXPECTED_PER_TASK="${EXPECTED_PER_TASK:-${LIMIT:-1000}}"
CONCURRENCY="${CONCURRENCY:-16}"
CLSP_MODEL="${CLSP_MODEL:-yfyeung/CLSP}"
TOKEN2WAV_PATH="${TOKEN2WAV_PATH:-Step-Audio-2-mini}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

RESULT_JSONL="${EVAL_OUT}/results.jsonl"
CLSP_JSONL="${EVAL_OUT}/results_clsp.jsonl"
GEN_DIR="${EVAL_OUT}/gen_wav"
mkdir -p "$EVAL_OUT" "$GEN_DIR"

fail() { echo "[eval] FATAL: $*" >&2; exit 1; }

# --- [1] resolve adapter ---------------------------------------------------
if [[ -n "${ADAPTER:-}" ]]; then
    ADAPTER_REL="$ADAPTER"
elif [[ -f "$TRAIN_OUT/best_ckpt.json" ]]; then
    ADAPTER_REL="$(python -c "import json;print(json.load(open('$TRAIN_OUT/best_ckpt.json'))['ckpt'])")"
else
    ADAPTER_REL="$(ls -d "$TRAIN_OUT"/ckpt-step*/adapter 2>/dev/null | sort -V | tail -1)"
fi
[[ -n "$ADAPTER_REL" ]] || fail "no ckpt found under $TRAIN_OUT"
case "$ADAPTER_REL" in /*) ADAPTER_PATH="$ADAPTER_REL";; *) ADAPTER_PATH="$REPO_ROOT/$ADAPTER_REL";; esac
[[ -d "$ADAPTER_PATH" ]] || fail "adapter dir missing: $ADAPTER_PATH"
echo "[eval] adapter = $ADAPTER_PATH"

# --- [1b] merge adapter into base weights ----------------------------------
# Stepfun's vLLM fork rejects LoRA at load ("StepAudio2ForCausalLM does not
# support LoRA yet"), so bake the adapter into a merged model dir and serve it
# as a plain MODEL_DIR (no --enable-lora). Merged dir on /work (off $HOME quota).
SERVED_NAME="${SERVED_NAME:-step-audio-2-mini}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH:-out}/merged-onehop_spkrag}"
if [[ -f "$MERGED_DIR/config.json" && -z "${FORCE_MERGE:-}" ]]; then
    echo "[eval] reuse merged dir $MERGED_DIR"
else
    echo "[eval] merging adapter -> $MERGED_DIR"
    python infer/merge_lora.py \
        --base_model   Step-Audio2/Step-Audio-2-mini \
        --adapter_path "$ADAPTER_PATH" \
        --output_dir   "$MERGED_DIR" --force || fail "merge_lora failed"
fi

# --- [2] launch vLLM server (Singularity) ----------------------------------
PORT="${PORT:-$(( 20000 + (SLURM_JOB_ID % 30000) ))}"
# SIF lives on /work (off $HOME quota), not the repo root the launcher defaults to.
export SIF="${SIF:-${SCRATCH:-.}/stepaudio2-vllm.sif}"
# Serve the merged dir as the base model — NO ADAPTER_PATH (avoids the LoRA path).
export PORT SERVED_NAME
export MODEL_DIR="$MERGED_DIR"
unset ADAPTER_PATH LORA_NAME
SERVER_LOG="out/slurm/vllm_${SLURM_JOB_ID}.server.log"
echo "[eval] START $(date '+%F %T')  job=$SLURM_JOB_ID  port=$PORT"

bash scripts/infer/run_vllm_server.singularity.sh > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[eval] server pid=$SERVER_PID  log=$SERVER_LOG"

cleanup() {
    echo "[eval] shutting down server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

API_BASE="http://127.0.0.1:${PORT}/v1"
echo "[eval] waiting for $API_BASE/models ..."
READY=0
for i in $(seq 1 120); do
    if curl -fsS "${API_BASE}/models" >/dev/null 2>&1; then READY=1; echo "[eval] server ready after ${i}*5s"; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 80 "$SERVER_LOG" >&2; fail "server died early (see $SERVER_LOG)"; fi
    sleep 5
done
[[ "$READY" == "1" ]] || { tail -n 80 "$SERVER_LOG" >&2; fail "timed out waiting for server"; }

# --- [3] synth WITH speaker RAG --------------------------------------------
EXTRA=( --speaker_rag )
[[ -n "$LIMIT" ]] && EXTRA+=(--limit "$LIMIT")
echo "[eval] synth -> $RESULT_JSONL"
python infer/instructtseval_vllm.py \
    --split        "$SPLIT" \
    --api_url      "${API_BASE}/chat/completions" \
    --model_name   "$SERVED_NAME" \
    --token2wav_model_path "$TOKEN2WAV_PATH" \
    --output_jsonl "$RESULT_JSONL" \
    --audio_dir    "$GEN_DIR" \
    --concurrency  "$CONCURRENCY" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --skip_existing \
    "${EXTRA[@]}" || fail "synth failed"

cleanup; trap - EXIT INT TERM   # free the GPU before scoring loads whisper+CLSP

# --- [4] score CLSP + WER --------------------------------------------------
echo "[eval] scoring -> $CLSP_JSONL (+ $EVAL_OUT/final_result.txt)"
SCORE_EXTRA=()
[[ -n "$LIMIT" ]] && SCORE_EXTRA+=(--limit "$LIMIT")
INPUT_JSONLS="$RESULT_JSONL" RESULT_DIR="$EVAL_OUT" CLSP_MODEL="$CLSP_MODEL" \
    EXPECTED_PER_TASK="$EXPECTED_PER_TASK" \
    ${LIMIT:+LIMIT="$LIMIT"} \
    bash scripts/eval/run_clsp_score.sh || fail "clsp scoring failed"

RESULT_FILE="$EVAL_OUT/final_result.txt"
[[ -f "$RESULT_FILE" ]] || fail "no final_result.txt produced"
SUMMARY="$(cat "$RESULT_FILE")"
echo "[eval] RESULT:"; echo "$SUMMARY"

echo "[eval] DONE $(date '+%F %T')"
