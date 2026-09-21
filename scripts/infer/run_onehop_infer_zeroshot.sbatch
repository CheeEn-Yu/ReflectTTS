#!/bin/bash
# ONE-HOP (single-pass) inference on InstructTTSEval with the FROZEN base model
# — the paper's zero-shot baseline. Launches the Stepfun vLLM server (Singularity)
# in the background, runs the instructtseval_vllm client against it, then shuts the
# server down. Both processes share one GPU node. Point MODEL_DIR at a merged
# model dir to run the same single pass with an adapter instead.
# Scoring is a separate step: scripts/eval/run_clsp_score.sh.
#
# Usage:
#   sbatch -A <account> scripts/infer/run_onehop_infer_zeroshot.sbatch
#
# Env overrides (same names as run_vllm_server.singularity.sh):
#   SIF=...  MODEL_DIR=...  ADAPTER_PATH=...  LORA_NAME=v2
#   SPLIT=en LIMIT=100 CONCURRENCY=16
#
# Notes:
# - Server + client co-located on the same node. PORT defaults to a random
#   high port to avoid clashing with other users' jobs.
# - We wait for /v1/models to return 200 before launching the client.
# - On exit (or scancel) the trap kills the server cleanly.
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node 1
# NOTE: pass your Slurm account on the command line:  sbatch -A <account> ...
#SBATCH -p dev
#SBATCH -t 60
#SBATCH -o out/slurm/%x_%j.out
#SBATCH -e out/slurm/%x_%j.err
#SBATCH -J vllm_instructtseval

set -euo pipefail

# NOTE: sbatch stages the script to /var/spool/slurmd/..., so BASH_SOURCE
# can't locate the repo. Use SLURM_SUBMIT_DIR (the cwd at sbatch time).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p out/slurm
export PYTHONPATH="$REPO_ROOT"

# The vLLM *server* runs inside Singularity; the smoke test + client run on the
# host and need torch / token2wav / CLSP (speaker_rag) from the `step` env.
module load miniconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-step}"
echo "[submit] python: $(which python)"

# Pick a free-ish high port (job-id based -> stable across re-runs of same job,
# different between concurrent jobs).
PORT="${PORT:-$(( 20000 + (SLURM_JOB_ID % 30000) ))}"
export PORT

SPLIT="${SPLIT:-en}"
LIMIT="${LIMIT:-}"
CONCURRENCY="${CONCURRENCY:-16}"
# SPEAKER_RAG=1 -> pick the token2wav reference per-instruction (gender filter +
# CLSP, speaker_rag pkg) instead of the fixed --prompt_wav.
SPEAKER_RAG="${SPEAKER_RAG:-1}"
MODEL_NAME="${MODEL_NAME:-step-audio-2-mini}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
OUT_DIR="${OUT_DIR:-out/${SPLIT}_vllm}"
RESULT_JSONL="${RESULT_JSONL:-${OUT_DIR}/results.jsonl}"
GEN_DIR="${GEN_DIR:-${OUT_DIR}/gen_wav}"
# Local snapshot whose token2wav/ the client uses (server-side weights live in SIF mount).
TOKEN2WAV_PATH="${TOKEN2WAV_PATH:-Step-Audio-2-mini}"

mkdir -p "$OUT_DIR" "$GEN_DIR"

SERVER_LOG="out/slurm/vllm_${SLURM_JOB_ID}.server.log"

echo "[submit] START $(date '+%F %T')  job=$SLURM_JOB_ID  port=$PORT"

# --- launch server in background -----------------------------------------
bash scripts/infer/run_vllm_server.singularity.sh > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[submit] server pid=$SERVER_PID  log=$SERVER_LOG"

cleanup() {
    echo "[submit] shutting down server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    # give vLLM up to 30s for graceful shutdown, then SIGKILL.
    for _ in $(seq 1 30); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- wait for server ready ------------------------------------------------
API_BASE="http://127.0.0.1:${PORT}/v1"
echo "[submit] waiting for $API_BASE/models ..."
READY=0
for i in $(seq 1 120); do      # up to ~10 min (model load on cold cache is slow)
    if curl -fsS "${API_BASE}/models" >/dev/null 2>&1; then
        READY=1
        echo "[submit] server ready after ${i}*5s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[submit] server died early — see $SERVER_LOG" >&2
        tail -n 80 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 5
done
if [[ "$READY" != "1" ]]; then
    echo "[submit] timed out waiting for server" >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 1
fi

# --- batch synth ----------------------------------------------------------
EXTRA=()
[[ -n "$LIMIT" ]] && EXTRA+=(--limit "$LIMIT")
[[ "$SPEAKER_RAG" == "1" ]] && EXTRA+=(--speaker_rag)

START_TS=$(date +%s)
python infer/instructtseval_vllm.py \
    --split        "$SPLIT" \
    --api_url      "${API_BASE}/chat/completions" \
    --model_name   "$MODEL_NAME" \
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
    "${EXTRA[@]}"
ELAPSED=$(( $(date +%s) - START_TS ))

echo "[submit] END   $(date '+%F %T')  synth elapsed=${ELAPSED}s"
echo "[submit] results -> $RESULT_JSONL"
