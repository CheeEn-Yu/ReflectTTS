#!/bin/bash
# Singularity / Apptainer launcher for the Stepfun-vLLM server.
#
# Assumes Singularity or Apptainer is available as `singularity`.
#
# One-time setup (slow, ~10 GB image):
#   singularity pull stepaudio2-vllm.sif \
#       docker://stepfun2025/vllm:step-audio-2-v20250909
#   # put the .sif on $SCRATCH or project storage, not $HOME quota.
#
# Then on an interactive GPU node (or inside an sbatch job — see
# run_onehop_infer_zeroshot.sbatch):
#   bash scripts/infer/run_vllm_server.singularity.sh
#
# With LoRA:
#   ADAPTER_PATH=$PWD/out/sft/normal_reasoning_v2/adapter_final \
#   LORA_NAME=v2 \
#   bash scripts/infer/run_vllm_server.singularity.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# SIF image (override via env).
SIF="${SIF:-${REPO_ROOT}/stepaudio2-vllm.sif}"
if [[ ! -f "$SIF" ]]; then
    echo "SIF not found at $SIF" >&2
    echo "Run once: singularity pull $SIF docker://stepfun2025/vllm:step-audio-2-v20250909" >&2
    exit 1
fi

MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/Step-Audio2/Step-Audio-2-mini}"
SERVED_NAME="${SERVED_NAME:-step-audio-2-mini}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
TP="${TP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Cache dirs — keep big artifacts off $HOME.
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/.cache}"
mkdir -p "$CACHE_DIR"/{hf,vllm,xdg,triton,tmp}

# Optional LoRA (absolute host path).
ADAPTER_PATH="${ADAPTER_PATH:-}"
LORA_NAME="${LORA_NAME:-lora}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
MAX_LORAS="${MAX_LORAS:-1}"

BINDS=( -B "$MODEL_DIR:/model" )
EXTRA_FLAGS=()
if [[ -n "$ADAPTER_PATH" ]]; then
    if [[ ! -d "$ADAPTER_PATH" ]]; then
        echo "ADAPTER_PATH=$ADAPTER_PATH does not exist" >&2; exit 1
    fi
    BINDS+=( -B "$ADAPTER_PATH:/lora/$LORA_NAME" )
    EXTRA_FLAGS+=(
        --enable-lora
        --max-lora-rank "$MAX_LORA_RANK"
        --max-loras     "$MAX_LORAS"
        --lora-modules  "$LORA_NAME=/lora/$LORA_NAME"
    )
    echo "[vllm] LoRA: $ADAPTER_PATH -> served as model='$LORA_NAME'"
    echo "[vllm] NOTE: requires Stepfun model class with supports_lora=True."
    echo "[vllm]       If load fails, merge first with infer/merge_lora.py."
fi

# Singularity env: NVIDIA driver via --nv, isolate caches via SINGULARITYENV_*.
export SINGULARITYENV_HF_HOME="$CACHE_DIR/hf"
export SINGULARITYENV_VLLM_CACHE_ROOT="$CACHE_DIR/vllm"
export SINGULARITYENV_XDG_CACHE_HOME="$CACHE_DIR/xdg"
# nvcc (flashinfer JIT) writes intermediates to $TMPDIR; the container's /tmp is
# not writable on this cluster, so point it at a writable $HOME-backed dir.
export SINGULARITYENV_TMPDIR="$CACHE_DIR/tmp"
export SINGULARITYENV_TRITON_CACHE_DIR="$CACHE_DIR/triton"

set -x
exec singularity exec --nv \
    "${BINDS[@]}" \
    "$SIF" \
    vllm serve /model \
        --served-model-name "$SERVED_NAME" \
        --port "$PORT" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs  "$MAX_NUM_SEQS" \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --enable-auto-tool-choice \
        --tool-call-parser step_audio_2 \
        --tokenizer-mode  step_audio_2 \
        --chat_template_content_format string \
        --audio-parser    step_audio_2_tts_ta4 \
        --trust-remote-code \
        "${EXTRA_FLAGS[@]}"
