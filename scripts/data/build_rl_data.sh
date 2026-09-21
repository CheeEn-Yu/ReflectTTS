#!/bin/bash
# Build the RL two-hop training data end-to-end:
#   download   ParaSpeechCaps captions          (network/CPU)
#   prompts    captions -> uid/instruction/text  (CPU)
#   twohop     + v1 tokens + critique + clsp_v1 + wer_v1   (CUDA + Step-Audio-2)
#
# Stages 1-2 run anywhere; `twohop` is server-only (GPU + weights).
#
# Required once (for `download`):
#   huggingface-cli login    # accept the ParaSpeechCaps (CC-BY-NC-SA) license
#
# Optional env:
#   STAGES="download prompts twohop"   # which stages (default)
#   SPLIT_NAME=paraspeechcaps_en        # names data/rl/<SPLIT_NAME>{,_twohop}
#   DL_SPLITS="train_base dev"          # PSC source splits to pull
#   DEV_LIMIT=500                       # cap rows for the dev twohop build
#   TRAIN_LIMIT=                        # cap rows for the train twohop build ('' = all)
#   TRAIN_SAMPLE_SIZE=1000              # random training rows before WER filtering
#   DATA_SEED=0                         # sampling seed
#   MAX_WER_V1=0.10                     # paper's first-pass WER filter
#   MAX_ROWS=                           # smoke cap for download+prompts ('' = all)
#   CHAT_MODEL=Step-Audio-2-mini        # twohop v1/critic model
#   PROMPT_WAV=assets/default_male.wav
#   ASR_MODEL=openai/whisper-large-v3
#   CLSP_MODEL=yfyeung/CLSP
#   SPEAKER_RAG=1                       # match speaker-RAG training/evaluation
#   SPEAKER_EMB_CACHE=out/speaker_refs.emb.pt
#   DRY_RUN=1                           # print commands without executing
#
# Examples:
#   bash scripts/data/build_rl_data.sh                       # full
#   STAGES="download prompts" bash scripts/.../build_rl_data.sh            # local prep
#   STAGES="twohop" bash scripts/.../build_rl_data.sh                      # server gen
#   DRY_RUN=1 bash scripts/.../build_rl_data.sh                            # preview

set -euo pipefail

CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3}"
if [[ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda activate "${STEP_ENV:-step}" 2>/dev/null || true
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ====== Configurable ======
STAGES="${STAGES:-download prompts twohop}"
SPLIT_NAME="${SPLIT_NAME:-paraspeechcaps_en}"
DL_SPLITS="${DL_SPLITS:-train_base dev}"
DEV_LIMIT="${DEV_LIMIT:-500}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
TRAIN_SAMPLE_SIZE="${TRAIN_SAMPLE_SIZE:-1000}"
DATA_SEED="${DATA_SEED:-0}"
MAX_WER_V1="${MAX_WER_V1:-0.10}"
MAX_ROWS="${MAX_ROWS:-}"

EXTERNAL_DIR="${EXTERNAL_DIR:-data/external/paraspeechcaps}"
PROMPT_DIR="${PROMPT_DIR:-data/rl/${SPLIT_NAME}}"
TWOHOP_DIR="${TWOHOP_DIR:-data/rl/${SPLIT_NAME}_twohop}"

CHAT_MODEL="${CHAT_MODEL:-Step-Audio-2-mini}"
PROMPT_WAV="${PROMPT_WAV:-assets/default_male.wav}"
ASR_MODEL="${ASR_MODEL:-openai/whisper-large-v3}"
CLSP_MODEL="${CLSP_MODEL:-yfyeung/CLSP}"
SPEAKER_RAG="${SPEAKER_RAG:-0}"
SPEAKER_EMB_CACHE="${SPEAKER_EMB_CACHE:-out/speaker_refs.emb.pt}"

run_stage() {
    local label="$1"; shift
    echo "──────── $label ────────"
    if [[ -n "${DRY_RUN:-}" ]]; then
        printf '  %q ' "$@"; echo
    else
        "$@"
    fi
}

for stage in $STAGES; do
    case "$stage" in
        download)
            ARGS=(data_preprocess/download_paraspeechcaps.py
                  --splits $DL_SPLITS --output_dir "$EXTERNAL_DIR")
            [[ -n "$MAX_ROWS" ]] && ARGS+=(--max_rows "$MAX_ROWS")
            run_stage "1. download ParaSpeechCaps" python "${ARGS[@]}"
            ;;
        prompts)
            ARGS=(data_preprocess/build_prompts_paraspeechcaps.py
                  --input_dir "$EXTERNAL_DIR" --output_dir "$PROMPT_DIR")
            [[ -n "$MAX_ROWS" ]] && ARGS+=(--max_rows "$MAX_ROWS")
            run_stage "2. build prompts" python "${ARGS[@]}"
            ;;
        twohop)
            TRAIN_ARGS=(data_preprocess/build_twohop_prompts.py
                  --input_jsonl  "$PROMPT_DIR/train.jsonl"
                  --output_jsonl "$TWOHOP_DIR/train.jsonl"
                  --chat_model_path "$CHAT_MODEL" --asr_model "$ASR_MODEL"
                  --prompt_wav "$PROMPT_WAV" --clsp_model "$CLSP_MODEL"
                  --sample_size "$TRAIN_SAMPLE_SIZE" --seed "$DATA_SEED"
                  --max_wer_v1 "$MAX_WER_V1" --skip_existing)
            [[ "$SPEAKER_RAG" == "1" ]] && TRAIN_ARGS+=(--speaker_rag --speaker_emb_cache "$SPEAKER_EMB_CACHE")
            [[ -n "$TRAIN_LIMIT" ]] && TRAIN_ARGS+=(--limit "$TRAIN_LIMIT")
            run_stage "3a. twohop build (train)" python "${TRAIN_ARGS[@]}"

            DEV_ARGS=(data_preprocess/build_twohop_prompts.py
                  --input_jsonl  "$PROMPT_DIR/dev.jsonl"
                  --output_jsonl "$TWOHOP_DIR/dev.jsonl"
                  --chat_model_path "$CHAT_MODEL" --asr_model "$ASR_MODEL"
                  --prompt_wav "$PROMPT_WAV" --clsp_model "$CLSP_MODEL"
                  --max_wer_v1 "$MAX_WER_V1" --skip_existing)
            [[ "$SPEAKER_RAG" == "1" ]] && DEV_ARGS+=(--speaker_rag --speaker_emb_cache "$SPEAKER_EMB_CACHE")
            [[ -n "$DEV_LIMIT" ]] && DEV_ARGS+=(--limit "$DEV_LIMIT")
            run_stage "3b. twohop build (dev)" python "${DEV_ARGS[@]}"
            ;;
        *)
            echo "Unknown stage: $stage" >&2; exit 1 ;;
    esac
done

echo
echo "Done. RL twohop data in: $TWOHOP_DIR/{train,dev}.jsonl"
echo "Train RL with: STEPS=1000 sbatch scripts/train/run_rl_twohop.sbatch"
