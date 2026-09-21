"""Build twohop RL training data: v1 tokens (chat model) + LALM critique.

For each row in an existing RL JSONL (uid, instruction, text, lang):
  1. Run Step-Audio-2-mini (chat) with a proper style-instruction TTS prompt
     → v1 raw token IDs  (the base model is gone; bare-text prompting produced
     unintelligible v1 with WER ~1.0, so we use the chat model + real prompt)
  2. Vocode v1 tokens → wav bytes
  3. Run LALM critic on (instruction, v1_wav) → critique text
  4. Score CLSP(v1_wav, instruction) → clsp_v1 baseline (for relative RL reward)
  5. ASR(v1_wav) → wer_v1 baseline (for the v1-relative WER penalty)
  6. Write new JSONL: original fields + v1_tokens + critique + clsp_v1 + wer_v1

Output: data/rl/<split>_twohop/{train,dev}.jsonl

Server-only (CUDA + Step-Audio-2 weights). Run from the repository root.

Usage:
    python data_preprocess/build_twohop_prompts.py \\
        --input_jsonl  data/rl/paraspeechcaps_en/train.jsonl \\
        --output_jsonl data/rl/paraspeechcaps_en_twohop/train.jsonl \\
        --limit 5000

    # Process both splits:
    python data_preprocess/build_twohop_prompts.py \\
        --input_jsonl data/rl/paraspeechcaps_en/train.jsonl \\
        --output_jsonl data/rl/paraspeechcaps_en_twohop/train.jsonl
    python data_preprocess/build_twohop_prompts.py \\
        --input_jsonl data/rl/paraspeechcaps_en/dev.jsonl \\
        --output_jsonl data/rl/paraspeechcaps_en_twohop/dev.jsonl \\
        --limit 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path

import torch as _torch
import torchaudio as _torchaudio
import soundfile as _sf


# torchcodec bypass (must happen before stepaudio2 / token2wav imports)
def _sf_load(uri, *_, **__):
    data, sr = _sf.read(uri, always_2d=True, dtype="float32")
    return _torch.from_numpy(data.T.copy()), sr

def _sf_save(uri, src, sample_rate, *, format=None, channels_first=True, **__):
    if isinstance(src, _torch.Tensor):
        src = src.detach().cpu().numpy()
    if src.ndim == 2 and channels_first:
        src = src.T
    _sf.write(uri, src, sample_rate, format=(format.upper() if format else None))

_torchaudio.load = _sf_load
_torchaudio.save = _sf_save

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
STEP_DIR = REPO_ROOT / "Step-Audio2"
sys.path.insert(0, str(STEP_DIR))
sys.path.insert(0, str(REPO_ROOT / "infer"))
sys.path.insert(0, str(REPO_ROOT))  # for clsp_eval.clsp_utils

AUDIO_TOKEN_VOCAB_SIZE = 6561
AUDIO_TOKEN_OFFSET = 151696


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else str(STEP_DIR / path)


def _gen_v1_tokens(chat_model, instruction: str, text: str, lang: str,
                   max_new_tokens: int, temperature: float) -> list[int]:
    """Run the chat model with a style-instruction TTS prompt → raw LM token IDs.

    Mirrors the twohop turn-1 template (verified to give WER ~0): a system turn
    carrying the style instruction + "read aloud" framing, the text as the human
    turn, and an open assistant turn primed with <tts_start>.
    """
    system = f"{instruction}\nRead the following text aloud in the speaking style described above."
    messages = [
        {"role": "system", "content": system},
        {"role": "human", "content": text},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]
    output_token_ids, _decoded, _audio_tokens = chat_model(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )
    # raw_token_ids: only the audio-range tokens from the full output
    return [
        t for t in output_token_ids
        if AUDIO_TOKEN_OFFSET <= t < AUDIO_TOKEN_OFFSET + AUDIO_TOKEN_VOCAB_SIZE
    ]


def _vocode_to_tmp(token2wav, raw_token_ids: list[int], prompt_wav: str) -> str:
    """Vocode tokens → wav bytes → temp file path (caller must delete)."""
    audio_codes = [
        t - AUDIO_TOKEN_OFFSET
        for t in raw_token_ids
        if (t - AUDIO_TOKEN_OFFSET) < AUDIO_TOKEN_VOCAB_SIZE
    ]
    wav_bytes = token2wav(audio_codes, prompt_wav=prompt_wav)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(wav_bytes)
    return tmp_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--chat_model_path", default="Step-Audio-2-mini",
                   help="Chat model used for BOTH v1 generation and LALM critique "
                        "(resolved relative to Step-Audio2/).")
    p.add_argument("--asr_model", default="openai/whisper-large-v3",
                   help="ASR model for precomputing the v1 WER baseline.")
    p.add_argument("--prompt_wav", default="assets/default_male.wav",
                   help="Speaker reference for token2wav.")
    p.add_argument("--clsp_model", default="yfyeung/CLSP",
                   help="CLSP model for precomputing the v1 baseline score.")
    p.add_argument("--clsp_device", default="cuda",
                   help="Device for the CLSP model.")
    p.add_argument("--speaker_rag", action="store_true",
                   help="Select the vocoder reference per instruction, matching "
                        "speaker-RAG training and inference.")
    p.add_argument("--speaker_emb_cache", default=None,
                   help="Optional reference-embedding cache for --speaker_rag.")
    p.add_argument("--v1_max_new_tokens", type=int, default=2048,
                   help="Max audio tokens for v1 generation.")
    p.add_argument("--v1_temperature", type=float, default=0.9)
    p.add_argument("--critic_max_new_tokens", type=int, default=256)
    p.add_argument("--critic_temperature", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N rows (smoke test).")
    p.add_argument("--sample_size", type=int, default=None,
                   help="Randomly sample this many input rows before --limit. "
                        "Use 1000 to reproduce the paper dataset.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed used by --sample_size.")
    p.add_argument("--max_wer_v1", type=float, default=0.10,
                   help="Drop completed rows whose first-pass WER exceeds this "
                        "threshold (paper: 0.10). Use a negative value to disable.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip rows that already appear in output_jsonl (for resuming).")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Items per batch (keep 1 to avoid OOM with long audio).")
    return p.parse_args()


def _load_existing_uids(path: Path) -> set[str]:
    uids: set[str] = set()
    if not path.exists():
        return uids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    uids.add(json.loads(line)["uid"])
                except Exception:
                    pass
    return uids


def main() -> None:
    args = parse_args()
    from tqdm import tqdm

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    selected_uids = None
    if args.sample_size is not None:
        if len(rows) > args.sample_size:
            rows = random.Random(args.seed).sample(rows, args.sample_size)
        selected_uids = {r["uid"] for r in rows}
    if args.limit is not None:
        rows = rows[:args.limit]
    print(f"[build_twohop] input rows: {len(rows)}")

    existing_uids = _load_existing_uids(output_path) if args.skip_existing else set()
    rows = [r for r in rows if r["uid"] not in existing_uids]
    print(f"[build_twohop] rows to process: {len(rows)} "
          f"(skipped {len(existing_uids)} existing)")

    # ---- Load models ----
    from stepaudio2 import StepAudio2  # type: ignore
    from token2wav import Token2wav  # type: ignore
    from text_critic import make_critique  # type: ignore
    from clsp_eval.clsp_utils import (  # type: ignore
        load_clsp, score as clsp_score,
        load_asr, transcribe, wer as wer_fn, to_mono_16k,
    )

    chat_dir = _resolve(args.chat_model_path)
    prompt_wav = _resolve(args.prompt_wav)

    print(f"[build_twohop] loading chat model (v1 gen + critic): {chat_dir}")
    chat_model = StepAudio2(chat_dir)
    token2wav = Token2wav(os.path.join(chat_dir, "token2wav"))

    clsp_device = _torch.device(args.clsp_device)
    print(f"[build_twohop] loading CLSP model: {args.clsp_model} on {clsp_device}")
    clsp_model = load_clsp(args.clsp_model, clsp_device)
    print(f"[build_twohop] loading ASR model: {args.asr_model} on {clsp_device}")
    asr_model, asr_proc = load_asr(args.asr_model, clsp_device)

    references = ref_embeddings = None
    if args.speaker_rag:
        from speaker_rag import DEFAULT_REFERENCES, embed_references

        references = DEFAULT_REFERENCES
        ref_embeddings = embed_references(
            references,
            clsp_model,
            clsp_device,
            cache_path=args.speaker_emb_cache,
        )
        print(
            "[build_twohop] speaker_rag ON: "
            + ", ".join(ref.name for ref in references)
        )

    def select_prompt_wav(instruction: str) -> str:
        if references is None:
            return prompt_wav
        from speaker_rag import select_reference

        return select_reference(
            instruction,
            clsp_model,
            clsp_device,
            references,
            ref_embeddings,
        ).reference.wav

    n_ok = n_err = 0
    mode = "a" if args.skip_existing else "w"
    with output_path.open(mode, encoding="utf-8", buffering=1) as out_f:
        for row in tqdm(rows, desc="build_twohop"):
            uid = row["uid"]
            instruction = row["instruction"]
            text = row["text"]
            lang = row.get("lang", "en")
            result = {k: row[k] for k in ("uid", "instruction", "text", "lang")}

            try:
                # Step 1: generate v1 tokens (chat model + style-instruction prompt)
                v1_tokens = _gen_v1_tokens(
                    chat_model, instruction, text, lang,
                    max_new_tokens=args.v1_max_new_tokens,
                    temperature=args.v1_temperature,
                )
                if not v1_tokens:
                    raise ValueError("chat model produced no audio tokens")

                # Step 2: vocode to temp wav (shared by critic + CLSP + ASR)
                row_prompt_wav = select_prompt_wav(instruction)
                tmp_wav = _vocode_to_tmp(token2wav, v1_tokens, row_prompt_wav)
                try:
                    # Step 3: run LALM critic
                    critique = make_critique(
                        chat_model, instruction, text, tmp_wav,
                        lang=lang, variant="chat",
                        max_new_tokens=args.critic_max_new_tokens,
                        temperature=args.critic_temperature,
                    )
                    # Step 4: precompute v1 CLSP baseline (same wav as critic)
                    clsp_v1 = clsp_score(clsp_model, clsp_device, tmp_wav, instruction)
                    # Step 5: precompute v1 WER baseline (ASR vs reference text)
                    hyp = transcribe(asr_model, asr_proc, to_mono_16k(tmp_wav),
                                     lang, clsp_device)
                    wer_v1 = wer_fn(hyp, text, lang)
                finally:
                    os.unlink(tmp_wav)

                result["v1_tokens"] = v1_tokens
                result["critique"] = critique
                result["clsp_v1"] = clsp_v1
                result["wer_v1"] = wer_v1
                result["clsp_v1_reference"] = Path(row_prompt_wav).name
                n_ok += 1

            except Exception as e:
                traceback.print_exc()
                result["v1_tokens"] = None
                result["critique"] = None
                result["clsp_v1"] = None
                result["wer_v1"] = None
                result["error"] = f"{type(e).__name__}: {e}"
                n_err += 1

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    total = n_ok + n_err
    print(f"[build_twohop] done: {n_ok}/{total} ok, {n_err} errors → {output_path}")

    # Filter out incomplete/high-WER rows so the RL loader receives the dataset
    # described in the paper (random 1k is selected above; WER(v1) <= 0.10).
    _filter_rows(output_path, args.max_wer_v1, selected_uids)


def _filter_rows(path: Path, max_wer_v1: float,
                 selected_uids: set[str] | None = None) -> None:
    """Remove incomplete rows and, when enabled, high-WER first passes."""
    rows = []
    n_incomplete = n_high_wer = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if selected_uids is not None and r.get("uid") not in selected_uids:
                    continue
                complete = (
                    bool(r.get("v1_tokens"))
                    and r.get("critique") is not None
                    and r.get("clsp_v1") is not None
                    and r.get("wer_v1") is not None
                )
                if not complete:
                    n_incomplete += 1
                    continue
                if max_wer_v1 >= 0 and float(r["wer_v1"]) > max_wer_v1:
                    n_high_wer += 1
                    continue
                rows.append(r)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"[build_twohop] kept {len(rows)} rows in {path} "
        f"(incomplete={n_incomplete}, wer_v1>{max_wer_v1}={n_high_wer})"
    )


if __name__ == "__main__":
    main()
