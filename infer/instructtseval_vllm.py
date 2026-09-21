"""vLLM client for InstructTTSEval.

Produces a JSONL with {id, text, APS/DSD/RP each {instruction, gen_path}}
that the benchmark's Gemini judge and this repository's CLSP evaluator consume.
The synthesis loop:

  - hits a Stepfun-vLLM OpenAI-compatible server (chat completions) instead
    of loading the LM in-process,
  - issues N requests concurrently (asyncio + httpx) for continuous batching,
  - decodes the returned <audio_NNNN> tokens to int codes and runs token2wav
    locally to write wavs.

Server (must be the Stepfun vLLM fork — plain vLLM does NOT speak
step_audio_2's audio tokenizer / chat template):
    bash scripts/infer/run_vllm_server.singularity.sh   # see that script for flags
    # or for an all-in-one Slurm job: sbatch scripts/infer/run_onehop_infer_zeroshot.sbatch


Usage:
    python infer/instructtseval_vllm.py \\
        --split en \\
        --output_jsonl out/en/results.jsonl \\
        --audio_dir   out/en/gen_wav \\
        --api_url     http://localhost:8000/v1/chat/completions \\
        --model_name  step-audio-2-mini \\
        --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import traceback
import wave
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm as atqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
STEP_DIR = REPO_ROOT / "Step-Audio2"
sys.path.insert(0, str(STEP_DIR))
# Repo root on path so `speaker_rag` (optional --speaker_rag) imports.
sys.path.insert(0, str(REPO_ROOT))

# --- torchcodec bypass (needed only for token2wav's torchaudio.save) ---
import torch as _torch  # noqa: E402
import torchaudio as _torchaudio  # noqa: E402
import soundfile as _sf  # noqa: E402


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
# --- end bypass ---

AUDIO_TOKEN_VOCAB_SIZE = 6561
AUDIO_TOKEN_RE = re.compile(r"<audio_(\d+)>")
# Offset between audio token code <audio_N> and its raw LM token ID.
# Step-Audio-2 audio token range.
AUDIO_TOKEN_RAW_ID_OFFSET = 151696
TASKS = ("APS", "DSD", "RP")

_TEMPLATES = {
    "en": "{instruction}\nRead the following text aloud in the speaking style described above.\n",
    "zh": "{instruction}\n请按照以上描述的风格朗读下面的文字。\n",
}


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else str(STEP_DIR / path)


def _infer_lang(row_id: str, fallback: str) -> str:
    if isinstance(row_id, str):
        if row_id.startswith("en"):
            return "en"
        if row_id.startswith("zh"):
            return "zh"
    return fallback


def _shard_path(path: Path, num_shards: int, shard_index: int) -> Path:
    if num_shards <= 1:
        return path
    return path.with_suffix(f".shard{shard_index}of{num_shards}{path.suffix}")


def _build_messages_chat(instruction: str, text: str, lang: str):
    """Chat variant: role-tagged turns + <tts_start> assistant prefill.

    The vLLM chat-completions handler in the Stepfun fork looks for the prefilled
    <tts_start> and continues with audio tokens (eot=False -> continue_final_message).
    """
    template = _TEMPLATES.get(lang, _TEMPLATES["en"])
    sys_msg = template.format(instruction=instruction.strip()).rstrip()
    return [
        {"role": "system", "content": sys_msg},
        {"role": "human", "content": [{"type": "text", "text": text}]},
        # eot=False -> stepaudio2vllm client maps this to continue_final_message=True
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]


def _normalize_chat_messages(messages):
    """Mirror stepaudio2vllm.StepAudio2.apply_chat_template: pass through, but
    pop the trailing assistant prefill and set continue_final_message accordingly.

    Returns (messages_for_payload, continue_final_message: bool, add_generation_prompt: bool)
    """
    msgs = list(messages)
    last = msgs[-1]
    if last.get("role") == "assistant" and last.get("content") is None:
        msgs.pop()
        return msgs, False, True
    if last.get("eot", True):
        return msgs, False, True
    return msgs, True, False


def _extract_audio_codes(message_obj) -> list[int]:
    """Pull <audio_NNNN> tokens out of a chat-completions message dict.

    Stepfun vLLM puts them either in message['tts_content']['tts_audio'] (the
    structured field) or inline in message['content'] (fallback)."""
    audio_str = None
    tts = message_obj.get("tts_content") if isinstance(message_obj, dict) else None
    if tts:
        audio_str = tts.get("tts_audio")
    if not audio_str:
        audio_str = message_obj.get("content") or ""
    codes = [int(m) for m in AUDIO_TOKEN_RE.findall(audio_str or "")]
    return [c for c in codes if c < AUDIO_TOKEN_VOCAB_SIZE]


def _load_rows(args) -> list[dict]:
    if args.input_jsonl:
        rows = []
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    from huggingface_hub import hf_hub_download  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    parquet_path = hf_hub_download(
        repo_id="CaasiHUANG/InstructTTSEval",
        filename=f"{args.split}.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path)
    keep = [c for c in table.column_names if c in ("id", "text", *TASKS)]
    table = table.select(keep)

    rows = []
    for d in table.to_pylist():
        row = {"id": d.get("id"), "text": d.get("text")}
        for t in TASKS:
            v = d.get(t)
            if isinstance(v, str) and v.strip():
                row[t] = {"instruction": v}
            elif isinstance(v, dict) and "instruction" in v:
                row[t] = {"instruction": v["instruction"]}
        rows.append(row)
    return rows


async def _call_vllm(
    client: httpx.AsyncClient,
    api_url: str,
    model_name: str,
    messages: list,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    extra_sampling: dict | None = None,
) -> dict:
    msgs, cont, add_gen = _normalize_chat_messages(messages)
    payload = {
        "model": model_name,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "skip_special_tokens": False,
        "stream": False,
        "continue_final_message": cont,
        "add_generation_prompt": add_gen,
    }
    if extra_sampling:
        payload.update(extra_sampling)

    r = await client.post(api_url, json=payload,
                          headers={"Content-Type": "application/json"})
    r.raise_for_status()
    body = r.json()
    return body["choices"][0]["message"]


async def _process_one(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    args,
    token2wav,
    prompt_wav: str,
    audio_dir: Path,
    rid: str,
    task: str,
    instruction: str,
    text: str,
    lang: str,
    select_ref=None,
) -> tuple[str, str, dict]:
    """Generate one (rid, task) wav. Returns (rid, task, entry) for result merge."""
    wav_path = audio_dir / f"{rid}_{task}.wav"
    tokens_path = wav_path.with_suffix(".tokens.json") if args.save_tokens else None
    entry = {"instruction": instruction, "gen_path": str(wav_path)}
    if tokens_path is not None:
        entry["tokens_path"] = str(tokens_path)

    if args.skip_existing and wav_path.exists() and (
        tokens_path is None or tokens_path.exists()
    ):
        return rid, task, entry

    messages = _build_messages_chat(instruction, text, lang)
    async with sem:
        try:
            msg = await _call_vllm(
                client, args.api_url, args.model_name, messages,
                max_tokens=args.max_new_tokens, temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            audio_codes = _extract_audio_codes(msg)
            if not audio_codes:
                return rid, task, {**entry, "gen_path": None, "error": "empty audio"}

            # Speaker RAG: pick the reference whose voice best matches this
            # instruction (CLSP cosine). Gated by `sem` so the GPU encode +
            # vocoder don't pile up. Falls back to the fixed --prompt_wav.
            this_prompt_wav = prompt_wav
            if select_ref is not None:
                res = select_ref(instruction)
                this_prompt_wav = res.reference.wav
                entry["rag_reference"] = res.reference.name
                entry["rag_clsp_score"] = res.score
                entry["rag_desired_gender"] = res.desired_gender
                entry["rag_method"] = res.method

            # token2wav is CPU-ish + GPU vocoder; safe to call from the asyncio
            # loop thread since requests are gated by `sem` (low parallelism).
            wav_bytes = token2wav(audio_codes, prompt_wav=this_prompt_wav)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)

            if tokens_path is not None:
                # Reconstruct raw LM token IDs from extracted codes. Mirrors the
                # offline audio-token filter; v2 splices these back via
                # {"type": "token", "token": ...}.
                raw_token_ids = [c + AUDIO_TOKEN_RAW_ID_OFFSET for c in audio_codes]
                tokens_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tokens_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "raw_token_ids": raw_token_ids,
                            "audio_codes": audio_codes,
                            "decoded_text": msg.get("content") or "",
                        },
                        f,
                        ensure_ascii=False,
                    )
            return rid, task, entry
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return rid, task, {**entry, "gen_path": None, "error": str(e)}


async def _run(args, rows: list[dict]) -> None:
    from token2wav import Token2wav  # type: ignore

    model_dir = _resolve(args.token2wav_model_path or args.model_path_for_token2wav)
    token2wav = Token2wav(os.path.join(model_dir, "token2wav"))

    prompt_wav = _resolve(args.prompt_wav)
    audio_dir = Path(args.audio_dir).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Speaker RAG: load CLSP + reference embeddings once, expose a select_ref
    # closure. None when --speaker_rag is off (keeps the fixed prompt_wav path).
    select_ref = None
    if args.speaker_rag:
        import torch  # noqa: PLC0415
        from speaker_rag import (  # noqa: PLC0415
            DEFAULT_REFERENCES, embed_references, select_reference,
        )
        from clsp_utils import load_clsp  # type: ignore  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clsp_model = load_clsp(args.clsp_model_id, device)
        references = DEFAULT_REFERENCES
        ref_emb = embed_references(references, clsp_model, device,
                                   cache_path=args.speaker_emb_cache)

        def select_ref(instruction: str):
            return select_reference(instruction, clsp_model, device,
                                    references, ref_emb)

        names = ", ".join(r.name for r in references)
        print(f"[vllm] speaker_rag ON: {len(references)} references [{names}]")

    default_lang = args.language_hint or args.split or "en"
    out_path = _shard_path(Path(args.output_jsonl), args.num_shards, args.shard_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build (rid, task, instruction, text, lang) tasks
    jobs = []
    row_index: dict[str, dict] = {}
    for row in rows:
        rid = row["id"]
        text = row["text"]
        lang = _infer_lang(rid, default_lang)
        row_index[rid] = {"id": rid, "text": text}
        for task in args.tasks:
            block = row.get(task)
            if not block or "instruction" not in block:
                continue
            jobs.append((rid, task, block["instruction"], text, lang))

    print(f"[vllm] {len(rows)} rows -> {len(jobs)} synth calls; "
          f"concurrency={args.concurrency}")

    sem = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.request_timeout, connect=30.0)
    limits = httpx.Limits(max_connections=args.concurrency * 2,
                          max_keepalive_connections=args.concurrency)
    results_by_row: dict[str, dict] = {rid: dict(row_index[rid]) for rid in row_index}

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        coros = [
            _process_one(sem, client, args, token2wav, prompt_wav, audio_dir,
                         rid, task, instruction, text, lang, select_ref=select_ref)
            for (rid, task, instruction, text, lang) in jobs
        ]
        for fut in atqdm.as_completed(coros, total=len(coros), desc="synth"):
            rid, task, entry = await fut
            results_by_row[rid][task] = entry

    with open(out_path, "w", encoding="utf-8") as out_f:
        # Preserve input row order.
        for rid in row_index:
            out_f.write(json.dumps(results_by_row[rid], ensure_ascii=False) + "\n")
    print(f"[vllm] wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input_jsonl")
    src.add_argument("--split", choices=["en", "zh"], default="en")

    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--audio_dir", required=True)

    p.add_argument("--api_url", default="http://localhost:8000/v1/chat/completions",
                   help="Stepfun vLLM OpenAI-compatible chat completions endpoint.")
    p.add_argument("--model_name", default="step-audio-2-mini",
                   help="Matches --served-model-name on the server. If serving a LoRA "
                        "via --lora-modules name=path, set this to that LoRA name to "
                        "route requests through the adapter.")

    # Kept for backward compat with run_instructtseval.sh callers (they set
    # --model_path even though vLLM serves it server-side). We only need it
    # to locate token2wav weights locally.
    p.add_argument("--model_path_for_token2wav", default="Step-Audio-2-mini",
                   help="Local Step-Audio-2 snapshot dir whose token2wav/ subdir we "
                        "use to decode audio tokens -> wav. Server-side model is "
                        "selected by --model_name.")
    p.add_argument("--token2wav_model_path", default=None,
                   help="Alias of --model_path_for_token2wav (takes precedence).")
    p.add_argument("--prompt_wav", default="assets/default_male.wav",
                   help="Fixed speaker reference (used when --speaker_rag is off, "
                        "or as fallback).")
    p.add_argument("--speaker_rag", action="store_true",
                   help="Per-utterance, pick the reference whose voice best matches "
                        "the instruction by CLSP cosine (speaker_rag package), "
                        "instead of the fixed --prompt_wav.")
    p.add_argument("--clsp_model_id", default="yfyeung/CLSP",
                   help="CLSP model id used for speaker retrieval (--speaker_rag).")
    p.add_argument("--speaker_emb_cache",
                   default=str(REPO_ROOT / "speaker_rag" / "references.emb.pt"),
                   help="Where to cache reference audio embeddings.")
    p.add_argument("--language_hint", default=None, choices=[None, "en", "zh"])

    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)

    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=-1,
                   help="vLLM top-k cutoff; -1 disables it (paper/default).")
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--save_tokens", action="store_true",
                   help="Dump <rid>_<task>.tokens.json next to each wav with "
                        "raw_token_ids + audio_codes for later replay.")

    p.add_argument("--concurrency", type=int, default=16,
                   help="Number of in-flight HTTP requests. Should be <= the server's "
                        "--max-num-seqs (default 32 in the Singularity launcher).")
    p.add_argument("--request_timeout", type=float, default=300.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        sys.exit("Require num_shards >= 1 and 0 <= shard_index < num_shards.")

    rows = _load_rows(args)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = rows[args.shard_index :: args.num_shards]
        print(f"[vllm] shard {args.shard_index}/{args.num_shards}: {len(rows)} rows")

    asyncio.run(_run(args, rows))


if __name__ == "__main__":
    main()
