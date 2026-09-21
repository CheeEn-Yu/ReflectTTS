"""Full 3-stage inference via vLLM sidecar — ALL stages use the adapter.

  Stage 1 (v1):      instruction + text → single-pass audio  (adapter)
  Stage 2 (critique): instruction + v1 tokens → self-critique in <think>  (adapter)
  Stage 3 (v2):      prompt + critique + bridge → refined audio  (adapter)

This script generates v1 WITH the adapter so all three stages reflect the
RL-trained policy, and self-generates the critique — so it needs no precomputed
reuse data. Outputs v1.jsonl + v2.jsonl, both consumable by clsp_eval/clsp_eval.py
unchanged.

Run via scripts/infer/run_twohop_infer.sbatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from train.rl.train_rl import _resolve_step
from train.rl.prompt import build_singlepass_prompt_chat, build_twohop_selfcritique_chat
from train.rl.rollout import (
    _build_prompt_ids,
    _trim_at_eos,
    AUDIO_TOKEN_OFFSET,
    AUDIO_TOKEN_VOCAB_SIZE,
    TEXT_TOKEN_MAX,
)
from train.rl_vllm.train_rl_vllm import SidecarClient

TASKS = ("APS", "DSD", "RP")


def _infer_lang(rid: str, default: str = "en") -> str:
    s = str(rid or "")
    if s.startswith("zh"):
        return "zh"
    if s.startswith("en"):
        return "en"
    return default


def _load_rows(split: str = "en", input_jsonl: str | None = None) -> list[dict]:
    if input_jsonl:
        rows = []
        with open(input_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    parquet_path = hf_hub_download(
        repo_id="CaasiHUANG/InstructTTSEval",
        filename=f"{split}.parquet",
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


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, help="sidecar file-RPC dir")
    ap.add_argument("--output_dir", required=True, help="output dir for v1.jsonl, v2.jsonl, wavs")
    ap.add_argument("--split", default="en", help="InstructTTSEval HF split")
    ap.add_argument("--input_jsonl", default=None, help="local JSONL (overrides --split)")
    ap.add_argument("--base", default="Step-Audio-2-mini")
    ap.add_argument("--prompt_wav", default="assets/default_male.wav",
                    help="Speaker reference for token2wav; used when --speaker_rag is off.")
    ap.add_argument("--speaker_rag", action="store_true",
                    help="Per-instruction, pick the token2wav reference whose voice best "
                         "matches the instruction (gender filter + CLSP, speaker_rag pkg) "
                         "instead of the fixed --prompt_wav.")
    ap.add_argument("--speaker_emb_cache", default=None,
                    help="Cache path for reference audio embeddings (--speaker_rag).")
    ap.add_argument("--clsp_model", default="yfyeung/CLSP",
                    help="CLSP model id used to select the reference speaker.")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=-1,
                    help="vLLM top-k cutoff; -1 disables it (paper/default).")
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--critic_temperature", type=float, default=None)
    ap.add_argument("--max_think_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout_chunk", type=int, default=128)
    ap.add_argument("--v2_len_cap", type=float, default=None,
                    help="reject v2 whose audio length > cap * v1 length and resample "
                         "(inference-time guard against off-text drift); None disables.")
    ap.add_argument("--v2_max_retries", type=int, default=4,
                    help="max resamples per item when over --v2_len_cap (keeps shortest).")
    ap.add_argument("--no_stop", action="store_true", default=False)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    from stepaudio2 import StepAudio2  # type: ignore
    from token2wav import Token2wav  # type: ignore

    base_dir = _resolve_step(args.base)
    prompt_wav = _resolve_step(args.prompt_wav)
    crit_temp = args.critic_temperature if args.critic_temperature is not None else args.temperature

    # Optional speaker RAG: per instruction, pick the reference whose voice best
    # matches it (gender filter + CLSP). Returns the chosen wav, or the fixed
    # prompt_wav when disabled. Selection is memoised per instruction.
    select_prompt_wav = lambda _instruction: prompt_wav  # noqa: E731 (disabled default)
    if args.speaker_rag:
        from train.rl.reward import clsp_load
        from speaker_rag import DEFAULT_REFERENCES, embed_references, select_reference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[pipeline] loading CLSP for speaker RAG: {args.clsp_model}", flush=True)
        clsp = clsp_load(args.clsp_model, device=device)
        references = DEFAULT_REFERENCES
        ref_embeddings = embed_references(references, clsp, device,
                                          cache_path=args.speaker_emb_cache)
        names = ", ".join(r.name for r in references)
        print(f"[pipeline] speaker_rag ON: {len(references)} references [{names}]", flush=True)
        _ref_cache: dict[str, str] = {}

        def select_prompt_wav(instruction: str) -> str:  # noqa: F811
            key = instruction or ""
            hit = _ref_cache.get(key)
            if hit is None:
                hit = select_reference(key, clsp, device, references, ref_embeddings).reference.wav
                _ref_cache[key] = hit
            return hit

    client = SidecarClient(args.workdir)
    print("[pipeline] waiting for sidecar ready ...", flush=True)
    client.wait_ready()
    print("[pipeline] sidecar ready", flush=True)

    print(f"[pipeline] loading StepAudio2 (tokenizer) + token2wav from {base_dir}", flush=True)
    model = StepAudio2(base_dir)
    token2wav = Token2wav(Path(base_dir, "token2wav").as_posix())
    eos = int(model.eos_token_id)
    bridge_ids = model.llm_tokenizer(
        "\n</think>\n<tts_start>", return_tensors="pt", add_special_tokens=False
    )["input_ids"].squeeze(0)

    out_dir = Path(args.output_dir).resolve()
    v1_wav_dir = out_dir / "v1_wav"
    v2_wav_dir = out_dir / "v2_wav"
    v1_wav_dir.mkdir(parents=True, exist_ok=True)
    v2_wav_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.split, args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"[pipeline] {len(rows)} rows x {len(args.tasks)} tasks", flush=True)

    # ---- Build work items ----
    v1_results: dict[str, dict] = {}
    v2_results: dict[str, dict] = {}
    order: list[str] = []
    work: list[dict] = []
    for row in rows:
        rid = row.get("id")
        text = row.get("text", "")
        lang = _infer_lang(rid)
        if rid not in v1_results:
            v1_results[rid] = {"id": rid, "text": text}
            v2_results[rid] = {"id": rid, "text": text}
            order.append(rid)
        for task in args.tasks:
            block = row.get(task) or {}
            instruction = block.get("instruction")
            if not instruction:
                continue
            v1_wav = v1_wav_dir / f"{rid}_{task}.wav"
            v2_wav = v2_wav_dir / f"{rid}_{task}_v2.wav"
            try:
                v1_prompt = build_singlepass_prompt_chat(instruction, text, lang)
                v1_prompt_ids = _build_prompt_ids(model, v1_prompt).squeeze(0).tolist()
            except Exception as e:  # noqa: BLE001
                v1_results[rid][task] = {"instruction": instruction, "gen_path": None, "error": str(e)}
                v2_results[rid][task] = {"instruction": instruction, "gen_path": None, "error": str(e)}
                continue
            item_prompt_wav = select_prompt_wav(instruction)
            work.append({
                "rid": rid, "task": task, "instruction": instruction,
                "text": text, "lang": lang, "prompt_wav": item_prompt_wav,
                "v1_wav": str(v1_wav), "v2_wav": str(v2_wav),
                "v1_prompt_ids": v1_prompt_ids,
                "v1_tokens_path": str(v1_wav.with_suffix(".tokens.json")),
                "v2_tokens_path": str(v2_wav.with_suffix(".tokens.json")),
            })
    print(f"[pipeline] {len(work)} items to generate", flush=True)

    # ==== Stage 1: v1 (single-pass) ====
    print(f"[pipeline] Stage 1: generating v1 ({len(work)} items)", flush=True)
    v1_sampling = dict(n=1, temperature=args.temperature, top_p=args.top_p,
                       top_k=args.top_k, max_tokens=args.max_new_tokens,
                       repetition_penalty=args.repetition_penalty, eos=eos)
    v1_ok = 0
    for start in tqdm(range(0, len(work), args.rollout_chunk), desc="v1"):
        chunk = work[start:start + args.rollout_chunk]
        sampling = dict(v1_sampling, seed=args.seed + start)
        comps = client.rollout([w["v1_prompt_ids"] for w in chunk], sampling)
        for w, comp in zip(chunk, comps):
            gen = torch.tensor(comp[0], dtype=torch.long)
            gen, _ = _trim_at_eos(gen, eos)
            audio_codes = [
                t - AUDIO_TOKEN_OFFSET for t in gen.tolist()
                if t >= AUDIO_TOKEN_OFFSET and (t - AUDIO_TOKEN_OFFSET) < AUDIO_TOKEN_VOCAB_SIZE
            ]
            if len(audio_codes) < 5:
                w["v1_failed"] = True
                v1_results[w["rid"]][w["task"]] = {
                    "instruction": w["instruction"], "gen_path": None,
                    "error": f"too_few_audio_codes:{len(audio_codes)}"}
                continue
            try:
                wav_bytes = token2wav(audio_codes, prompt_wav=w["prompt_wav"])
                with open(w["v1_wav"], "wb") as f:
                    f.write(wav_bytes)
                raw_audio_ids = [
                    t for t in gen.tolist()
                    if AUDIO_TOKEN_OFFSET <= t < AUDIO_TOKEN_OFFSET + AUDIO_TOKEN_VOCAB_SIZE
                ]
                json.dump({"raw_token_ids": raw_audio_ids, "audio_codes": audio_codes},
                          open(w["v1_tokens_path"], "w", encoding="utf-8"), ensure_ascii=False)
                w["v1_raw_ids"] = raw_audio_ids
                v1_results[w["rid"]][w["task"]] = {
                    "instruction": w["instruction"],
                    "gen_path": w["v1_wav"],
                    "tokens_path": w["v1_tokens_path"],
                    "prompt_wav": w["prompt_wav"],
                }
                v1_ok += 1
            except Exception as e:  # noqa: BLE001
                w["v1_failed"] = True
                v1_results[w["rid"]][w["task"]] = {
                    "instruction": w["instruction"], "gen_path": None, "error": f"vocode:{e}"}
    print(f"[pipeline] Stage 1 done: {v1_ok}/{len(work)} v1 wavs", flush=True)

    # Write v1.jsonl immediately so it can be scored even if later stages fail
    v1_jsonl = out_dir / "v1.jsonl"
    with open(v1_jsonl, "w", encoding="utf-8") as f:
        for rid in order:
            f.write(json.dumps(v1_results[rid], ensure_ascii=False) + "\n")
    print(f"[pipeline] wrote {v1_jsonl}", flush=True)

    # Filter to items where v1 succeeded
    work_v2 = [w for w in work if not w.get("v1_failed")]
    print(f"[pipeline] {len(work_v2)} items proceed to critique+v2", flush=True)

    # ==== Stage 2: self-critique ====
    print(f"[pipeline] Stage 2: generating critique ({len(work_v2)} items)", flush=True)
    for w in tqdm(work_v2, desc="crit-prompt"):
        messages = build_twohop_selfcritique_chat(
            w["instruction"], w["text"], w["lang"], v1_tokens=w["v1_raw_ids"]
        )
        w["p1_ids"] = _build_prompt_ids(model, messages).squeeze(0).tolist()

    crit_sampling = dict(n=1, temperature=crit_temp, top_p=args.top_p, top_k=args.top_k,
                         max_tokens=args.max_think_tokens,
                         repetition_penalty=args.repetition_penalty,
                         eos=eos, stop=["</think>"])
    for start in tqdm(range(0, len(work_v2), args.rollout_chunk), desc="critique"):
        chunk = work_v2[start:start + args.rollout_chunk]
        sampling = dict(crit_sampling, seed=args.seed + start + 10000)
        comps = client.rollout([w["p1_ids"] for w in chunk], sampling)
        for w, comp in zip(chunk, comps):
            c = torch.tensor(comp[0], dtype=torch.long)
            if c.numel() and c[-1] == eos:
                c = c[:-1]
            if any(AUDIO_TOKEN_OFFSET <= t < AUDIO_TOKEN_OFFSET + AUDIO_TOKEN_VOCAB_SIZE
                   for t in c.tolist()):
                w["crit_failed"] = True
                v2_results[w["rid"]][w["task"]] = {
                    "instruction": w["instruction"], "gen_path": None,
                    "error": "critic_emitted_audio_tokens",
                }
                continue
            w["crit_ids"] = c
            w["p2_ids"] = w["p1_ids"] + c.tolist() + bridge_ids.tolist()
            w["critique_text"] = model.llm_tokenizer.decode(
                [t for t in c.tolist() if t < TEXT_TOKEN_MAX], skip_special_tokens=False
            ).strip()
    print("[pipeline] Stage 2 done", flush=True)

    work_v2 = [w for w in work_v2 if not w.get("crit_failed")]
    print(f"[pipeline] {len(work_v2)} items proceed to v2", flush=True)

    # ==== Stage 3: v2 audio (optional length-cap rejection sampling) ====
    print(f"[pipeline] Stage 3: generating v2 ({len(work_v2)} items)", flush=True)
    v2_sampling = dict(n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                       max_tokens=args.max_new_tokens,
                       repetition_penalty=args.repetition_penalty, eos=eos)
    cap = args.v2_len_cap            # None disables; e.g. 1.5 = reject v2 > 1.5x v1
    retries = args.v2_max_retries if cap else 0

    def _over_cap(w) -> bool:
        if not cap:
            return False
        v1n = len(w.get("v1_raw_ids") or [])
        return v1n > 0 and len(w.get("_v2_codes", [])) > cap * v1n

    # Round 0 reproduces the original generation exactly (same seeds); later rounds
    # only resample items still over the cap, keeping the SHORTEST candidate seen.
    pending = list(work_v2)
    for r in range(retries + 1):
        desc = "v2" if r == 0 else f"v2-retry{r}"
        for start in tqdm(range(0, len(pending), args.rollout_chunk), desc=desc):
            chunk = pending[start:start + args.rollout_chunk]
            sampling = dict(v2_sampling, seed=args.seed + start + 20000 + r * 100000)
            comps = client.rollout([w["p2_ids"] for w in chunk], sampling)
            for w, comp in zip(chunk, comps):
                gen = torch.tensor(comp[0], dtype=torch.long)
                gen, _ = _trim_at_eos(gen, eos)
                audio_codes = [
                    t - AUDIO_TOKEN_OFFSET for t in gen.tolist()
                    if t >= AUDIO_TOKEN_OFFSET and (t - AUDIO_TOKEN_OFFSET) < AUDIO_TOKEN_VOCAB_SIZE
                ]
                if "_v2_codes" not in w or len(audio_codes) < len(w["_v2_codes"]):
                    w["_v2_codes"] = audio_codes
                    w["_v2_raw"] = [
                        t for t in gen.tolist()
                        if AUDIO_TOKEN_OFFSET <= t < AUDIO_TOKEN_OFFSET + AUDIO_TOKEN_VOCAB_SIZE
                    ]
        pending = [w for w in work_v2 if _over_cap(w)]
        if cap and pending and r < retries:
            print(f"[pipeline] length-cap round {r+1}/{retries}: {len(pending)} items still "
                  f">{cap}x v1, resampling", flush=True)
        if not pending:
            break
    if cap:
        still = sum(1 for w in work_v2 if _over_cap(w))
        print(f"[pipeline] length-cap: {still}/{len(work_v2)} items still >{cap}x v1 after "
              f"{retries} retries (kept shortest)", flush=True)

    v2_ok = 0
    for w in work_v2:
        audio_codes = w.get("_v2_codes", [])
        if len(audio_codes) < 5:
            v2_results[w["rid"]][w["task"]] = {
                "instruction": w["instruction"], "gen_path": None,
                "error": f"too_few_audio_codes:{len(audio_codes)}"}
            continue
        try:
            wav_bytes = token2wav(audio_codes, prompt_wav=w["prompt_wav"])
            with open(w["v2_wav"], "wb") as f:
                f.write(wav_bytes)
            json.dump({"raw_token_ids": w.get("_v2_raw", []), "audio_codes": audio_codes},
                      open(w["v2_tokens_path"], "w", encoding="utf-8"), ensure_ascii=False)
            v2_results[w["rid"]][w["task"]] = {
                "instruction": w["instruction"],
                "gen_path": w["v2_wav"],
                "v1_path": w["v1_wav"],
                "critique": w["critique_text"],
                "tokens_path": w["v2_tokens_path"],
                "prompt_wav": w["prompt_wav"],
            }
            v2_ok += 1
        except Exception as e:  # noqa: BLE001
            v2_results[w["rid"]][w["task"]] = {
                "instruction": w["instruction"], "gen_path": None, "error": f"vocode:{e}"}

    v2_jsonl = out_dir / "v2.jsonl"
    with open(v2_jsonl, "w", encoding="utf-8") as f:
        for rid in order:
            f.write(json.dumps(v2_results[rid], ensure_ascii=False) + "\n")
    print(f"[pipeline] Stage 3 done: {v2_ok}/{len(work_v2)} v2 wavs", flush=True)
    print(f"[pipeline] wrote {v2_jsonl}", flush=True)

    if not args.no_stop:
        client.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
