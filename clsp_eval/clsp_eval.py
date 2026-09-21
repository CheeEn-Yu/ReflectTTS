"""CLSP evaluator for InstructTTSEval outputs.

Scores pre-generated wavs produced by the inference scripts against their
styling instructions via CLSP (Contrastive Language-Speech Pre-training).

Two input modes:
  1. Single pair (smoke test):
       python clsp_eval/clsp_eval.py --audio path/to.wav --instruction "..."
  2. Batch over an InstructTTSEval results JSONL:
       python clsp_eval/clsp_eval.py \\
           --input_jsonl out/en_results.jsonl \\
           --output_jsonl out/en_clsp.jsonl

Batch input schema (inference output):
    {"id": "...", "text": "...",
     "APS": {"instruction": "...", "gen_path": "..."},
     "DSD": {"instruction": "...", "gen_path": "..."},
     "RP":  {"instruction": "...", "gen_path": "..."}}

Adds `"clsp_score": <float>` to each task block and prints per-task means.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from statistics import mean

import torch
from tqdm import tqdm

# torchcodec bypass — defensive even though we use soundfile directly,
# because CLSP's forward may call torchaudio internally.
import torchaudio as _torchaudio  # noqa: E402
import soundfile as _sf  # noqa: E402


def _sf_load(uri, *_, **__):
    data, sr = _sf.read(uri, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), sr


def _sf_save(uri, src, sample_rate, *, format=None, channels_first=True, **__):
    if isinstance(src, torch.Tensor):
        src = src.detach().cpu().numpy()
    if src.ndim == 2 and channels_first:
        src = src.T
    _sf.write(uri, src, sample_rate, format=(format.upper() if format else None))


_torchaudio.load = _sf_load
_torchaudio.save = _sf_save

from clsp_utils import (  # noqa: E402
    load_asr,
    load_clsp,
    score_batch,
    score_wav,
    to_mono_16k,
    transcribe,
    wer as wer_score,
)

TASKS = ("APS", "DSD", "RP")


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _infer_lang(row_id: str | None, fallback: str = "en") -> str:
    if row_id and row_id.startswith("zh"):
        return "zh"
    if row_id and row_id.startswith("en"):
        return "en"
    return fallback


def _shard_path(path: Path, num_shards: int, shard_index: int) -> Path:
    """When sharding, give each shard a distinct output file so concurrent
    Slurm-array jobs don't clobber each other. num_shards==1 -> unchanged."""
    if num_shards <= 1:
        return path
    return path.with_suffix(f".shard{shard_index}of{num_shards}{path.suffix}")


def _run_batch(args, model, asr, asr_processor, device) -> None:
    in_path = Path(args.input_jsonl)
    out_path = _shard_path(Path(args.output_jsonl), args.num_shards, args.shard_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_jsonl(in_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = rows[args.shard_index :: args.num_shards]
        print(f"[clsp_eval] shard {args.shard_index}/{args.num_shards}: {len(rows)} rows")

    per_task_scores: dict[str, list[float]] = {t: [] for t in args.tasks}
    per_task_wer: dict[str, list[float]] = {t: [] for t in args.tasks}
    expected_per_task = {
        task: sum(
            1 for row in rows
            if isinstance(row.get(task), dict) and row[task].get("instruction")
        )
        for task in args.tasks
    }
    report_expected = {
        task: (args.expected_per_task
               if args.expected_per_task is not None else expected_per_task[task])
        for task in args.tasks
    }

    # --- Phase 1: load all wavs, build result skeletons ---
    row_results: list[dict] = []
    valid_items: list[dict] = []  # {wav_16k, instruction, ref_text, lang, task, entry}

    for row in tqdm(rows, desc="load"):
        ref_text = row.get("text")
        lang = _infer_lang(row.get("id"))
        result = {"id": row.get("id"), "text": ref_text}
        row_results.append(result)
        for task in args.tasks:
            block = row.get(task)
            if not block:
                continue
            instr = block.get("instruction")
            wav_path = block.get("gen_path")
            entry = dict(block)
            if not wav_path or not instr:
                entry["clsp_score"] = None
                if asr is not None:
                    entry["wer"] = None
                entry.setdefault("error", "missing gen_path or instruction")
                result[task] = entry
                continue
            try:
                wav_16k = to_mono_16k(wav_path)
                entry["clsp_score"] = None  # filled in phase 2
                result[task] = entry
                valid_items.append({
                    "wav_16k": wav_16k,
                    "instruction": instr,
                    "ref_text": ref_text,
                    "lang": lang,
                    "task": task,
                    "entry": entry,  # mutable reference into result
                })
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                entry["clsp_score"] = None
                if asr is not None:
                    entry["wer"] = None
                entry["error"] = traceback.format_exc(limit=1)
                result[task] = entry

    # --- Phase 2: batch CLSP scoring ---
    bs = args.batch_size
    for i in tqdm(range(0, len(valid_items), bs), desc="clsp"):
        batch = valid_items[i : i + bs]
        scores = score_batch(model, device,
                             [it["wav_16k"] for it in batch],
                             [it["instruction"] for it in batch])
        for it, s in zip(batch, scores):
            it["entry"]["clsp_score"] = float(s)
            per_task_scores[it["task"]].append(float(s))

    # --- Phase 3: ASR + WER (sequential; handles mixed zh/en per item) ---
    if asr is not None:
        for it in tqdm(valid_items, desc="asr"):
            try:
                hyp = transcribe(asr, asr_processor, it["wav_16k"], it["lang"], device)
                w = wer_score(hyp, it["ref_text"] or "", it["lang"])
                it["entry"]["wer"] = float(w)
                it["entry"]["asr_hyp"] = hyp
                per_task_wer[it["task"]].append(float(w))
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                it["entry"]["wer"] = None
                it["entry"]["error"] = traceback.format_exc(limit=1)

    # --- Phase 4: write JSONL ---
    with open(out_path, "w", encoding="utf-8", buffering=1) as out_f:
        for result in row_results:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"[clsp_eval] wrote {out_path}")

    # --- Phase 5: stats + optional final_result.txt ---
    stat_lines: list[str] = []
    for task in args.tasks:
        scores = per_task_scores[task]
        wers = per_task_wer[task]
        clsp_str = f"clsp_mean={mean(scores):.4f}" if scores else "clsp_mean=n/a"
        wer_str = f"wer_mean={mean(wers):.4f}" if wers else (
            "wer=skipped" if asr is None else "wer_mean=n/a")
        line = (f"[clsp_eval] {task}: n={len(scores)}/{report_expected[task]} "
                f"{clsp_str} {wer_str}")
        stat_lines.append(line)
        print(line)

    all_scores = [s for t in args.tasks for s in per_task_scores[t]]
    all_wers   = [w for t in args.tasks for w in per_task_wer[t]]
    overall_clsp = f"clsp_mean={mean(all_scores):.4f}" if all_scores else "clsp_mean=n/a"
    overall_wer  = f"wer_mean={mean(all_wers):.4f}"   if all_wers   else (
        "wer=skipped" if asr is None else "wer_mean=n/a")
    expected_overall = sum(report_expected.values())
    overall_line = (f"[clsp_eval] OVERALL: n={len(all_scores)}/{expected_overall} "
                    f"{overall_clsp} {overall_wer}")
    stat_lines.append(overall_line)
    print(overall_line)

    incomplete = {
        task: (len(per_task_scores[task]), report_expected[task])
        for task in args.tasks
        if (expected_per_task[task] != report_expected[task]
            or len(per_task_scores[task]) != report_expected[task])
    } if args.require_complete else {}
    if incomplete:
        details = ", ".join(
            f"{task}={actual}/{expected}" for task, (actual, expected) in incomplete.items()
        )
        invalid_line = f"[clsp_eval] INVALID_INCOMPLETE: {details}"
        stat_lines.append(invalid_line)
        print(invalid_line)

    if args.result_dir:
        result_dir = Path(args.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / "final_result.txt"
        result_file.write_text("\n".join(stat_lines) + "\n", encoding="utf-8")
        print(f"[clsp_eval] final result -> {result_file}")

    if incomplete:
        raise RuntimeError(
            "incomplete evaluation; refusing to report a paper-comparable mean: " + details
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_id", default="yfyeung/CLSP")
    p.add_argument("--device", default=None)
    p.add_argument("--audio", help="Single wav path (single-pair mode).")
    p.add_argument("--instruction", help="Instruction text (single-pair mode).")
    p.add_argument("--input_jsonl", help="Batch input JSONL.")
    p.add_argument("--output_jsonl", help="Batch output JSONL.")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip_wer", action="store_true",
                   help="Score CLSP only; skip Whisper ASR + WER (faster).")
    p.add_argument("--asr_model_id", default="openai/whisper-large-v3",
                   help="HF id for the Whisper ASR model used to measure WER.")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Split the input rows across N shards (round-robin). Each "
                        "shard writes results.shard<i>of<N>.jsonl; merge afterward.")
    p.add_argument("--shard_index", type=int, default=0,
                   help="Which shard (0-based) this process handles.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for CLSP inference (audio is padded to max length in batch).")
    p.add_argument("--result_dir", default=None,
                   help="If set, write per-task + overall stats to <result_dir>/final_result.txt.")
    p.add_argument("--require_complete", action="store_true",
                   help="Fail after writing diagnostics if any requested task with an "
                        "instruction lacks a valid CLSP score.")
    p.add_argument("--expected_per_task", type=int, default=None,
                   help="Expected item count for every requested task (paper English "
                        "evaluation: 1000). Used with --require_complete.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.audio and (args.input_jsonl or args.output_jsonl):
        sys.exit("Use either --audio/--instruction or --input_jsonl/--output_jsonl, not both.")
    if args.audio and not args.instruction:
        sys.exit("--audio requires --instruction.")
    if args.input_jsonl and not args.output_jsonl:
        sys.exit("--input_jsonl requires --output_jsonl.")
    if not args.audio and not args.input_jsonl:
        sys.exit("Pass either --audio + --instruction or --input_jsonl + --output_jsonl.")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        sys.exit("Require num_shards >= 1 and 0 <= shard_index < num_shards.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[clsp_eval] loading {args.model_id} on {device} ...")
    model = load_clsp(args.model_id, device)

    if args.audio:
        s = score_wav(model, device, to_mono_16k(args.audio), args.instruction)
        print(f"clsp_score={s:.4f}")
        print(json.dumps({"audio": args.audio, "instruction": args.instruction,
                          "clsp_score": s}, ensure_ascii=False))
        return

    asr, asr_processor = (None, None)
    if not args.skip_wer:
        print(f"[clsp_eval] loading ASR {args.asr_model_id} on {device} ...")
        asr, asr_processor = load_asr(args.asr_model_id, device)

    _run_batch(args, model, asr, asr_processor, device)


if __name__ == "__main__":
    main()
