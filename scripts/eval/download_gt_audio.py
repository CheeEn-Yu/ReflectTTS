"""Download InstructTTSEval ground-truth audio and emit a CLSP-ready JSONL.

The dataset (`CaasiHUANG/InstructTTSEval`, arXiv:2506.16381) embeds the
reference (ground-truth) speech as a `reference_audio` column inside
`en.parquet` / `zh.parquet`. We pull the parquet directly via
`huggingface_hub.hf_hub_download` + `pyarrow` — same path
the inference scripts use. Reading the parquet directly also avoids decoding
the Audio column through torchcodec.

The HF Audio feature serializes to `struct<bytes: binary, path: string>`
with `bytes` already containing a complete WAV (header + PCM, 16 kHz
per the dataset card). We passthrough-write those bytes — no decode /
re-encode — so torchaudio is never imported here.

Output JSONL matches this repository's inference schema:
    {
      "id":   "<sample_id>",
      "text": "<text>",
      "APS":  {"instruction": "...", "gen_path": "<gt_wav_path>"},
      "DSD":  {"instruction": "...", "gen_path": "<gt_wav_path>"},
      "RP":   {"instruction": "...", "gen_path": "<gt_wav_path>"}
    }

`gen_path` is overloaded to point at the GT wav so `clsp_eval.py`
consumes the file unmodified. The same wav appears under all three task
blocks because InstructTTSEval ships one reference audio per id.

Usage:
    python scripts/eval/download_gt_audio.py \\
        --split en \\
        --audio_dir   out/en/gt_wav \\
        --output_jsonl out/en/results_gt.jsonl

Then score with the existing evaluator:
    python clsp_eval/clsp_eval.py \\
        --model_id    yfyeung/CLSP \\
        --input_jsonl out/en/results_gt.jsonl \\
        --output_jsonl out/en/results_gt_clsp.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

TASKS = ("APS", "DSD", "RP")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split", choices=["en", "zh"], default="en",
                   help="Which InstructTTSEval split to pull.")
    p.add_argument("--audio_dir", required=True,
                   help="Directory to write GT wavs into.")
    p.add_argument("--output_jsonl", required=True,
                   help="JSONL path consumed by clsp_eval.py.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows (smoke test).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip rows whose GT wav already exists on disk.")
    p.add_argument("--verify_sr", action="store_true",
                   help="Decode each WAV header with soundfile and assert "
                        "sample_rate == 16000. Off by default (the dataset "
                        "card guarantees 16 kHz; clsp_eval.py resamples "
                        "anyway). Use this once for a sanity check.")
    p.add_argument("--print_schema", action="store_true",
                   help="Print the parquet schema and the type of "
                        "`reference_audio[0]` then exit. Use this once on "
                        "first run to confirm the {bytes,path} layout.")
    return p.parse_args()


def _extract_wav_bytes(audio_field) -> bytes:
    """Pull WAV bytes out of a parquet `reference_audio` cell.

    HF's standard Audio feature serializes to a struct with `bytes` and
    `path`. Defensive: some parquet emitters drop one or the other, and
    a few datasets store raw bytes directly.
    """
    if isinstance(audio_field, dict):
        b = audio_field.get("bytes")
        if b:
            return b
        # Some Audio columns store only `path` pointing at a file on
        # disk — not what InstructTTSEval does, but call it out clearly.
        if audio_field.get("path"):
            raise ValueError(
                f"reference_audio cell has no `bytes`, only path={audio_field['path']!r}; "
                "this script expects embedded bytes."
            )
        raise ValueError(f"reference_audio struct missing `bytes`: keys={list(audio_field)}")
    if isinstance(audio_field, (bytes, bytearray)):
        return bytes(audio_field)
    raise TypeError(f"unexpected reference_audio cell type: {type(audio_field).__name__}")


def main():
    args = parse_args()

    from huggingface_hub import hf_hub_download  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    parquet_path = hf_hub_download(
        repo_id="CaasiHUANG/InstructTTSEval",
        filename=f"{args.split}.parquet",
        repo_type="dataset",
    )
    print(f"[gt-audio] parquet: {parquet_path}")
    table = pq.read_table(parquet_path)

    if args.print_schema:
        print("=== schema ===")
        print(table.schema)
        if "reference_audio" in table.column_names:
            cell = table.column("reference_audio")[0].as_py()
            print(f"reference_audio[0] type: {type(cell).__name__}")
            if isinstance(cell, dict):
                print(f"reference_audio[0] keys: {list(cell)}")
                b = cell.get("bytes")
                if b is not None:
                    print(f"reference_audio[0]['bytes'] len: {len(b)}; head: {bytes(b[:12])!r}")
        return

    if "reference_audio" not in table.column_names:
        raise SystemExit(
            f"`reference_audio` column missing from {parquet_path}. "
            f"Found columns: {table.column_names}. Run with --print_schema "
            "to inspect, and update this script if the dataset layout changed."
        )

    keep = [c for c in table.column_names
            if c in ("id", "text", "reference_audio", *TASKS)]
    table = table.select(keep)

    audio_dir = Path(args.audio_dir).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = table.to_pylist()
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"[gt-audio] split={args.split}; rows={len(rows)}; -> {audio_dir}")

    n_written = 0
    n_skipped = 0
    n_missing = 0

    with open(out_path, "w", encoding="utf-8", buffering=1) as out_f:
        for row in tqdm(rows, desc="gt-wav"):
            rid = row.get("id")
            text = row.get("text")
            if not rid:
                n_missing += 1
                continue

            wav_path = audio_dir / f"{rid}.wav"
            need_write = not (args.skip_existing and wav_path.exists())

            if need_write:
                try:
                    wav_bytes = _extract_wav_bytes(row.get("reference_audio"))
                except (ValueError, TypeError) as e:
                    print(f"[gt-audio] {rid}: {e}; skipping")
                    n_missing += 1
                    continue
                wav_path.write_bytes(wav_bytes)
                n_written += 1
            else:
                n_skipped += 1

            if args.verify_sr:
                import soundfile as sf  # lazy import
                info = sf.info(str(wav_path))
                if info.samplerate != 16000:
                    raise SystemExit(
                        f"{wav_path}: expected 16000 Hz, got {info.samplerate}. "
                        "Either update --verify_sr expectations or resample."
                    )

            result = {"id": rid, "text": text}
            gt_path_str = str(wav_path)
            for task in TASKS:
                instr = row.get(task)
                if isinstance(instr, str) and instr.strip():
                    result[task] = {"instruction": instr, "gen_path": gt_path_str}
                elif isinstance(instr, dict) and instr.get("instruction"):
                    result[task] = {"instruction": instr["instruction"], "gen_path": gt_path_str}
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"[gt-audio] wrote {out_path}")
    print(f"[gt-audio] wavs: written={n_written} skipped_existing={n_skipped} missing={n_missing}")


if __name__ == "__main__":
    main()
