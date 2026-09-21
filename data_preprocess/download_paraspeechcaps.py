#!/usr/bin/env python3
"""Download ParaSpeechCaps captions only (no audio) from Hugging Face.

Dataset: ajd12342/paraspeechcaps — license CC-BY-NC-SA 4.0 (non-commercial).
Audio is NOT bundled on the Hub; rows carry only `relative_audio_path` strings,
so this script never touches the source corpora (LibriTTS-R / VoxCeleb /
Expresso / EARS / Emilia-EN).

Output: one JSONL per split under --output_dir, schema matches the keep-list
below. Map to the RL prompt schema (uid/instruction/text/lang) in a
follow-up `build_prompts_paraspeechcaps.py` step.

Usage (run on the RL host, not locally):
    huggingface-cli login  # accept the dataset license once
    python data_preprocess/download_paraspeechcaps.py \
        --splits train_base dev \
        --output_dir data/external/paraspeechcaps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_REPO = "ajd12342/paraspeechcaps"
DEFAULT_SPLITS = ["train_base", "dev"]
ALL_SPLITS = ["train_base", "train_scaled", "dev", "holdout", "test"]

KEEP_COLUMNS = [
    "relative_audio_path",
    "transcription",
    "text_description",
    "intrinsic_tags",
    "situational_tags",
    "basic_tags",
    "all_tags",
    "source",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=ALL_SPLITS,
        help=f"Splits to download (default: {' '.join(DEFAULT_SPLITS)}).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/external/paraspeechcaps"),
        help="Destination directory for JSONL outputs.",
    )
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id.")
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Override HF datasets cache (defaults to ~/.cache/huggingface).",
    )
    p.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="If set, write only the first N rows per split (smoke run).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSONL outputs (default: skip if present).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Missing dependency. Install with:\n"
            "    pip install -U 'datasets>=2.18' huggingface_hub",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        out_path = args.output_dir / f"{split}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[paraspeechcaps] skip existing {out_path} (use --overwrite to redo)")
            continue

        print(f"[paraspeechcaps] loading split={split}")
        ds = load_dataset(
            args.repo,
            split=split,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )

        drop = [c for c in ds.column_names if c not in KEEP_COLUMNS]
        if drop:
            ds = ds.remove_columns(drop)

        if args.max_rows is not None:
            ds = ds.select(range(min(args.max_rows, len(ds))))

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path.replace(out_path)

        print(f"[paraspeechcaps] wrote {len(ds):,} rows -> {out_path}")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
