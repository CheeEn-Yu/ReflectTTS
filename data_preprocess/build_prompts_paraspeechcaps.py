#!/usr/bin/env python3
"""Convert ParaSpeechCaps captions JSONL to RL prompt JSONL.

Input schema (output of data_preprocess/download_paraspeechcaps.py):
  relative_audio_path, transcription, text_description (list), *_tags, source

Output schema (see README.md, "Data format"):
  {uid, instruction, text, lang, source}

PSC ships its own train/dev/holdout/test splits, so we reuse them rather than
re-splitting by id. Default mapping: train_base -> train.jsonl, dev -> dev.jsonl.

Usage:
    python data_preprocess/build_prompts_paraspeechcaps.py \
        --input_dir data/external/paraspeechcaps \
        --output_dir data/rl/paraspeechcaps_en
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

DEFAULT_INPUT_DIR = Path("data/external/paraspeechcaps")
DEFAULT_OUTPUT_DIR = Path("data/rl/paraspeechcaps_en")
DEFAULT_SPLIT_MAP = {"train_base": "train", "dev": "dev"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--split_map",
        nargs="+",
        default=[f"{k}={v}" for k, v in DEFAULT_SPLIT_MAP.items()],
        help='source_split=output_split pairs, e.g. "train_base=train dev=dev".',
    )
    p.add_argument("--min_text_chars", type=int, default=4)
    p.add_argument("--max_text_chars", type=int, default=400)
    p.add_argument("--min_instruction_chars", type=int, default=20)
    p.add_argument(
        "--drop_sources",
        nargs="*",
        default=(),
        help="Skip rows whose source is in this list (e.g. voxceleb).",
    )
    p.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Cap rows per output split (smoke / debug).",
    )
    return p.parse_args()


def normalize(s) -> str:
    if s is None:
        return ""
    return str(s).strip()


def to_rl_row(row: dict) -> dict | None:
    desc = row.get("text_description") or []
    if isinstance(desc, list):
        instruction = normalize(desc[0]) if desc else ""
    else:
        instruction = normalize(desc)
    text = normalize(row.get("transcription"))
    uid = row.get("relative_audio_path")
    if not uid or not instruction or not text:
        return None
    return {
        "uid": uid,
        "instruction": instruction,
        "text": text,
        "lang": "en",
        "source": row.get("source", "paraspeechcaps"),
    }


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_map = dict(s.split("=", 1) for s in args.split_map)
    drop_sources = set(args.drop_sources)
    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "filters": {
            "min_text_chars": args.min_text_chars,
            "max_text_chars": args.max_text_chars,
            "min_instruction_chars": args.min_instruction_chars,
            "drop_sources": sorted(drop_sources),
            "max_rows": args.max_rows,
        },
        "splits": {},
    }

    seen_uids: set[str] = set()
    for src_split, out_split in split_map.items():
        in_path = args.input_dir / f"{src_split}.jsonl"
        out_path = args.output_dir / f"{out_split}.jsonl"
        if not in_path.exists():
            print(f"[psc] missing {in_path}, skipping")
            continue

        n_in = n_out = 0
        n_drop_empty = n_drop_dup = n_drop_len = n_drop_instr = n_drop_src = 0
        source_counts: Counter[str] = Counter()
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        with in_path.open() as fin, tmp_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                if args.max_rows is not None and n_out >= args.max_rows:
                    break
                n_in += 1
                row = json.loads(line)

                if drop_sources and row.get("source") in drop_sources:
                    n_drop_src += 1
                    continue

                rl = to_rl_row(row)
                if rl is None:
                    n_drop_empty += 1
                    continue

                tlen = len(rl["text"])
                if tlen < args.min_text_chars or tlen > args.max_text_chars:
                    n_drop_len += 1
                    continue
                if len(rl["instruction"]) < args.min_instruction_chars:
                    n_drop_instr += 1
                    continue

                if rl["uid"] in seen_uids:
                    n_drop_dup += 1
                    continue
                seen_uids.add(rl["uid"])

                source_counts[rl["source"]] += 1
                fout.write(json.dumps(rl, ensure_ascii=False) + "\n")
                n_out += 1
        tmp_path.replace(out_path)

        manifest["splits"][out_split] = {
            "source_split": src_split,
            "input_path": str(in_path),
            "output_path": str(out_path),
            "rows_in": n_in,
            "rows_out": n_out,
            "dropped_empty": n_drop_empty,
            "dropped_short_instruction": n_drop_instr,
            "dropped_length": n_drop_len,
            "dropped_duplicate_uid": n_drop_dup,
            "dropped_source": n_drop_src,
            "source_counts": dict(source_counts.most_common()),
            "sha256": hash_file(out_path),
        }
        print(
            f"[psc] {src_split} -> {out_split}: kept {n_out:,}/{n_in:,} "
            f"(empty={n_drop_empty}, len={n_drop_len}, instr={n_drop_instr}, "
            f"dup={n_drop_dup}, src={n_drop_src})"
        )

    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[psc] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
