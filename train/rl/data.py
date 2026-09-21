"""RL prompt dataset.

Reads `data/rl/<split>/{train,dev}.jsonl` (schema documented in README.md).
Source-agnostic: works for InstructTTSEval-derived JSONL or
ParaSpeechCaps-derived JSONL produced by data_preprocess/build_prompts_paraspeechcaps.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch.utils.data


REQUIRED_FIELDS = ("uid", "instruction", "text", "lang")


class RLPromptDataset(torch.utils.data.Dataset):
    """Yields {uid, instruction, text, lang} dicts. Reads JSONL once on init."""

    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        self.rows: list[dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                missing = [k for k in REQUIRED_FIELDS if not row.get(k)]
                if missing:
                    raise ValueError(f"{self.path}:{ln} missing fields {missing}")
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"{self.path} is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        item = {
            "uid": r["uid"],
            "instruction": r["instruction"],
            "text": r["text"],
            "lang": r["lang"],
        }
        # Optional fields present only in two-hop JSONL files.
        if "v1_tokens" in r:
            item["v1_tokens"] = r["v1_tokens"]
        if "critique" in r:
            item["critique"] = r["critique"]
        if "clsp_v1" in r:
            item["clsp_v1"] = r["clsp_v1"]
        if "wer_v1" in r:
            item["wer_v1"] = r["wer_v1"]
        if "clsp_v1_reference" in r:
            item["clsp_v1_reference"] = r["clsp_v1_reference"]
        return item


def collate_rl(batch: Iterable[dict]) -> dict:
    """Stack into parallel lists (no padding — rollouts are per-item)."""
    batch = list(batch)
    out: dict = {
        "uids": [b["uid"] for b in batch],
        "instructions": [b["instruction"] for b in batch],
        "texts": [b["text"] for b in batch],
        "langs": [b["lang"] for b in batch],
    }
    if "v1_tokens" in batch[0]:
        out["v1_tokens"] = [b["v1_tokens"] for b in batch]
    if "critique" in batch[0]:
        out["critiques"] = [b["critique"] for b in batch]
    if "clsp_v1" in batch[0]:
        out["clsp_v1"] = [b["clsp_v1"] for b in batch]
    if "wer_v1" in batch[0]:
        out["wer_v1"] = [b["wer_v1"] for b in batch]
    if "clsp_v1_reference" in batch[0]:
        out["clsp_v1_reference"] = [b["clsp_v1_reference"] for b in batch]
    return out
