"""Validate an InstructTTSEval result JSONL before paper-level evaluation.

The official Gemini evaluator excludes missing/null results from its denominator.
That is useful for interrupted runs but can silently inflate a reported score. This
preflight requires every requested task to have the expected number of existing
audio files before an expensive judge run starts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("APS", "DSD", "RP")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--expected_per_task", type=int, default=1000)
    parser.add_argument("--require_gemini_score", action="store_true",
                        help="Also require every task's gemini_score to be bool.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.input_jsonl)
    counts = {task: 0 for task in args.tasks}
    errors: list[str] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id") or "")
            if not row_id:
                errors.append(f"line {line_no}: missing id")
            elif row_id in seen_ids:
                errors.append(f"line {line_no}: duplicate id {row_id!r}")
            seen_ids.add(row_id)

            for task in args.tasks:
                block = row.get(task)
                if not isinstance(block, dict):
                    errors.append(f"line {line_no} ({row_id}): missing {task} block")
                    continue
                instruction = block.get("instruction")
                audio_path = block.get("gen_path")
                if not instruction:
                    errors.append(f"line {line_no} ({row_id}/{task}): missing instruction")
                    continue
                if not audio_path:
                    errors.append(f"line {line_no} ({row_id}/{task}): missing gen_path")
                    continue
                if not Path(audio_path).is_file():
                    errors.append(
                        f"line {line_no} ({row_id}/{task}): audio does not exist: {audio_path}"
                    )
                    continue
                if args.require_gemini_score and not isinstance(block.get("gemini_score"), bool):
                    errors.append(
                        f"line {line_no} ({row_id}/{task}): gemini_score is not boolean"
                    )
                    continue
                counts[task] += 1

    for task, count in counts.items():
        if count != args.expected_per_task:
            errors.append(f"{task}: valid={count}, expected={args.expected_per_task}")

    if errors:
        preview = "\n".join(f"  - {msg}" for msg in errors[:20])
        extra = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise SystemExit(f"Incomplete evaluation input:\n{preview}{extra}")

    summary = ", ".join(f"{task}={count}" for task, count in counts.items())
    print(f"[validate_results] OK: {summary}; unique_ids={len(seen_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
