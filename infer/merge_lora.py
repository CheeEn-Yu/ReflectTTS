"""Merge a peft LoRA adapter into Step-Audio-2 base weights → new model dir.

Stepfun's vLLM fork rejects LoRA at load time:
    ValueError: StepAudio2ForCausalLM does not support LoRA yet.
Workaround: bake the adapter into the base weights and serve the merged dir
as a normal MODEL_DIR (no --enable-lora flag).

The merged output dir is a drop-in replacement for the original model dir:
  - LLM files (config.json, *.safetensors, tokenizer*, generation_config.json,
    modeling_*.py) are rewritten with merged weights.
  - All other files / subdirs (token2wav/, audio tokenizer assets, etc.) are
    symlinked from the original dir so we don't duplicate ~5GB of vocoder.

Usage:
    python infer/merge_lora.py \\
        --base_model    Step-Audio2/Step-Audio-2-mini \\
        --adapter_path  out/rl/your_ckpt/adapter_final \\
        --output_dir    Step-Audio2/Step-Audio-2-mini-merged-v2_rl

Then start vLLM with MODEL_DIR pointing at the merged dir (no ADAPTER_PATH):
    MODEL_DIR=$PWD/Step-Audio2/Step-Audio-2-mini-merged-v2_rl \\
    bash scripts/infer/run_vllm_server.singularity.sh
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "Step-Audio2"))

# Weight + config files that save_pretrained AUTHORITATIVELY rewrites. If the
# merged save produces these, we drop the originals; if it doesn't, we still
# treat them as stale (don't pull old shards forward).
LLM_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model.bin.index.json",
)

# Large subdirs that the LLM server doesn't need (only the offline phase reads
# token2wav). Symlinked, not copied, to save disk. NOTE: symlinks pointing at
# the base dir will appear broken INSIDE the Singularity container (only the
# merged dir is bind-mounted), but that's fine — vLLM never touches them.
SYMLINK_SUBDIRS = ("token2wav",)


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _weight_filenames(base_dir: Path) -> set[str]:
    names: set[str] = set()
    for pat in LLM_WEIGHT_PATTERNS:
        for f in base_dir.glob(pat):
            if f.is_file():
                names.add(f.name)
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_model", required=True,
                    help="Step-Audio-2 base snapshot dir (the one currently bound "
                         "to /model in run_vllm_server.singularity.sh).")
    ap.add_argument("--adapter_path", required=True,
                    help="peft LoRA adapter dir (adapter_config.json + "
                         "adapter_model.safetensors).")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--force", action="store_true",
                    help="Wipe output_dir if it already exists.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_dir = _resolve(args.base_model).resolve()
    adapter_dir = _resolve(args.adapter_path).resolve()
    out_dir = _resolve(args.output_dir).resolve()

    if not base_dir.is_dir():
        sys.exit(f"base_model not a dir: {base_dir}")
    if not (adapter_dir / "adapter_config.json").exists():
        sys.exit(f"no adapter_config.json under {adapter_dir}")

    if out_dir.exists():
        if not args.force:
            sys.exit(f"output_dir exists: {out_dir} (use --force to overwrite)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[merge] base    = {base_dir}")
    print(f"[merge] adapter = {adapter_dir}")
    print(f"[merge] output  = {out_dir}")
    print(f"[merge] dtype   = {args.dtype}")

    print("[merge] loading base LLM ...")
    llm = AutoModelForCausalLM.from_pretrained(
        str(base_dir), trust_remote_code=True, torch_dtype=dtype,
    )
    tok = AutoTokenizer.from_pretrained(
        str(base_dir), trust_remote_code=True, padding_side="right",
    )

    print("[merge] attaching adapter ...")
    llm = PeftModel.from_pretrained(llm, str(adapter_dir))

    print("[merge] merge_and_unload ...")
    llm = llm.merge_and_unload()

    print(f"[merge] saving merged LLM -> {out_dir}")
    llm.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))

    # Pull in everything else from base_dir:
    #   - large subdirs in SYMLINK_SUBDIRS -> symlink (saves disk; not used by vLLM)
    #   - other files (modeling_*.py, README, etc.) -> COPY (must be visible
    #     inside the Singularity container, where symlinks to host paths break)
    #   - stale weight shards from base -> SKIP (merged save is authoritative)
    weight_files = _weight_filenames(base_dir)
    written = {f.name for f in out_dir.iterdir()}
    print(f"[merge] wrote {len(written)} entries; pulling rest from base ...")
    copied, linked, skipped = 0, 0, 0
    for entry in base_dir.iterdir():
        if entry.name in written:
            continue
        dst = out_dir / entry.name
        if entry.is_dir() and entry.name in SYMLINK_SUBDIRS:
            os.symlink(entry.resolve(), dst)
            linked += 1
            continue
        if entry.is_file() and entry.name in weight_files:
            skipped += 1
            continue
        if entry.is_dir():
            shutil.copytree(entry, dst, symlinks=True)
        else:
            shutil.copy2(entry, dst)
        copied += 1
    print(f"[merge] copied={copied} symlinked={linked} skipped(stale-weights)={skipped}")
    print("[merge] DONE")


if __name__ == "__main__":
    main()
