"""Merge a LoRA adapter into Step-Audio-2-mini and write a *merged* Qwen2-view
for stock-vLLM inference.

Why this exists
---------------
`make_qwen2_view` writes a Qwen2-view of the BASE checkpoint. For inference with
a trained adapter we want the adapter baked into the weights so stock vLLM (which
loads the view as a vanilla `Qwen2ForCausalLM`) runs the policy directly — no
runtime LoRA, no hot weight-sync. At twohop generation time the model runs with
`wavs=None`, so the forward is just `embed_tokens -> Qwen2Model -> lm_head`; the
audio INPUT encoder never executes and is dropped from the view (see
make_qwen2_view docstring).

This:
  * loads StepAudio2 (chat) + the LoRA adapter, `merge_adapter()`;
  * extracts the merged `model.* + lm_head.weight` (clean Qwen2 names, bf16 cpu)
    via the same `merged_state_dict` the vLLM trainer uses for weight-sync;
  * writes a single-file `model.safetensors` + a flattened Qwen2 `config.json`
    + symlinked tokenizer — a directory both vLLM and transformers load directly.

Usage:
    python -m train.rl_vllm.merge_view \
        --adapter out/rl/trl_twohop_v2_hp4/checkpoint-120 \
        --base    Step-Audio-2-mini \
        --out     out/qwen2_view/hp4_cum120_merged
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# side effects: torchcodec bypass + sys.path for stepaudio2; _resolve_step helper.
from train.rl.train_rl import _resolve_step
from train.rl_vllm.make_qwen2_view import (
    _TOKENIZER_FILES,
    _build_qwen2_config,
    _symlink,
)
from train.rl_vllm.train_rl_vllm import merged_state_dict


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapter", required=True,
                    help="LoRA adapter dir (e.g. an RL checkpoint-N) to merge into base.")
    ap.add_argument("--base", default="Step-Audio-2-mini",
                    help="base Step-Audio-2-mini checkpoint (resolved under Step-Audio2/).")
    ap.add_argument("--out", required=True, help="output merged Qwen2-view dir.")
    args = ap.parse_args()

    from safetensors.torch import save_file
    from stepaudio2 import StepAudio2  # type: ignore
    from peft import PeftModel  # type: ignore

    base_dir = _resolve_step(args.base)
    adapter = args.adapter
    print(f"[merge_view] loading base {base_dir}", flush=True)
    model = StepAudio2(base_dir)
    print(f"[merge_view] attaching adapter {adapter}", flush=True)
    model.llm = PeftModel.from_pretrained(model.llm, adapter)
    model.llm.eval()

    # merge LoRA into the base_layer weights, then extract clean qwen2 names.
    model.llm.merge_adapter()
    try:
        state = merged_state_dict(model.llm)  # {model.*/lm_head.weight: cpu bf16}
    finally:
        model.llm.unmerge_adapter()

    total = sum(t.numel() * t.element_size() for t in state.values())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # stale single-file / shard artifacts from an earlier run confuse loaders.
    for stale in list(out.glob("model-*-of-*.safetensors")) + [out / "model.safetensors.index.json"]:
        if stale.is_symlink() or stale.exists():
            stale.unlink()

    save_file(state, str(out / "model.safetensors"),
              metadata={"format": "pt", "_view_of": "Step-Audio-2-mini+LoRA(merged)"})

    src_cfg = json.loads((Path(base_dir) / "config.json").read_text())
    (out / "config.json").write_text(json.dumps(_build_qwen2_config(src_cfg), indent=2))

    linked = []
    for fn in _TOKENIZER_FILES:
        sp = Path(base_dir) / fn
        if sp.exists():
            _symlink(sp, out / fn)
            linked.append(fn)

    print(f"[merge_view] merged adapter : {adapter}")
    print(f"[merge_view] wrote {len(state)} tensors ({total/1e9:.2f} GB) -> {out}")
    print(f"[merge_view] tokenizer files: {linked}")
    print("[merge_view] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
