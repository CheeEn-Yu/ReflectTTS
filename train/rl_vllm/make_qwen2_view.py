"""Produce a "Qwen2-view" of Step-Audio-2-mini for stock-vLLM rollout.

Why this exists
---------------
At twohop rollout time the model runs with `wavs=None`, so its forward is just
`embed_tokens -> Qwen2Model -> lm_head` (see modeling_step_audio_2.py:338-360).
The backbone is literally `transformers.Qwen2Model`; the 493 `encoder.*`/`adapter.*`
tensors (audio INPUT encoder) never execute during generation. So for rollout the
checkpoint is equivalent to a vanilla `Qwen2ForCausalLM` with vocab 158720.

This script writes a directory that *presents* the checkpoint as that vanilla
Qwen2 so stock vLLM (or stock transformers) can load it with its optimized,
battle-tested Qwen2 path — no model fork, no custom vLLM model class.

How (cheap + non-destructive)
-----------------------------
  * the source checkpoint is READ-ONLY — we never touch it;
  * we MATERIALIZE a fresh `model.safetensors` holding only `model.*` +
    `lm_head.weight`. (Symlinking the originals + a pruned index does NOT work
    for vLLM: its loader enumerates every tensor in the referenced shard files
    and ignores the index's weight_map, so it chokes on the leftover `encoder.*`
    tensors — `no module named 'encoder' in Qwen2ForCausalLM`. transformers does
    honor the index, but we need a checkpoint both loaders accept.)
  * we write a fresh `config.json` flattening `text_config` into a standard
    Qwen2Config with architectures=["Qwen2ForCausalLM"];
  * tokenizer files are symlinked.

This module is self-contained: it imports nothing from train/rl and does not
touch the existing pipeline. Run:

    python -m train.rl_vllm.make_qwen2_view \
        --src Step-Audio2/Step-Audio-2-mini \
        --out out/qwen2_view/step-audio-2-mini-qwen2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Files needed to init a tokenizer for the view (symlinked from src if present).
_TOKENIZER_FILES = [
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
]


def _build_qwen2_config(src_cfg: dict) -> dict:
    """Flatten StepAudio2's nested text_config into a standard Qwen2 config."""
    tc = src_cfg["text_config"]
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": tc["hidden_size"],
        "intermediate_size": tc["intermediate_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "vocab_size": tc["vocab_size"],
        "rms_norm_eps": tc["rms_norm_eps"],
        "rope_theta": tc["rope_theta"],
        "max_position_embeddings": tc["max_position_embeddings"],
        "rope_scaling": tc.get("rope_scaling"),
        "hidden_act": "silu",
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "use_cache": True,
        # lm_head.weight is materialized in the checkpoint -> NOT tied.
        "tie_word_embeddings": False,
        # Qwen2-7B does not use SWA; disable to avoid the sdpa SWA warning/path.
        "use_sliding_window": False,
        "sliding_window": None,
        "max_window_layers": tc["num_hidden_layers"],
        "bos_token_id": tc.get("eos_token_id", 151643),
        "eos_token_id": tc.get("eos_token_id", 151643),
        "pad_token_id": tc.get("pad_token_id", 151643),
        "torch_dtype": tc.get("torch_dtype", "bfloat16"),
        # provenance breadcrumb (ignored by loaders)
        "_view_of": "Step-Audio-2-mini (rollout-only Qwen2 view; audio encoder dropped)",
    }


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="Step-Audio2/Step-Audio-2-mini",
                    help="source Step-Audio-2-mini checkpoint dir (read-only)")
    ap.add_argument("--out", default="out/qwen2_view/step-audio-2-mini-qwen2",
                    help="output Qwen2-view dir (symlinks + pruned index + qwen2 config)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"src not found: {src}")
    out.mkdir(parents=True, exist_ok=True)

    # remove any stale per-shard symlinks from an earlier (symlink-based) run —
    # leaving them alongside the new single-file model.safetensors confuses loaders.
    for stale in out.glob("model-*-of-*.safetensors"):
        if stale.is_symlink() or stale.exists():
            stale.unlink()

    # ---- 1. prune the safetensors index to model.* + lm_head.* ----
    idx_path = src / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    wm = idx["weight_map"]
    keep = {k: s for k, s in wm.items()
            if k.startswith("model.") or k.startswith("lm_head.")}
    drop = [k for k in wm if k not in keep]
    if not keep:
        raise SystemExit("no model.*/lm_head.* tensors found — wrong src?")

    # ---- 2. materialize a fresh model.safetensors with ONLY kept tensors ----
    # (single-file checkpoint; both vLLM and transformers load it directly.)
    from safetensors import safe_open          # local imports: only needed here
    from safetensors.torch import save_file
    # group kept keys by source shard so each shard is opened once
    by_shard: dict[str, list[str]] = {}
    for k, s in keep.items():
        by_shard.setdefault(s, []).append(k)
    state: dict = {}
    total = 0
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(src / shard), framework="pt", device="cpu") as f:
            for k in keys:
                t = f.get_tensor(k)
                state[k] = t
                total += t.numel() * t.element_size()
    stale_index = out / "model.safetensors.index.json"
    if stale_index.exists():
        stale_index.unlink()  # single-file checkpoint needs no shard index
    save_file(state, str(out / "model.safetensors"),
              metadata={"format": "pt", "_view_of": "Step-Audio-2-mini"})
    del state

    # ---- 4. write the Qwen2 config ----
    src_cfg = json.loads((src / "config.json").read_text())
    qcfg = _build_qwen2_config(src_cfg)
    (out / "config.json").write_text(json.dumps(qcfg, indent=2))

    # ---- 5. symlink tokenizer files ----
    linked_tok = []
    for fn in _TOKENIZER_FILES:
        sp = src / fn
        if sp.exists():
            _symlink(sp, out / fn)
            linked_tok.append(fn)

    print(f"[make_qwen2_view] src  : {src}")
    print(f"[make_qwen2_view] out  : {out}")
    print(f"[make_qwen2_view] kept : {len(keep)} tensors  (dropped {len(drop)} encoder/adapter)")
    print(f"[make_qwen2_view] materialized model.safetensors: {total/1e9:.2f} GB")
    print(f"[make_qwen2_view] tokenizer files : {linked_tok}")
    print(f"[make_qwen2_view] config: architectures={qcfg['architectures']} "
          f"model_type={qcfg['model_type']} vocab={qcfg['vocab_size']} "
          f"tie_word_embeddings={qcfg['tie_word_embeddings']}")
    print("[make_qwen2_view] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
