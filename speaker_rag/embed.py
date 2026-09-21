"""CLSP embedding helpers + reference audio-embedding cache.

CLSP (yfyeung/CLSP) is a CLAP-style dual encoder: the audio tower and text
tower are independent (no cross-attention), both projected to a 512-d joint
space and L2-normalized. So we can encode each modality alone:

    audio_feat, _, _ = clsp_model(audio, audio_lens, None)
    _, text_feat, _  = clsp_model(None, None, [instruction])

Speaker RAG uses this to retrieve a reference: precompute each candidate
reference's audio_feat once (cached to disk), encode the instruction's text_feat
per utterance, pick the reference with the highest cosine similarity. No vocoder
pass needed during retrieval.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from . import _env  # noqa: F401  (side effects: sys.path so clsp_utils imports)
# Reuse the repo's audio loader (path/bytes -> mono 16k (1, T)).
from clsp_utils import to_mono_16k  # type: ignore


@torch.no_grad()
def encode_audio(clsp_model, device, source) -> torch.Tensor:
    """L2-normalized 512-d audio embedding for a wav (path or bytes)."""
    wav = to_mono_16k(source).to(device)            # (1, T)
    lens = torch.tensor([wav.size(1)], device=device)
    audio_feat, _, _ = clsp_model(wav, lens, None)
    return F.normalize(audio_feat, dim=-1).squeeze(0).cpu()


@torch.no_grad()
def encode_text(clsp_model, device, instruction: str) -> torch.Tensor:
    """L2-normalized 512-d text embedding for an instruction string."""
    _, text_feat, _ = clsp_model(None, None, [instruction])
    return F.normalize(text_feat, dim=-1).squeeze(0).cpu()


def embed_references(references, clsp_model, device,
                     cache_path: str | Path | None = None) -> torch.Tensor:
    """Return an (N, 512) matrix of references' audio embeddings.

    Embeddings are cached to `cache_path` (torch.save of a {name: entry} dict)
    and reused while the reference wav's path + mtime are unchanged, so adding a
    reference only re-encodes the new one.
    """
    cache: dict = {}
    if cache_path and Path(cache_path).is_file():
        try:
            cache = torch.load(cache_path, map_location="cpu")
        except Exception:  # noqa: BLE001  (corrupt cache -> recompute)
            cache = {}

    embs: list[torch.Tensor] = []
    dirty = False
    for ref in references:
        mtime = os.path.getmtime(ref.wav)
        ent = cache.get(ref.name)
        if ent is not None and ent.get("wav") == ref.wav and ent.get("mtime") == mtime:
            embs.append(ent["emb"])
            continue
        emb = encode_audio(clsp_model, device, ref.wav)
        cache[ref.name] = {"wav": ref.wav, "mtime": mtime, "emb": emb}
        embs.append(emb)
        dirty = True

    if cache_path and dirty:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)

    return torch.stack(embs)  # (N, 512), row i == references[i]
