"""Environment setup shared by the speaker_rag package integrations.

Mirrors the torchcodec bypass + sys.path wiring used by infer/instructtseval_vllm.py
so `token2wav` / `s3tokenizer` import cleanly in the singularity env (no FFmpeg
libav*), and so `clsp_utils` is importable.

Import this module (for its side effects) BEFORE importing stepaudio2 /
token2wav / s3tokenizer.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- torchcodec bypass (must run before importing token2wav / s3tokenizer) ---
# torchaudio>=2.9 routes torchaudio.load through torchcodec, which fails to
# dlopen libavutil.so.* here. soundfile reads/writes the same wavs without
# ffmpeg. Identical to the patch in infer/instructtseval_vllm.py.
import torch as _torch  # noqa: E402
import torchaudio as _torchaudio  # noqa: E402
import soundfile as _sf  # noqa: E402


def _sf_load(uri, *_, **__):
    data, sr = _sf.read(uri, always_2d=True, dtype="float32")
    return _torch.from_numpy(data.T.copy()), sr


def _sf_save(uri, src, sample_rate, *, format=None, channels_first=True, **__):
    if isinstance(src, _torch.Tensor):
        src = src.detach().cpu().numpy()
    if src.ndim == 2 and channels_first:
        src = src.T
    _sf.write(uri, src, sample_rate, format=(format.upper() if format else None))


_torchaudio.load = _sf_load
_torchaudio.save = _sf_save
# --- end bypass ---

# Repo layout: <repo>/speaker_rag/_env.py
REPO_DIR = Path(__file__).resolve().parent.parent
STEP_DIR = REPO_DIR / "Step-Audio2"
CLSP_DIR = REPO_DIR / "clsp_eval"

for _p in (STEP_DIR, CLSP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def resolve_under_step(path: str) -> str:
    """Resolve a path that is relative to Step-Audio2/ (matches infer scripts)."""
    import os
    return path if os.path.isabs(path) else str(STEP_DIR / path)
