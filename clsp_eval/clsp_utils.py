"""Shared CLSP scoring utilities.

Shared by post-hoc evaluation and the RL reward scorer.
"""
from __future__ import annotations

import io

import torch
import torch.nn.functional as F

TARGET_SR = 16000


def load_clsp(model_id: str, device: torch.device):
    from transformers import AutoModel  # type: ignore
    return AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()


def _decode_audio(source) -> tuple[torch.Tensor, int]:
    """Decode `source` (file path str or raw wav bytes) -> (tensor, sr)."""
    import soundfile as sf
    if isinstance(source, (str, bytes)) and not isinstance(source, bytes):
        data, sr = sf.read(source, dtype="float32", always_2d=True)
    else:
        data, sr = sf.read(io.BytesIO(source), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


def to_mono_16k(source) -> torch.Tensor:
    """Load wav from a file path or raw bytes as mono float32 at 16 kHz, shape (1, T)."""
    audio, sr = _decode_audio(source)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        import torchaudio.functional as AF  # type: ignore
        audio = AF.resample(audio, sr, TARGET_SR)
    return audio  # (1, T)


@torch.no_grad()
def score_batch(clsp_model, device: torch.device, wavs_16k: list,
                instructions: list) -> list:
    """Batch CLSP scoring. wavs_16k: list of (1, T) tensors (may differ in length)."""
    lengths = [w.size(1) for w in wavs_16k]
    max_len = max(lengths)
    B = len(wavs_16k)
    padded = torch.zeros(B, max_len)
    for i, w in enumerate(wavs_16k):
        padded[i, : w.size(1)] = w.squeeze(0)
    audio = padded.to(device)
    audio_lens = torch.tensor(lengths, device=device)
    audio_feat, text_feat, _ = clsp_model(audio, audio_lens, instructions)
    audio_feat = F.normalize(audio_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)
    return (audio_feat * text_feat).sum(dim=-1).cpu().tolist()


@torch.no_grad()
def score_wav(clsp_model, device: torch.device, wav_16k: torch.Tensor,
              instruction: str) -> float:
    """CLSP cosine similarity for a preloaded mono-16k tensor (1, T).

    Use when the wav is already decoded (e.g. shared with ASR) so the file is
    only read once.
    """
    audio = wav_16k.to(device)
    audio_lens = torch.tensor([audio.size(1)], device=device)
    audio_feat, text_feat, _ = clsp_model(audio, audio_lens, [instruction])
    audio_feat = F.normalize(audio_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)
    return float((audio_feat * text_feat).sum().cpu())


@torch.no_grad()
def score(clsp_model, device: torch.device, source, instruction: str) -> float:
    """Compute CLSP cosine similarity for a (wav, instruction) pair.

    `source` can be a file path (str) or raw wav bytes.
    """
    return score_wav(clsp_model, device, to_mono_16k(source), instruction)


# --------------------------------------------------------------------------- #
#  ASR + WER (intelligibility)                                                 #
#  Mirrors train/rl/reward.py so the eval-time WER matches the RL reward's     #
#  definition (en -> WER, zh -> CER, same normalization). Kept self-contained  #
#  here to avoid importing train/ code into eval/.                             #
# --------------------------------------------------------------------------- #
def load_asr(name: str = "openai/whisper-large-v3", device: torch.device | None = None,
             dtype=torch.bfloat16):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # type: ignore
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(name, torch_dtype=dtype)
    model = model.to(device).eval()
    return model, processor


@torch.no_grad()
def transcribe(asr_model, asr_processor, wav_16k: torch.Tensor, lang: str,
               device: torch.device) -> str:
    """Whisper transcription of a preloaded mono-16k tensor (1, T)."""
    feats = asr_processor(
        wav_16k.squeeze(0).numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in feats.items()}
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(asr_model.dtype)
    gen_kwargs = {"max_new_tokens": 256}
    forced = asr_processor.get_decoder_prompt_ids(
        language=("zh" if lang == "zh" else "en"), task="transcribe",
    ) if hasattr(asr_processor, "get_decoder_prompt_ids") else None
    if forced:
        gen_kwargs["forced_decoder_ids"] = forced
    ids = asr_model.generate(**inputs, **gen_kwargs)
    return asr_processor.batch_decode(ids, skip_special_tokens=True)[0]


def wer(hyp: str, ref: str, lang: str) -> float:
    """WER for en, CER for zh; returns float in [0, ~1+].

    Normalizes case/punctuation before scoring so they don't inflate the error
    rate. Identical definition to train/rl/reward.py:wer.
    """
    import jiwer  # type: ignore
    import jiwer.transforms as tr  # type: ignore
    h = (hyp or "").strip()
    r = (ref or "").strip()
    if not r:
        return 0.0 if not h else 1.0
    if lang == "zh":
        return float(jiwer.cer(r.replace(" ", ""), h.replace(" ", "")))
    _en_transform = tr.Compose([
        tr.RemovePunctuation(),
        tr.ToLowerCase(),
        tr.Strip(),
        tr.ReduceToListOfListOfWords(),
    ])
    return float(jiwer.wer(r, h,
                           reference_transform=_en_transform,
                           hypothesis_transform=_en_transform))
