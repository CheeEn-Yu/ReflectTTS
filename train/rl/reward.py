"""Reward for the vocoded rollout audio.

Two regimes (selected per-call by whether a v1 baseline is supplied):

  absolute  (no v1):  reward = beta * CLSP - alpha * min(WER, wer_cap)
  two-hop   (v1):     a WER gate guards content fidelity, then the reward is
                      driven by how much v2 *improves* CLSP over v1:
                        reward = beta * CLSP_v2           (raw, in [-1, 1])
                                 + lambda_improve * f(CLSP_v2 - CLSP_v1)
                      where f is a non-linear transform (default tanh) so that
                      the v1 baseline does NOT cancel in GRPO's within-group
                      advantage (a linear delta cancels because v1 is constant
                      per prompt).

Why relative + gate: in the refine setting every rollout conditions on the same
v1, so absolute CLSP barely varies within a group (tiny advantage std -> skipped
step) and nothing rewards actually *beating* v1. The improvement term injects
group variance and points the gradient straight at "v2 > v1"; the WER gate stops
the model from trading away intelligibility for style.

Components (all frozen, inference-only):
  - token2wav:    Step-Audio2/Step-Audio-2-mini/token2wav  (already on disk)
  - ASR:          openai/whisper-large-v3
  - CLSP:         yfyeung/CLSP                       (also used by clsp_eval.py)

Edge cases:
  - empty / too-short audio_codes  -> invalid_penalty
  - ASR/CLSP exception             -> invalid_penalty
  - the one-hop WER penalty is capped at wer_cap so a rollout where ASR fails
    completely cannot dominate the gradient with a huge negative.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .profiling import PROF


# Match clsp_eval.py: CLSP expects 16 kHz mono; Whisper does too.
TARGET_SR = 16000

# v1_tokens are stored as raw LM token ids; subtract this to get s3 codes (0..6560).
AUDIO_TOKEN_OFFSET = 151696


@dataclass
class RewardConfig:
    alpha_wer: float = 0.5
    beta_clsp: float = 0.5
    wer_floor: float = 0.0
    wer_cap: float = 0.5          # one-hop absolute reward: cap on the monotone WER penalty
    invalid_penalty: float = -1.0
    min_audio_codes: int = 5  # ~0.2s at 25Hz; below this is almost certainly empty
    # ---- two-hop (relative) reward ----
    lambda_improve: float = 1.0   # weight on the (possibly non-linear) improvement term
    improve_fn: str = "tanh"      # "none" = linear, "tanh", "exp" (see _apply_improve_fn)
    improve_scale: float = 10.0   # scale factor inside the non-linear fn; higher = sharper
    wer_penalty: float = 0.5      # weight on the WER penalty
    wer_gate: float = 0.10        # relative mode: WER ref when wer_v1 unknown.
                                  # also the hard-gate threshold (see wer_hard_gate)
    gate_penalty: float = -1.0    # fixed penalty added when wer_hard_gate and WER_v2 > wer_gate
    # ---- WER penalty shape (two-hop branch) ----
    wer_absolute: bool = False    # True: penalty = wer_penalty*min(WER_v2, wer_cap) — optimize WER
                                  # directly, NO v1 margin. False: relative max(0, WER_v2 - WER_v1).
    wer_hard_gate: bool = False   # True: also add gate_penalty whenever WER_v2 > wer_gate (cliff).


class RewardScorer:
    """Score a list of Rollouts. Self-contained and stateless across calls."""

    def __init__(
        self,
        token2wav,
        asr_model,
        asr_processor,
        clsp_model,
        prompt_wav: str,
        cfg: RewardConfig,
        device: torch.device | str = "cuda",
        token2wav_cuda_idx: int | None = None,
        references=None,
        ref_embeddings=None,
    ):
        self.token2wav = token2wav
        self.asr = asr_model
        self.asr_processor = asr_processor
        self.clsp = clsp_model
        self.prompt_wav = prompt_wav
        self.cfg = cfg
        self.device = torch.device(device)
        # token2wav internals use bare .cuda() / device='cuda'. When the scorer
        # lives on a non-default GPU we wrap each call in torch.cuda.device(idx).
        self.token2wav_cuda_idx = token2wav_cuda_idx
        # Speaker RAG: when references + their CLSP audio embeddings are given,
        # pick the reference whose voice best matches each prompt's instruction
        # (gender filter -> CLSP rerank) instead of the fixed prompt_wav. None
        # -> disabled (the fixed self.prompt_wav is used everywhere).
        self.references = references
        self.ref_embeddings = ref_embeddings

    def __call__(
        self,
        rollouts,
        ref_text: str,
        instruction: str,
        lang: str,
        v1_tokens: list[int] | None = None,
        clsp_v1: float | None = None,
        wer_v1: float | None = None,
        clsp_v1_reference: str | None = None,
    ) -> list[dict]:
        """Score G rollouts.

        Two-hop refine reward kicks in when a v1 CLSP baseline is available:

            reward = beta * CLSP_v2
                     + lambda_improve * f(CLSP_v2 - CLSP_v1)
                     - wer_penalty * max(0, WER_v2 - wer_ref)

        where f is a configurable non-linear transform (default tanh(scale*x))
        and wer_ref = WER_v1 when the v1 WER is known, else the absolute wer_gate.
        CLSP_v2 enters raw in [-1, 1] (no rescale) so a bad refine earns negative
        reward. The non-linear transform is critical for GRPO: with a
        linear delta, clsp_v1 cancels in the within-group advantage (it's a
        per-prompt constant), so the policy just maximizes absolute CLSP_v2. The
        non-linear f breaks this cancellation, making the advantage genuinely
        depend on improvement over v1.

        `clsp_v1`/`wer_v1` are normally precomputed by
        data_preprocess/build_twohop_prompts.py. Missing baselines fall back to
        on-the-fly CLSP and WER computation from `v1_tokens`. With no usable v1
        baseline, the absolute alpha*WER + beta*CLSP reward is used.
        """
        # Speaker RAG: choose the reference once per prompt (shared by all G
        # rollouts AND the v1 baseline, so clsp_v2 - clsp_v1 stays comparable).
        prompt_wav = self._select_prompt_wav(instruction)
        if (clsp_v1_reference is not None
                and Path(prompt_wav).name != Path(clsp_v1_reference).name):
            raise ValueError(
                "v1/v2 speaker-reference mismatch: "
                f"JSONL baseline used {clsp_v1_reference!r}, "
                f"current reward selected {Path(prompt_wav).name!r}"
            )

        if v1_tokens and (clsp_v1 is None or wer_v1 is None):
            computed_clsp, computed_wer = self._v1_baselines(
                v1_tokens, instruction, ref_text, lang, prompt_wav
            )
            if clsp_v1 is None:
                clsp_v1 = computed_clsp
            if wer_v1 is None:
                wer_v1 = computed_wer

        results: list[dict | None] = [None] * len(rollouts)

        # ---- Phase 1: vocode each rollout; reject empty / un-vocodable audio. ----
        valid_idx: list[int] = []
        valid_wavs: list[torch.Tensor] = []
        for i, r in enumerate(rollouts):
            if not r.audio_codes or len(r.audio_codes) < self.cfg.min_audio_codes:
                results[i] = {
                    "reward": self.cfg.invalid_penalty,
                    "wer": None, "clsp": None, "valid": False,
                    "reason": "empty_audio",
                }
                continue
            try:
                with PROF.section("scoring.vocode"):
                    valid_wavs.append(self._vocode_to_16k_mono(r.audio_codes, prompt_wav))
                valid_idx.append(i)
            except Exception as e:
                results[i] = {
                    "reward": self.cfg.invalid_penalty,
                    "wer": None, "clsp": None, "valid": False,
                    "reason": f"exception:{type(e).__name__}", "error": str(e),
                }

        # ---- Phase 2: batched ASR + CLSP over all valid wavs (fall back to
        # per-sample if the batched path raises). ----
        if valid_wavs:
            try:
                with PROF.section("scoring.asr"):
                    hyps = self._asr_batch(valid_wavs, lang)
                with PROF.section("scoring.clsp"):
                    clsps = self._clsp_score_batch(valid_wavs, instruction)
            except Exception:
                hyps, clsps = [], []
                for wav_16k in valid_wavs:
                    hyps.append(self._asr(wav_16k, lang))
                    clsps.append(self._clsp_score(wav_16k, instruction))
            for j, i in enumerate(valid_idx):
                results[i] = self._reward_from_metrics(
                    hyps[j], clsps[j], ref_text, lang, clsp_v1, wer_v1
                )
        return results

    @staticmethod
    def _apply_improve_fn(delta: float, fn: str, scale: float) -> float:
        """Non-linear transform on clsp_v2 - clsp_v1.

        In GRPO, a linear delta's v1 component cancels in the within-group
        advantage (per-prompt constant). A non-linear f(delta) breaks this:
        f(a-c) - mean(f(a_i-c)) != f(a) - mean(f(a_i)), so the advantage
        genuinely rewards improvement over v1 rather than just absolute CLSP.
        """
        import math
        if fn == "none":
            return delta
        if fn == "tanh":
            return math.tanh(scale * delta)
        if fn == "exp":
            return math.exp(scale * delta) - 1.0
        raise ValueError(f"Unknown improve_fn: {fn!r}")

    def _reward_from_metrics(
        self, hyp: str, c: float, ref_text: str, lang: str,
        clsp_v1: float | None, wer_v1: float | None,
    ) -> dict:
        """Assemble the reward dict from a single rollout's ASR hyp + CLSP score."""
        w = wer(hyp, ref_text, lang)
        clsp_term = c  # keep raw CLSP in [-1, 1]; a bad v2 should earn negative reward
        if clsp_v1 is not None:
            # two-hop reward. CLSP improvement stays relative to v1 (the core
            # mechanism); the WER term is either v1-relative (default) or absolute.
            if self.cfg.wer_absolute:
                # optimize WER directly — no v1 margin; capped for GRPO stability.
                wer_pen = self.cfg.wer_penalty * min(w, self.cfg.wer_cap)
            else:
                wer_ref = wer_v1 if wer_v1 is not None else self.cfg.wer_gate
                wer_pen = self.cfg.wer_penalty * max(0.0, w - wer_ref)
            improve = self._apply_improve_fn(
                c - clsp_v1, self.cfg.improve_fn, self.cfg.improve_scale,
            )
            reward = (self.cfg.beta_clsp * clsp_term
                      + self.cfg.lambda_improve * improve
                      - wer_pen)
            # hard gate: anything past wer_gate gets a fixed cliff penalty on top,
            # to crush off-text "drift" rollouts (high WER, and they also lose CLSP).
            if self.cfg.wer_hard_gate and w > self.cfg.wer_gate:
                reward += self.cfg.gate_penalty
        else:
            # one-hop absolute reward: CLSP minus a capped, MONOTONE WER penalty.
            # reward = beta_clsp*CLSP - alpha_wer*min(WER, wer_cap).
            # The cap bounds the penalty (and the GRPO group-advantage variance from
            # garbage rollouts with WER>>1) while keeping a non-zero gradient for any
            # WER below the cap — unlike the old max(0, 1-WER) floor, which went flat
            # (zero gradient) for WER>=1 and let the policy drift to unintelligible audio.
            wer_pen = self.cfg.alpha_wer * min(w, self.cfg.wer_cap)
            reward = self.cfg.beta_clsp * clsp_term - wer_pen
        result = {
            "reward": float(reward),
            "wer": float(w),
            "clsp": float(c),
            "clsp_v1": (float(clsp_v1) if clsp_v1 is not None else None),
            "wer_v1": (float(wer_v1) if wer_v1 is not None else None),
            "valid": True,
            "hyp": hyp,
        }
        if clsp_v1 is not None:
            result["clsp_delta"] = float(c - clsp_v1)
            result["improve_transformed"] = float(improve)
        return result

    def _select_prompt_wav(self, instruction: str) -> str:
        """Pick the reference for this prompt. Fixed prompt_wav unless speaker
        RAG is enabled (references + embeddings supplied)."""
        if self.references is None or self.ref_embeddings is None:
            return self.prompt_wav
        from speaker_rag import select_reference  # lazy: optional dependency
        res = select_reference(
            instruction, self.clsp, self.device,
            self.references, self.ref_embeddings,
        )
        return res.reference.wav

    def _v1_baselines(self, v1_tokens: list[int], instruction: str,
                      ref_text: str, lang: str,
                      prompt_wav: str | None = None
                      ) -> tuple[float | None, float | None]:
        """Compute CLSP(v1) and WER(v1) from one shared vocoder pass."""
        # v1_tokens may be raw LM ids (>=offset) or already-shifted codes (0..6560).
        if any(t >= AUDIO_TOKEN_OFFSET for t in v1_tokens):
            codes = [t - AUDIO_TOKEN_OFFSET for t in v1_tokens if t >= AUDIO_TOKEN_OFFSET]
        else:
            codes = [t for t in v1_tokens if 0 <= t < AUDIO_TOKEN_OFFSET]
        if len(codes) < self.cfg.min_audio_codes:
            return None, None
        try:
            wav_16k = self._vocode_to_16k_mono(codes, prompt_wav)
            clsp_v1 = self._clsp_score(wav_16k, instruction)
            hyp_v1 = self._asr(wav_16k, lang)
            return clsp_v1, wer(hyp_v1, ref_text, lang)
        except Exception:
            return None, None

    def _vocode_to_16k_mono(self, audio_codes: list[int],
                            prompt_wav: str | None = None) -> torch.Tensor:
        """audio_codes -> wav bytes -> [1, T] float32 mono @ 16kHz on CPU."""
        import soundfile as sf  # local import keeps module import light
        wav_bytes = self.vocode_bytes(audio_codes, prompt_wav)
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T.copy())  # [C, T]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            import torchaudio.functional as AF  # local import
            wav = AF.resample(wav, sr, TARGET_SR)
        return wav  # [1, T] cpu float32

    def vocode_bytes(self, audio_codes: list[int],
                     prompt_wav: str | None = None) -> bytes:
        """audio_codes -> WAV bytes using `prompt_wav` (fixed default if None)."""
        ref = prompt_wav if prompt_wav is not None else self.prompt_wav
        if self.token2wav_cuda_idx is not None:
            with torch.cuda.device(self.token2wav_cuda_idx):
                return self.token2wav(audio_codes, prompt_wav=ref)
        return self.token2wav(audio_codes, prompt_wav=ref)

    def save_audio_codes(self, audio_codes: list[int], path: str) -> None:
        """Persist audio_codes as a wav file for manual listening."""
        with open(path, "wb") as f:
            f.write(self.vocode_bytes(audio_codes))

    @torch.no_grad()
    def _asr(self, wav_16k: torch.Tensor, lang: str) -> str:
        feats = self.asr_processor(
            wav_16k.squeeze(0).numpy(),
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in feats.items()}
        # cast input_features to ASR weight dtype (whisper is bf16/fp16 below)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.asr.dtype)
        gen_kwargs = {"max_new_tokens": 256}
        forced = self.asr_processor.get_decoder_prompt_ids(
            language=("zh" if lang == "zh" else "en"),
            task="transcribe",
        ) if hasattr(self.asr_processor, "get_decoder_prompt_ids") else None
        if forced:
            gen_kwargs["forced_decoder_ids"] = forced
        ids = self.asr.generate(**inputs, **gen_kwargs)
        return self.asr_processor.batch_decode(ids, skip_special_tokens=True)[0]

    @torch.no_grad()
    def _clsp_score(self, wav_16k: torch.Tensor, instruction: str) -> float:
        wav = wav_16k.to(self.device)
        wav_lens = torch.tensor([wav.size(1)], device=self.device)
        af, tf, _ = self.clsp(wav, wav_lens, [instruction])
        af = F.normalize(af, dim=-1)
        tf = F.normalize(tf, dim=-1)
        return float((af * tf).sum(dim=-1).squeeze().cpu())

    @torch.no_grad()
    def _asr_batch(self, wavs_16k: list[torch.Tensor], lang: str) -> list[str]:
        """Transcribe a batch of mono-16k wavs in one Whisper forward.

        Whisper's feature extractor pads/truncates every clip to 30 s, so the
        batch is length-uniform regardless of the input wav durations.
        """
        feats = self.asr_processor(
            [w.squeeze(0).numpy() for w in wavs_16k],
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in feats.items()}
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.asr.dtype)
        gen_kwargs = {"max_new_tokens": 256}
        forced = self.asr_processor.get_decoder_prompt_ids(
            language=("zh" if lang == "zh" else "en"),
            task="transcribe",
        ) if hasattr(self.asr_processor, "get_decoder_prompt_ids") else None
        if forced:
            gen_kwargs["forced_decoder_ids"] = forced
        ids = self.asr.generate(**inputs, **gen_kwargs)
        return self.asr_processor.batch_decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def _clsp_score_batch(self, wavs_16k: list[torch.Tensor], instruction: str) -> list[float]:
        """CLSP cosine similarity for a batch of wavs against one instruction."""
        lengths = [w.size(1) for w in wavs_16k]
        max_len = max(lengths)
        B = len(wavs_16k)
        padded = torch.zeros(B, max_len)
        for i, w in enumerate(wavs_16k):
            padded[i, : w.size(1)] = w.squeeze(0)
        wav = padded.to(self.device)
        wav_lens = torch.tensor(lengths, device=self.device)
        af, tf, _ = self.clsp(wav, wav_lens, [instruction] * B)
        af = F.normalize(af, dim=-1)
        tf = F.normalize(tf, dim=-1)
        return (af * tf).sum(dim=-1).cpu().tolist()


def asr_load(name: str = "openai/whisper-large-v3", device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(name, torch_dtype=dtype)
    model = model.to(device).eval()
    return model, processor


def clsp_load(name: str = "yfyeung/CLSP", device="cuda"):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
    return model


def wer(hyp: str, ref: str, lang: str) -> float:
    """WER for en, CER for zh. Both via jiwer; returns float in [0, ~1+].

    Applies text normalization before scoring so that case and punctuation
    differences between ASR output and ref text don't inflate the error rate.
    """
    import jiwer
    import jiwer.transforms as tr
    h = (hyp or "").strip()
    r = (ref or "").strip()
    if not r:
        return 0.0 if not h else 1.0
    if lang == "zh":
        # CER: strip spaces so character boundary differences don't add errors
        h_norm = h.replace(" ", "")
        r_norm = r.replace(" ", "")
        return float(jiwer.cer(r_norm, h_norm))
    _en_transform = tr.Compose([
        tr.RemovePunctuation(),
        tr.ToLowerCase(),
        tr.Strip(),
        tr.ReduceToListOfListOfWords(),
    ])
    return float(jiwer.wer(r, h,
                           reference_transform=_en_transform,
                           hypothesis_transform=_en_transform))
