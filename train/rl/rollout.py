"""Sample G rollouts from the policy and parse into (think_text, audio_codes).

Three phases:
  1. `sample_rollouts(...)`        — two-pass generation per rollout:
                                     pass-1 generates think text (stop at </think>),
                                     pass-2 injects <tts_start> and generates audio.
                                     Returns Rollout list.
  2. `policy_logprobs(...)`        — teacher-forcing forward on the *current*
                                     policy with grad enabled (used for the
                                     GRPO loss).
  3. `reference_logprobs(...)`     — same but with PEFT adapter disabled
                                     (frozen base, no grad). Used for the KL term.

Token-vocab boundaries match Step-Audio2's existing scripts:
  text:   t < 151688
  audio:  t >= 151696, codes = (t - 151696) ∈ [0, 6561)

## Why two-pass rollout

Single-pass rollout (prefill="<think>", hope the model emits <tts_start> after
</think>) fails: the instruct model has no training signal for that autonomous
transition and generates a prose text answer instead of audio tokens.

The official examples-think.py shows the correct pattern:
  pass-1  stop_strings=["</think>"]  → think text
  pass-2  inject "</think>\\n<tts_start>"  → audio tokens

We replicate this during rollout.  The gen_ids stored in each Rollout are the
CONCATENATION of both passes (think tokens + bridge tokens + audio tokens) so
that policy_logprobs / reference_logprobs still get the full sequence.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList


AUDIO_TOKEN_VOCAB_SIZE = 6561  # codes per the s3tokenizer codebook
AUDIO_TOKEN_OFFSET = 151696    # add to LM-token id to get audio code
TEXT_TOKEN_MAX = 151688        # t < this is normal text
TTS_END_TOKEN_ID = 151694      # <tts_end>
EOT_TOKEN_ID = 151665          # <|EOT|>


class _SanitizeLogits(LogitsProcessor):
    """Replace nan/±inf logits with finite values so multinomial sampling can't
    trip a CUDA device-side assert ("probability tensor contains inf/nan").

    The bf16 forward occasionally emits a nan/inf logit (rare, more likely as the
    policy sharpens under higher LR/temperature). Natural twohop generation runs
    with NO _AudioOnlyMask, so without this the raw logits go straight into
    softmax→multinomial and crash. Applied ALWAYS, before any sampling.
    """

    def __call__(self, input_ids, scores):
        return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)


class _AudioOnlyMask(LogitsProcessor):
    """Pass-2 mask: keep only audio tokens (>=AUDIO_TOKEN_OFFSET), <tts_end>,
    and <|EOT|>. Everything else (text, padding range, <tts_start>) gets -inf.

    Stateless — the mask is the same at every step of pass 2 because pass 2
    starts with `<tts_start>` already in the prompt prefix.
    """

    def __init__(self):
        self.keep: torch.Tensor | None = None

    def __call__(self, input_ids, scores):
        V = scores.shape[-1]
        if (self.keep is None
                or self.keep.device != scores.device
                or self.keep.numel() != V):
            keep = torch.zeros(V, dtype=torch.bool, device=scores.device)
            keep[AUDIO_TOKEN_OFFSET:] = True
            keep[TTS_END_TOKEN_ID] = True
            keep[EOT_TOKEN_ID] = True
            self.keep = keep
        # Sanitize the KEPT logits before excluding the rest: as the policy
        # sharpens, the bf16 forward occasionally emits nan/+-inf on an audio
        # token, which flows into softmax -> multinomial and trips a CUDA
        # device-side assert ("probability tensor contains inf, nan ...").
        # nan_to_num + clamp keeps the audio distribution finite and valid;
        # non-audio tokens are then forced to -inf (prob 0) as before.
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        return scores.masked_fill(~self.keep, float("-inf"))


@dataclass
class Rollout:
    prompt_ids:    torch.LongTensor    # [T_prompt]   shared across G rollouts
    gen_ids:       torch.LongTensor    # [T_gen_i]    trimmed to first EOS
    think_text:    str
    audio_codes:   list                # in 0..6560
    truncated:     bool                # hit max_new_tokens before EOS


def _build_prompt_ids(model, messages: list) -> torch.LongTensor:
    """Mirror StepAudio2.__call__'s tokenization to obtain prompt_ids: [1, T]."""
    msg_segs, mels = model.apply_chat_template(messages)
    if mels:
        raise ValueError("RL prompt is text-only; got audio mels")
    parts: list[torch.Tensor] = []
    for seg in msg_segs:
        if isinstance(seg, str):
            parts.append(
                model.llm_tokenizer(seg, return_tensors="pt", padding=False)["input_ids"]
            )
        elif isinstance(seg, list):  # pre-tokenized audio token ids spliced via {"type":"token"}
            parts.append(torch.tensor([seg], dtype=torch.long))
        else:
            raise ValueError(f"Unsupported chat-template segment: {type(seg)}")
    return torch.cat(parts, dim=-1).long()  # [1, T_prompt]


def _split_gen_ids(gen_ids: torch.LongTensor, tokenizer) -> tuple[str, list[int]]:
    """Split a single rollout's gen_ids into (think_text, audio_codes)."""
    ids = gen_ids.tolist()
    text_ids = [t for t in ids if t < TEXT_TOKEN_MAX]
    audio_ids = [t - AUDIO_TOKEN_OFFSET for t in ids if t >= AUDIO_TOKEN_OFFSET]
    audio_codes = [c for c in audio_ids if 0 <= c < AUDIO_TOKEN_VOCAB_SIZE]
    raw = tokenizer.decode(text_ids, skip_special_tokens=False)
    think_text = raw.split("</think>", 1)[0].lstrip()
    return think_text, audio_codes


def _generate_one(
    model,
    prompt_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
    stop_strings: list[str] | None = None,
    eos_token_id: int,
    pad_token_id: int,
    logits_processor: LogitsProcessorList | None = None,
) -> torch.LongTensor:
    """Single-sequence (batch=1) generate. Returns gen_ids [T_gen] on CPU."""
    device = prompt_ids.device
    attention_mask = torch.ones_like(prompt_ids)
    kwargs: dict = dict(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        wavs=None,
        wav_lens=None,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        return_dict_in_generate=True,
        output_scores=False,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        tokenizer=model.llm_tokenizer,
    )
    if stop_strings:
        kwargs["stop_strings"] = stop_strings
    if logits_processor is not None:
        kwargs["logits_processor"] = logits_processor
    out = model.llm.generate(**kwargs)
    T_prompt = prompt_ids.shape[-1]
    return out.sequences[0, T_prompt:].cpu()


def _generate_batch(
    model,
    prompt_ids: torch.LongTensor,
    G: int,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
    eos_token_id: int,
    pad_token_id: int,
    logits_processor: LogitsProcessorList | None = None,
) -> torch.LongTensor:
    """Generate G sequences in a single batched `generate()` call.

    `prompt_ids` is [1, T] (a single shared prompt); it is expanded to [G, T] so
    all G rollouts are produced in one forward stream instead of G separate
    batch-1 calls. Each row samples independently (do_sample). Sequences that hit
    EOS early are right-padded with pad_token_id by HF; the caller trims per row.

    Returns gen_ids [G, T_gen] on CPU (T_gen = longest generation in the batch).
    """
    batch = prompt_ids.expand(G, -1).contiguous()
    attention_mask = torch.ones_like(batch)
    kwargs: dict = dict(
        input_ids=batch,
        attention_mask=attention_mask,
        wavs=None,
        wav_lens=None,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        return_dict_in_generate=True,
        output_scores=False,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        tokenizer=model.llm_tokenizer,
    )
    if logits_processor is not None:
        kwargs["logits_processor"] = logits_processor
    out = model.llm.generate(**kwargs)
    T_prompt = batch.shape[-1]
    return out.sequences[:, T_prompt:].cpu()  # [G, T_gen]


def _trim_at_eos(gen_row: torch.LongTensor, eos_id: int) -> tuple[torch.LongTensor, bool]:
    """Trim a single generated row at (and including) the first EOS.

    Returns (trimmed_ids, truncated) where truncated=True means no EOS was found
    within the budget.
    """
    eos_pos = (gen_row == eos_id).nonzero(as_tuple=False)
    if eos_pos.numel() > 0:
        return gen_row[: int(eos_pos[0]) + 1], False
    return gen_row, True


@torch.no_grad()
def sample_rollouts(
    model,
    messages: list,
    *,
    G: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    eos_token_id: int | None = None,
    do_sample: bool = True,
    max_think_tokens: int = 512,
) -> list[Rollout]:
    """Sample G rollouts using two-pass generation.

    Pass 1: generate think text (stop at </think>), max_think_tokens budget.
    Pass 2: inject "</think>\\n<tts_start>" into prompt, generate audio tokens,
            max_new_tokens budget.

    gen_ids in each Rollout = think_ids ++ bridge_ids ++ audio_ids so that
    policy_logprobs / reference_logprobs see the full sequence.
    """
    from .prompt import build_rl_prompt_pass2  # avoid circular at module level

    device = next(model.llm.parameters()).device
    eos_id = eos_token_id if eos_token_id is not None else model.eos_token_id
    pad_id = model.llm_tokenizer.pad_token_id or eos_id

    # Tokenize the </think>\n<tts_start> bridge once (same for every rollout).
    bridge_str = "\n</think>\n<tts_start>"
    bridge_ids = model.llm_tokenizer(
        bridge_str, return_tensors="pt", add_special_tokens=False
    )["input_ids"].squeeze(0)  # [T_bridge]

    # Pass-2 logits processor: restrict to audio tokens + <tts_end> + <|EOT|>.
    # Built once and shared across all G rollouts (the keep mask is cached
    # inside the processor on first call).
    pass2_logits_processor = LogitsProcessorList([_AudioOnlyMask()])

    rollouts: list[Rollout] = []
    prompt_cpu = _build_prompt_ids(model, messages).squeeze(0).cpu()

    for _ in range(G):
        # ---- Pass 1: think ----
        p1_ids = prompt_cpu.unsqueeze(0).to(device)
        think_gen = _generate_one(
            model, p1_ids,
            max_new_tokens=max_think_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            stop_strings=["</think>"],
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )  # [T_think]  may or may not end with </think> token(s)

        # Decode think text, strip trailing </think> so build_rl_prompt_pass2
        # can reconstruct the boundary cleanly.
        think_text_ids = [t for t in think_gen.tolist() if t < TEXT_TOKEN_MAX]
        think_raw = model.llm_tokenizer.decode(think_text_ids, skip_special_tokens=False)
        think_text = think_raw.split("</think>", 1)[0].strip()

        # ---- Pass 2: audio ----
        msgs_p2 = build_rl_prompt_pass2(messages, think_text)
        p2_ids = _build_prompt_ids(model, msgs_p2).to(device)
        audio_gen = _generate_one(
            model, p2_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            stop_strings=None,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            logits_processor=pass2_logits_processor,
        )  # [T_audio]

        # ---- Combine: gen_ids = think_gen + bridge + audio_gen ----
        # Strip the think_gen trailing EOS/pad so it doesn't confuse logprob.
        think_trimmed = think_gen
        if think_trimmed.numel() > 0 and think_trimmed[-1] == eos_id:
            think_trimmed = think_trimmed[:-1]

        # Trim audio at first EOS.
        eos_pos = (audio_gen == eos_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            cutoff = int(eos_pos[0]) + 1
            truncated = False
        else:
            cutoff = audio_gen.numel()
            truncated = True
        audio_trimmed = audio_gen[:cutoff]

        gen_ids = torch.cat([think_trimmed, bridge_ids, audio_trimmed])

        # Parse audio codes from the audio pass only.
        audio_ids = audio_trimmed.tolist()
        audio_codes = [
            t - AUDIO_TOKEN_OFFSET
            for t in audio_ids
            if t >= AUDIO_TOKEN_OFFSET and (t - AUDIO_TOKEN_OFFSET) < AUDIO_TOKEN_VOCAB_SIZE
        ]

        rollouts.append(
            Rollout(
                prompt_ids=prompt_cpu,
                gen_ids=gen_ids,
                think_text=think_text,
                audio_codes=audio_codes,
                truncated=truncated,
            )
        )
    return rollouts


@torch.no_grad()
def sample_rollouts_twohop(
    model,
    messages: list,
    *,
    G: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    eos_token_id: int | None = None,
    do_sample: bool = True,
    audio_only_mask: bool = False,
) -> list[Rollout]:
    """Single-pass audio generation for twohop refine mode.

    The messages already contain v1_tokens + critique (built by
    build_twohop_prompt_chat), with the assistant turn pre-filled to
    '<tts_start>' (eot=False).  Generation continues directly with audio
    tokens — no think phase.

    `audio_only_mask` (default False): when True, restrict every step to the
    audio-token range via `_AudioOnlyMask`. This was the original behaviour but
    a WER sweep showed the mask is what drove RL
    v2 WER to ~1.1 — under natural generation the *same* prompt yields WER ~0.13
    because the model picks its own audio/stop boundary. Default is now natural
    generation; gen_ids may include a few non-audio tokens (filtered out of
    `audio_codes`) but the full sequence is still correct for the GRPO logprob.

    Returns Rollout objects with think_text='' and gen_ids = full generation.
    """
    device = next(model.llm.parameters()).device
    eos_id = eos_token_id if eos_token_id is not None else model.eos_token_id
    pad_id = model.llm_tokenizer.pad_token_id or eos_id

    # Always sanitize logits (guards multinomial against nan/inf); add the
    # audio-only mask on top only when requested.
    procs = [_SanitizeLogits()]
    if audio_only_mask:
        procs.append(_AudioOnlyMask())
    logits_proc = LogitsProcessorList(procs)
    prompt_cpu = _build_prompt_ids(model, messages).squeeze(0).cpu()
    p_ids = prompt_cpu.unsqueeze(0).to(device)

    # Single batched generate for all G rollouts (shared prompt -> no left pad).
    gen_batch = _generate_batch(
        model, p_ids, G,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        logits_processor=logits_proc,
    )  # [G, T_gen] on CPU

    rollouts: list[Rollout] = []
    for g in range(gen_batch.size(0)):
        audio_trimmed, truncated = _trim_at_eos(gen_batch[g], eos_id)
        audio_codes = [
            t - AUDIO_TOKEN_OFFSET
            for t in audio_trimmed.tolist()
            if t >= AUDIO_TOKEN_OFFSET and (t - AUDIO_TOKEN_OFFSET) < AUDIO_TOKEN_VOCAB_SIZE
        ]
        rollouts.append(Rollout(
            prompt_ids=prompt_cpu,
            gen_ids=audio_trimmed,
            think_text="",
            audio_codes=audio_codes,
            truncated=truncated,
        ))
    return rollouts


@torch.no_grad()
def sample_rollouts_twohop_selfcritique(
    model,
    messages: list,
    *,
    G: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    eos_token_id: int | None = None,
    do_sample: bool = True,
    max_think_tokens: int = 256,
) -> list[Rollout]:
    """Two-pass SELF-critique rollout for two-hop refine mode.

    `messages` is the pass-1 layout from `build_twohop_selfcritique_chat`: the
    turn-2 assistant is open at `<think>` and the v1 audio tokens are already
    spliced into turn-1. The policy generates its OWN critique, then the v2 audio:

      Pass 1: generate the critique (stop at </think>), max_think_tokens budget.
      Pass 2: inject `</think>\\n<tts_start>` and generate v2 audio with NATURAL
              generation (sanitize only, NO _AudioOnlyMask) — the mask is what
              drove twohop WER to ~1.1 (see sample_rollouts_twohop).

    gen_ids = critique_ids ++ bridge_ids ++ audio_ids, so the critique tokens are
    part of the trained completion and the GRPO gradient shapes critique quality.
    This is the on-policy analogue of `sample_rollouts_twohop` (which conditions on
    a frozen, precomputed critique that never receives gradient).

    Per-rollout loop (not batched): pass-1 critique lengths differ across the G
    rollouts, so a single batched pass-2 would need left-padding. Correctness over
    speed here — the static path stays batched.
    """
    from .prompt import build_rl_prompt_pass2  # avoid circular at module level

    device = next(model.llm.parameters()).device
    eos_id = eos_token_id if eos_token_id is not None else model.eos_token_id
    pad_id = model.llm_tokenizer.pad_token_id or eos_id

    # Bridge tokens injected between the (generated) critique and the audio pass.
    bridge_str = "\n</think>\n<tts_start>"
    bridge_ids = model.llm_tokenizer(
        bridge_str, return_tensors="pt", add_special_tokens=False
    )["input_ids"].squeeze(0)

    # Always sanitize logits (guards multinomial against nan/inf) in both passes.
    sanitize = LogitsProcessorList([_SanitizeLogits()])
    prompt_cpu = _build_prompt_ids(model, messages).squeeze(0).cpu()

    rollouts: list[Rollout] = []
    for _ in range(G):
        # ---- Pass 1: critique ----
        p1_ids = prompt_cpu.unsqueeze(0).to(device)
        think_gen = _generate_one(
            model, p1_ids,
            max_new_tokens=max_think_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            stop_strings=["</think>"],
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            logits_processor=sanitize,
        )  # [T_think]
        think_text_ids = [t for t in think_gen.tolist() if t < TEXT_TOKEN_MAX]
        think_raw = model.llm_tokenizer.decode(think_text_ids, skip_special_tokens=False)
        critique_text = think_raw.split("</think>", 1)[0].strip()

        # ---- Pass 2: audio (natural generation) ----
        msgs_p2 = build_rl_prompt_pass2(messages, critique_text)
        p2_ids = _build_prompt_ids(model, msgs_p2).to(device)
        audio_gen = _generate_one(
            model, p2_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            stop_strings=None,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            logits_processor=sanitize,
        )  # [T_audio]

        # ---- Combine: gen_ids = critique + bridge + audio ----
        think_trimmed = think_gen
        if think_trimmed.numel() > 0 and think_trimmed[-1] == eos_id:
            think_trimmed = think_trimmed[:-1]
        audio_trimmed, truncated = _trim_at_eos(audio_gen, eos_id)
        gen_ids = torch.cat([think_trimmed, bridge_ids, audio_trimmed])

        audio_codes = [
            t - AUDIO_TOKEN_OFFSET
            for t in audio_trimmed.tolist()
            if t >= AUDIO_TOKEN_OFFSET and (t - AUDIO_TOKEN_OFFSET) < AUDIO_TOKEN_VOCAB_SIZE
        ]
        rollouts.append(Rollout(
            prompt_ids=prompt_cpu,
            gen_ids=gen_ids,
            think_text=critique_text,
            audio_codes=audio_codes,
            truncated=truncated,
        ))
    return rollouts


def policy_logprobs(
    llm,
    prompt_ids: torch.LongTensor,
    gen_ids: torch.LongTensor,
    attention_mask: torch.LongTensor | None = None,
) -> torch.FloatTensor:
    """Teacher-forcing logprobs of `gen_ids` under the **current** LM (with grad).

    prompt_ids: [B, T_prompt]
    gen_ids:    [B, T_gen]   (right-padded with pad_id; mask handled by caller)
    Returns:    [B, T_gen]   log π_θ(g_t | g_<t, prompt)
    """
    inputs = torch.cat([prompt_ids, gen_ids], dim=-1)
    if attention_mask is None:
        attention_mask = torch.ones_like(inputs)
    out = llm(
        input_ids=inputs,
        attention_mask=attention_mask,
        wavs=None,
        wav_lens=None,
        use_cache=False,
    )
    T_prompt = prompt_ids.shape[-1]
    # logits at position i predict token i+1; target gen_ids[t] sits at full
    # position T_prompt + t, so we need logits[T_prompt + t - 1].
    target_logits = out.logits[:, T_prompt - 1 : T_prompt - 1 + gen_ids.shape[-1], :]
    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    return log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)


def reference_logprobs(
    llm,
    prompt_ids: torch.LongTensor,
    gen_ids: torch.LongTensor,
    attention_mask: torch.LongTensor | None = None,
) -> torch.FloatTensor:
    """Same as `policy_logprobs` but with the PEFT adapter disabled (frozen base).

    Falls back to a no-op context if `llm` is not PEFT-wrapped — useful for the
    cold-start dry run before LoRA is attached.
    """
    ctx = llm.disable_adapter() if hasattr(llm, "disable_adapter") else nullcontext()
    with torch.no_grad(), ctx:
        return policy_logprobs(llm, prompt_ids, gen_ids, attention_mask).detach()
