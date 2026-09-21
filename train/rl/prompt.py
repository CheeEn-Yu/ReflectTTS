"""RL prompt builder for Step-Audio-2.

## Why two-pass

Step-Audio-2's official examples-think.py (and every working Think demo) use a
two-pass generation loop:

  Pass 1  prefill="<think>", stop_strings=["</think>"]  → collect think_text
  Pass 2  inject "</think>\n<tts_start>", continue      → collect audio tokens

Single-pass (hoping the model autonomously emits <tts_start> after </think>)
does not work reliably — the instruct model has no training signal for that
transition and defaults to a prose text response instead.

## Usage in RL rollouts

The rollout code (rollout.py) must call `sample_rollout_two_pass` rather than
vanilla model.generate.  The combined token sequence (think_ids + audio_ids) is
then used for GRPO logprob computation exactly as before.

## Prompt templates

`TEMPLATES` contains the prompt variants used during development. The default
is `"format_spec"`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System-prompt templates
# Each value: format-string with {instruction}.
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, str] = {
    # Describes exact output format; most explicit about <tts_start> placement.
    "format_spec": (
        "You are a text-to-speech synthesizer. Follow this EXACT output format:\n"
        "<think>\n[reason about speaking style]\n</think>\n"
        "<tts_start>[audio tokens]\n\n"
        "Speaking style: {instruction}\n\n"
        "The human turn contains TEXT TO VOCALIZE, not a question to answer. "
        "You must NEVER write a prose response. Only vocalize."
    ),

    # TTS system role with explicit negative constraint.
    "tts_role_neg": (
        "You are a TTS engine, not a conversational assistant. "
        "Your only output is speech audio.\n\n"
        "Speaking style: {instruction}\n\n"
        "Briefly reason in <think>...</think> about pitch, pace, and emotion. "
        "Then emit <tts_start> and output audio tokens. "
        "Do NOT write any prose response after </think>."
    ),

    # Instruction + text both in system, human is neutral trigger.
    "sys_contains_text": (
        "You are a TTS synthesizer. Vocalize the following text exactly "
        "as written, applying the speaking style.\n\n"
        "Speaking style: {instruction}\n\n"
        "Text to speak: {text}"
    ),

    # Audio book narrator framing.
    "audiobook_reader": (
        "You are a professional audio book narrator. "
        "You hear a direction and then read the passage that follows aloud.\n"
        "Direction: {instruction}\n\n"
        "Briefly reflect in <think>...</think> on how to embody the direction, "
        "then narrate after <tts_start>. Never discuss the content."
    ),

    # Original (kept for ablation comparison in the sweep).
    "original": (
        "Read the following text aloud in the speaking style described below:"
        "{instruction}\n"
        "Briefly reason inside <think>...</think> about how to "
        "apply the instruction (pitch, pace, emotion, persona, etc.) to your speaking style and then"
        "read the following text aloud in the speaking style described above after <tts_start>\n"
    ),
}

DEFAULT_TEMPLATE = "format_spec"

# ---------------------------------------------------------------------------
# Two-hop prompt templates shared by training and inference.
# ---------------------------------------------------------------------------

_TWOHOP_TURN1: dict[str, str] = {
    "en": "{instruction}\nRead the following text aloud in the speaking style described above.\n",
    "zh": "{instruction}\n请按照以上描述的风格朗读下面的文字。\n",
}

_TWOHOP_RETRY: dict[str, str] = {
    "en": (
        "Reviewer critique of your last attempt:\n{critique}\n\n"
        "Please read the same text again, applying the critique:\n{text}"
    ),
    "zh": (
        "对你上次朗读的点评:\n{critique}\n\n"
        "请再次朗读相同文字,根据点评修正:\n{text}"
    ),
}

# Self-critique retry: the model writes its OWN critique (in a <think> block)
# instead of being handed one. No {critique} placeholder — the critique is
# generated on-policy so the GRPO gradient can shape it.
_TWOHOP_SELFCRITIQUE_RETRY: dict[str, str] = {
    "en": (
        "Critique your last attempt: judge how well it matched the required "
        "speaking style and name concrete acoustic fixes (pitch, speed, volume, "
        "emotion, persona). Then read the same text again applying your critique:\n{text}"
    ),
    "zh": (
        "请点评你上次的朗读:判断它与要求风格的差距,并指出具体的声学修正"
        "(音高、语速、音量、情绪、人设)。然后根据你的点评再次朗读相同文字:\n{text}"
    ),
}


def build_twohop_prompt_chat(
    instruction: str,
    text: str,
    lang: str | None = None,
    *,
    v1_tokens: list[int],
    critique: str,
    **_kwargs,
) -> list:
    """Two-hop refine prompt for RL rollout.

    Message layout:
      turn-1: system(instruction) + human(text) + assistant(v1 audio tokens)
      turn-2: human(critique + retry) + assistant(<tts_start>, eot=False)

    v1_tokens: raw LM token IDs (>=151696) from a prior generation or
               s3tokenizer output with +AUDIO_TOKEN_OFFSET applied.
    critique:  text critique of v1 from LALM or text_critic.py.
    """
    lng = lang or "en"
    sys_prompt = _TWOHOP_TURN1.get(lng, _TWOHOP_TURN1["en"]).format(
        instruction=instruction.strip()
    ).rstrip()
    retry_content = _TWOHOP_RETRY.get(lng, _TWOHOP_RETRY["en"]).format(
        critique=(critique or "").strip(),
        text=text,
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "human", "content": [{"type": "text", "text": text}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "<tts_start>"},
            {"type": "token", "token": v1_tokens},
        ]},
        {"role": "human", "content": retry_content},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]


def build_twohop_selfcritique_chat(
    instruction: str,
    text: str,
    lang: str | None = None,
    *,
    v1_tokens: list[int],
    **_kwargs,
) -> list:
    """Two-hop refine prompt where the POLICY generates the critique itself.

    Contrast with `build_twohop_prompt_chat`, which injects a *precomputed*
    critique into the prompt: that critique is frozen prompt context, so the GRPO
    gradient never reaches it and RL cannot improve critique quality. Here the
    turn-2 assistant is left open at `\\n<think>\\n` (eot=False) so the rollout's
    pass-1 generates the critique inside a <think> block; pass-2 then injects
    `</think>\\n<tts_start>` and generates the v2 audio. The critique tokens land
    in the rollout's gen_ids → they are part of the trained completion → the
    reward shapes the critique.

    Layout (mirrors build_twohop_prompt_chat turn-1; only turn-2 differs):
      turn-1: system(instruction) + human(text) + assistant(v1 audio tokens)
      turn-2: human(self-critique request + text) + assistant prefilled "<think>"

    `**_kwargs` absorbs `critique=...` passed by the trainer (unused here).
    """
    lng = lang or "en"
    sys_prompt = _TWOHOP_TURN1.get(lng, _TWOHOP_TURN1["en"]).format(
        instruction=instruction.strip()
    ).rstrip()
    retry_content = _TWOHOP_SELFCRITIQUE_RETRY.get(
        lng, _TWOHOP_SELFCRITIQUE_RETRY["en"]
    ).format(text=text)
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "human", "content": [{"type": "text", "text": text}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "<tts_start>"},
            {"type": "token", "token": v1_tokens},
        ]},
        {"role": "human", "content": retry_content},
        {"role": "assistant", "content": "\n<think>\n", "eot": False},
    ]


def build_critique_prompt_chat(
    instruction: str,
    text: str,
    lang: str | None = None,
    *,
    v1_tokens: list[int],
    **_kwargs,
) -> list:
    """Critique prompt for vLLM: instruction + text + v1 → text critique.

    Keep v1 in the role and format in which the model generated it: an
    assistant turn beginning with ``<tts_start>``. The following human turn asks
    for the structured critique, and the final assistant turn is left open.
    """
    from infer.text_critic import critic_prompt
    lng = lang or "en"
    request = critic_prompt(lng).format(
        instruction=instruction.strip(),
        text=text.strip(),
    )
    sys_prompt = _TWOHOP_TURN1.get(lng, _TWOHOP_TURN1["en"]).format(
        instruction=instruction.strip()
    ).rstrip()
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "human", "content": [{"type": "text", "text": text}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "<tts_start>"},
            {"type": "token", "token": v1_tokens},
        ]},
        {"role": "human", "content": request},
        {"role": "assistant", "content": None},
    ]


def build_singlepass_prompt_chat(
    instruction: str,
    text: str,
    lang: str | None = None,
    **_kwargs,
) -> list:
    """Single-pass TTS prompt: instruction + text -> direct audio generation.

    Same system prompt as two-hop turn-1 but without the v1 audio / critique
    turns. Used as the baseline half of mixed-group GRPO so the advantage
    directly rewards "two-hop with critique > blind single-pass".
    """
    lng = lang or "en"
    sys_prompt = _TWOHOP_TURN1.get(lng, _TWOHOP_TURN1["en"]).format(
        instruction=instruction.strip()
    ).rstrip()
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "human", "content": [{"type": "text", "text": text}]},
        {"role": "assistant", "content": "<tts_start>", "eot": False},
    ]


def build_rl_prompt_chat(
    instruction: str,
    text: str,
    lang: str | None = None,
    template: str = DEFAULT_TEMPLATE,
    **_kwargs,
) -> list:
    """Return messages list for pass-1 of two-pass rollout.

    Pass-1 prefills the assistant with "<think>\\n" so generation starts inside
    the think block.  The caller stops at stop_strings=["</think>"] then calls
    `build_rl_prompt_pass2` to inject <tts_start> for audio generation.

    `lang` is accepted for API compatibility (passed through to the reward
    scorer) but the prompt template is always English.
    """
    sys_template = TEMPLATES[template]

    if template == "sys_contains_text":
        # Text is embedded in system prompt; human turn is a neutral trigger.
        sys_content = sys_template.format(
            instruction=instruction.strip(), text=text
        )
        human_text = "(speak)"
    else:
        sys_content = sys_template.format(instruction=instruction.strip())
        human_text = f"[TEXT TO VOCALIZE]: {text}"

    return [
        {"role": "system", "content": sys_content},
        {"role": "human", "content": [{"type": "text", "text": human_text}]},
        {"role": "assistant", "content": "\n<think>\n", "eot": False},
    ]


def build_rl_prompt_pass2(messages_pass1: list, think_text: str) -> list:
    """Patch the assistant turn to close </think> and open <tts_start>.

    `messages_pass1` is the list returned by `build_rl_prompt_chat`.
    `think_text` is the text generated by pass-1 (EXCLUDING </think> — the
    caller strips it; if it's already included, the function is still safe
    because the model's generate() output when using stop_strings may or may
    not include the stop token depending on the transformers version).

    Returns a new messages list ready for pass-2 (audio generation).
    """
    import copy
    msgs = copy.deepcopy(messages_pass1)
    # Strip any trailing </think> from think_text so we control the boundary.
    cleaned = think_text.rstrip()
    if cleaned.endswith("</think>"):
        cleaned = cleaned[: -len("</think>")].rstrip()
    # Re-open with clean close + <tts_start>
    msgs[-1]["content"] = f"\n<think>\n{cleaned}\n</think>\n<tts_start>"
    # eot=False keeps the turn open for audio token generation.
    msgs[-1]["eot"] = False
    return msgs
