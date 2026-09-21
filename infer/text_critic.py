"""Text critic used to prepare two-hop training examples.

The critic listens to a first-pass waveform and describes only audible
departures from the requested speaking style. ``data_preprocess`` uses the
offline audio-input helper, while the vLLM training and ablation paths reuse
the same prompt with audio tokens embedded in the chat history.
"""

from __future__ import annotations


CRITIC_PROMPT_EN = (
    "You are reviewing a TTS audio sample for compliance with a styling instruction.\n\n"
    "Styling instruction (the only source of style requirements):\n"
    "{instruction}\n\n"
    "Spoken text (content only, not a source of style requirements):\n"
    "{text}\n\n"
    "Listen carefully to the audio. Write at most three bullets, each about one "
    "dimension where the audio clearly differs from the styling instruction. "
    "Omit dimensions that already match; zero bullets is acceptable.\n\n"
    "Use exactly these three lines for each bullet:\n"
    "(i) what the instruction asked for: <quote or briefly paraphrase it>\n"
    "(ii) what the audio actually does: <a concrete audible observation>\n"
    "(iii) one concrete adjustment: <one specific change for the next reading>\n\n"
    "Line (ii) must describe what is audible rather than merely negating the "
    "instruction. Do not invent requirements from the spoken text, repeat a "
    "dimension, or add a preamble, summary, or closing remark."
)

CRITIC_PROMPT_ZH = (
    "你正在評估一段 TTS 音訊是否符合風格指令。\n\n"
    "風格指令（唯一的風格要求來源）：\n{instruction}\n\n"
    "朗讀文本（僅是內容，不是風格要求來源）：\n{text}\n\n"
    "請仔細聆聽音訊。最多寫三條 bullet，每條只描述一個明顯偏離風格指令的"
    "維度；已符合的維度不要提，全部符合時可不寫 bullet。\n\n"
    "每條 bullet 嚴格使用以下三行：\n"
    "(i) 指令要求什麼：<引用或簡短改寫相關要求>\n"
    "(ii) 音訊實際如何：<具體且可聽見的觀察>\n"
    "(iii) 一個具體調整：<下次朗讀時的一項明確修改>\n\n"
    "第 (ii) 行必須描述實際聽到的特徵，不能只把指令反過來說。不要從朗讀"
    "文本臆造風格要求，不要重複維度，也不要加入前言、總結或結語。"
)


def critic_prompt(lang: str) -> str:
    """Return the critic template for ``lang`` (English is the fallback)."""
    return CRITIC_PROMPT_ZH if lang == "zh" else CRITIC_PROMPT_EN


def build_critic_messages(
    instruction: str,
    text: str,
    audio_path: str,
    lang: str = "en",
) -> list[dict]:
    """Build a Step-Audio-2 chat request containing the first-pass audio."""
    prompt = critic_prompt(lang).format(
        instruction=instruction.strip(),
        text=text.strip(),
    )
    return [
        {"role": "system", "content": prompt},
        {
            "role": "human",
            "content": [{"type": "audio", "audio": audio_path}],
        },
        {"role": "assistant", "content": None},
    ]


def make_critique(
    model,
    instruction: str,
    text: str,
    audio_path: str,
    *,
    lang: str = "en",
    variant: str = "chat",
    max_new_tokens: int = 256,
    temperature: float = 0.3,
) -> str:
    """Generate one critique with an already-loaded Step-Audio-2 model."""
    if variant != "chat":
        raise ValueError("Only the Step-Audio-2 chat critic is supported")
    _, critique, _ = model(
        build_critic_messages(instruction, text, audio_path, lang),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )
    return (critique or "").strip()
