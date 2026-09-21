"""Infer the desired speaker gender from a style instruction.

Three-tier priority (most reliable first):
  1. Structured field `gender: male/female` (InstructTTSEval APS instructions
     list this explicitly — near-100% reliable).
  2. Natural-language gender words, including pronouns (he/his/she/her).
  3. None -> caller falls back to CLSP.

This is the coarse gender constraint that runs *before* CLSP retrieval. On
InstructTTSEval, explicit gender (esp. the APS `gender:` field) is more reliable
than letting CLSP infer gender from pitch/timbre wording — which mis-picks when
"high-pitched"/"bright" descriptors fight an explicit "Male" label.
"""
from __future__ import annotations

import re

# Tier 1: structured "gender: male" / "gender: female" field (APS).
_GENDER_FIELD_RE = re.compile(r"gender\s*[:=]\s*(male|female)", re.IGNORECASE)

# Tier 2: natural-language words. Pronouns ARE included here (he/his/she/her):
# they're noisier than nouns but a useful signal when no explicit noun appears.
_FEMALE_EN = {
    "female", "woman", "women", "girl", "girls", "lady", "ladies", "feminine",
    "she", "her", "hers", "herself",
    "mother", "mom", "mum", "grandmother", "grandma", "sister", "aunt",
    "queen", "princess", "actress", "waitress", "soprano",
}
_MALE_EN = {
    "male", "man", "men", "boy", "boys", "gentleman", "gentlemen", "masculine",
    "he", "him", "his", "himself",
    "father", "dad", "grandfather", "grandpa", "brother", "uncle",
    "king", "prince", "actor", "waiter", "tenor", "baritone",
}
_FEMALE_ZH = ["女", "她", "妈", "母", "姐", "妹", "阿姨", "奶奶", "婆", "姑娘", "女士", "小姐"]
_MALE_ZH = ["男", "他", "爸", "父", "哥", "弟", "叔", "爷爷", "先生", "小伙", "汉子"]


def _count_en(text_lower: str, vocab: set[str]) -> int:
    # Word-boundary match so "man" doesn't fire inside "woman"/"human", and
    # "he" doesn't fire inside "the".
    return sum(1 for w in vocab if re.search(rf"\b{re.escape(w)}\b", text_lower))


def _count_zh(text: str, markers: list[str]) -> int:
    return sum(text.count(m) for m in markers)


def infer_gender(instruction: str) -> str | None:
    """Return 'male', 'female', or None (unknown / ambiguous) for an instruction."""
    if not instruction:
        return None

    # Tier 1: explicit structured field wins outright.
    m = _GENDER_FIELD_RE.search(instruction)
    if m:
        return m.group(1).lower()

    # Tier 2: count gender words (incl. pronouns); majority wins, tie -> None.
    low = instruction.lower()
    female = _count_en(low, _FEMALE_EN) + _count_zh(instruction, _FEMALE_ZH)
    male = _count_en(low, _MALE_EN) + _count_zh(instruction, _MALE_ZH)
    if female > male:
        return "female"
    if male > female:
        return "male"
    return None
