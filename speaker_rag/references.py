"""Candidate reference speakers to choose between.

The defaults are Step-Audio-2's two reference WAV files (one male and one
female). CLSP picks whichever matches the instruction better. Edit
``DEFAULT_REFERENCES`` or pass a custom list to add more.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Reuse the assets supplied by the required Step-Audio-2 checkout.
_STEP_DIR = Path(__file__).resolve().parent.parent / "Step-Audio2"


@dataclass(frozen=True)
class Reference:
    name: str
    wav: str            # resolved absolute path to the reference wav
    gender: str | None = None  # optional metadata


def _ref(name: str, wav: str, gender: str | None = None) -> Reference:
    # Relative paths resolve under Step-Audio2/; absolute paths pass through.
    path = wav if os.path.isabs(wav) else str(_STEP_DIR / wav)
    return Reference(name=name, wav=path, gender=gender)


DEFAULT_REFERENCES: list[Reference] = [
    _ref("default_female", "assets/default_female.wav", "female"),
    _ref("default_male", "assets/default_male.wav", "male"),
]
