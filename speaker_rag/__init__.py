"""Speaker RAG: retrieve the best reference speaker for a generated utterance.

The Step-Audio-2 LLM emits audio codec tokens that carry content/prosody/style,
but the *voice timbre* (incl. gender) is fixed entirely by the `prompt_wav` fed
to the flow vocoder (`Token2wav` -> `CausalMaskedDiffWithXvec`). So a "female"
style instruction synthesized off a male reference stays male — the codec can't
move gender.

This package keeps a small list of candidate references, filters them by an
explicit gender cue when available, then uses CLSP text/audio cosine similarity
to select among the survivors. Reference audio embeddings are precomputed and
cached. Retrieval needs no vocoder pass; the chosen reference is synthesized
once.

Use it as a library (`select_reference`); the repository's inference, data
preparation, and training entry points integrate it through `--speaker_rag`.
"""
from .references import Reference, DEFAULT_REFERENCES
from .attributes import infer_gender
from .embed import embed_references, encode_audio, encode_text
from .select import SelectionResult, select_reference

__all__ = [
    "Reference",
    "DEFAULT_REFERENCES",
    "infer_gender",
    "embed_references",
    "encode_audio",
    "encode_text",
    "SelectionResult",
    "select_reference",
]
