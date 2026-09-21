"""Speaker-RAG retrieval: explicit-gender constraint, then CLSP cosine.

Two-stage:
  1. Coarse gender filter — `infer_gender(instruction)` (structured `gender:`
     field > natural-language words/pronouns). Keep only references of that
     gender. Fixes the case where CLSP alone mis-picks gender because pitch/
     timbre wording ("high-pitched", "bright") fights an explicit "Male" label.
  2. CLSP rerank — among the survivors, pick the highest cosine similarity
     between the instruction's text embedding and each reference's audio
     embedding. With one survivor this is a no-op; with no gender cue the whole
     pool is reranked.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .attributes import infer_gender
from .embed import encode_text
from .references import Reference


@dataclass
class SelectionResult:
    reference: Reference                  # chosen reference
    score: float                          # cosine(text_feat, reference audio_feat)
    candidates: list[tuple[str, float]]   # [(name, score), ...] sorted desc, ALL refs
    desired_gender: str | None            # gender inferred from instruction (None if unknown)
    method: str                           # "gender" | "gender+clsp" | "clsp"


def select_reference(
    instruction: str,
    clsp_model,
    device,
    references: list[Reference],
    ref_embeddings: torch.Tensor,
    *,
    gender_filter: bool = True,
) -> SelectionResult:
    """Pick the reference best matching `instruction` (gender filter then CLSP).

    Args:
        instruction: the style instruction.
        clsp_model: loaded CLSP model (clsp_utils.load_clsp).
        device: torch device the CLSP model lives on.
        references: list of Reference (order must match `ref_embeddings`).
        ref_embeddings: (N, 512) L2-normalized audio embeddings from
            `embed.embed_references(references, ...)`.
        gender_filter: when True (default), constrain to the instruction's
            inferred gender before CLSP. Set False to rerank the whole pool.
    """
    if ref_embeddings.shape[0] != len(references):
        raise ValueError("ref_embeddings rows != number of references")

    # CLSP cosine for ALL references (cheap; pool is small). Used for the rerank
    # and for transparent candidate logging even when gender decides the pick.
    text_feat = encode_text(clsp_model, device, instruction)        # (512,)
    sims = (ref_embeddings @ text_feat).tolist()                    # (N,)
    ranked = sorted(
        ((ref.name, s) for ref, s in zip(references, sims)),
        key=lambda x: x[1],
        reverse=True,
    )

    # Stage 1: coarse gender filter. A reference with unknown gender (None) is
    # never excluded; if the desired gender isn't in the pool, fall back to all.
    desired = infer_gender(instruction) if gender_filter else None
    if desired is None:
        allowed, method = list(range(len(references))), "clsp"
    else:
        allowed = [
            i for i, r in enumerate(references)
            if r.gender is None or r.gender == desired
        ]
        if not allowed:                       # desired gender absent from pool
            allowed, method = list(range(len(references))), "clsp"
        elif len(allowed) == 1:
            method = "gender"                 # gender alone forced the pick
        else:
            method = "gender+clsp"            # gender narrowed, CLSP broke the tie

    # Stage 2: CLSP rerank within the survivors.
    best_idx = max(allowed, key=lambda i: sims[i])

    return SelectionResult(
        reference=references[best_idx],
        score=sims[best_idx],
        candidates=ranked,
        desired_gender=desired,
        method=method,
    )
