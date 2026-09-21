# Speaker RAG

Retrieve the best reference speaker for a generated utterance.

## Why

In Step-Audio-2 the LLM emits **audio codec tokens** (content + prosody +
style), but the **voice timbre — including gender — is fixed entirely by the
`prompt_wav`** fed to the flow vocoder (`Token2wav` →
`CausalMaskedDiffWithXvec`). So a "female" style instruction synthesized off a
male reference stays male; the codec can't move gender.

Speaker RAG picks, per utterance, the candidate reference whose voice best
matches the instruction:

1. **Precompute** each reference's audio embedding with the CLSP **audio tower**
   (cached to disk; only new/changed wavs are re-encoded).
2. **Per utterance**, infer an explicit speaker gender from the instruction when
   available and filter the candidate pool accordingly.
3. Embed the instruction with the CLSP **text tower** and pick the surviving
   reference with the highest cosine similarity.
4. Vocode the generated codec tokens **once** with the chosen reference.

CLSP (`yfyeung/CLSP`) is a CLAP-style dual encoder (independent audio/text
towers, 512-d L2-normalized joint space), so audio and text can be embedded
separately and compared by dot product. Retrieval needs **no vocoder pass**.

> Tradeoff: this scores the *reference wav* against the instruction, not the
> generated content. That's exactly the "which voice fits the requested
> style/gender" signal we want, and it's cheap.

The package is used by the inference, data-preparation, and training entry
points. It can also be imported directly as a library.

## References

The candidates are a small built-in list in `references.py` (`DEFAULT_REFERENCES`):
the two clean single-speaker WAV files from `Step-Audio2/assets/`
(`default_male.wav`, `default_female.wav`). Relative `wav` paths resolve under
`Step-Audio2/`; use an absolute path for references elsewhere. With one ref per
gender the choice is essentially binary. Add entries to `DEFAULT_REFERENCES` to
make retrieval finer. Cached embeddings are keyed by each wav's path + mtime.
The one-hop inference entry point defaults to
`speaker_rag/references.emb.pt`; other integrations cache embeddings when a
`--speaker_emb_cache` path is supplied.

## Usage

Speaker RAG is integrated into the inference entry points; there is no separate
`speaker_rag.cli` module. Enable it with `--speaker_rag`. For example, with a
Step-Audio-2 vLLM server already running:

```bash
python infer/instructtseval_vllm.py \
    --split en \
    --output_jsonl out/eval/results.jsonl \
    --audio_dir out/eval/audio \
    --speaker_rag
```

The supplied Slurm launchers expose the same option through the
`SPEAKER_RAG=1` environment variable:

```bash
OUT_DIR=out/eval/zero_shot SPEAKER_RAG=1 \
    sbatch -A <account> scripts/infer/run_onehop_infer_zeroshot.sbatch
```

The one-hop entry point accepts `--clsp_model_id` (default `yfyeung/CLSP`) and
`--speaker_emb_cache`. The two-hop inference entry points use `--clsp_model`
and `--speaker_emb_cache` for the equivalent settings.

## Library

```python
from speaker_rag import DEFAULT_REFERENCES, embed_references, select_reference

ref_emb = embed_references(DEFAULT_REFERENCES, clsp_model, device,
                           cache_path="speaker_rag/references.emb.pt")
res = select_reference(instruction, clsp_model, device, DEFAULT_REFERENCES, ref_emb)
res.reference.name, res.score, res.method, res.candidates
# Then call token2wav(codes, prompt_wav=res.reference.wav).
```
