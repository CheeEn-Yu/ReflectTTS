# Critique ablations (inference-only)

Ablations that isolate **what the self-critique step contributes**, all
**inference-only — nothing is retrained**. They differ only in Stage 2 of the
3-stage two-hop pipeline (`v1 → critique → v2`); Stages 1 and 3 are identical
across modes, so the critique is the single independent variable.

The paper's `none` / `fixed` / `self` conditions share the same turn-2
self-critique scaffold (assistant prefilled with `\n<think>\n`), so only the
critique content changes within that comparison. The paper adapter itself was
trained with a separately generated structured critique injected into the v2
prompt; the extra `trained` mode below exposes that training-matched path. Both
v1 and v2 are CLSP+WER scored on InstructTTSEval.

## Two arms

The same modes run on two policies, which separates "critique ability the RL
gave us" from "ability the base model already had":

| arm | policy | script | submit |
|-----|--------|--------|--------|
| **RL adapter** | the trained adapter, applied at *all three* stages | `infer_ablation_pipeline_vllm.py` | `run_ablation_adapter.sbatch` |
| **base (pre-RL)** | plain Step-Audio-2-mini Qwen2-view, **no adapter** | `infer_ablation_vllm.py` | `run_ablation_base.sbatch` |

## Modes

| mode | Stage-2 `<think>` content | question answered | arm |
|------|---------------------------|-------------------|-----|
| `none` | empty (v1 → v2 directly) | does critique help at all? | both |
| `fixed` | rule-based "this is already good" line — deliberately content-free praise, names no concrete acoustic fix | does the critique *content* matter, or just the extra pass? | both |
| `self` | model-generated self-critique (the paper inference pipeline) | does an instance-specific critique help? | both |
| `random` | 40-word English word salad: fluent tokens, zero meaning, neither praise nor criticism | control for the `fixed` result — does the v2 gain need a *benign* critique, or merely a *non-empty* think block? | adapter only |
| `trained` | *(not a `<think>` block)* structured critique via the training critic prompt (`build_critique_prompt_chat` / `CRITIC_PROMPT_EN`), injected through `build_twohop_prompt_chat` | the maximally in-distribution path — exactly what `static`-mode RL trained on | adapter only |

The `random` salad is deterministic per item (`crc32` of `<rid>_<task>` seeds the
RNG), so reruns are comparable.

## Run

```bash
# Submit one mode first; its compute job builds the shared view. After it finishes,
# submit the other modes, which reuse that view without racing on it.
ADAPTER=... VIEW=out/qwen2_view/<run_tag>_merged CRITIQUE_MODE=self \
  sbatch -A <account> ablation/run_ablation_adapter.sbatch
CRITIQUE_MODE=self sbatch -A <account> ablation/run_ablation_base.sbatch

# After both view-building jobs finish, the remaining modes can run in parallel:
for m in none fixed; do
  ADAPTER=... VIEW=... CRITIQUE_MODE=$m sbatch -A <account> ablation/run_ablation_adapter.sbatch
done
for m in none fixed; do
  CRITIQUE_MODE=$m sbatch -A <account> ablation/run_ablation_base.sbatch     # base arm
done
ADAPTER=... LIMIT=20 CRITIQUE_MODE=none \
  sbatch -A <account> ablation/run_ablation_adapter.sbatch  # smoke first
```

The adapter arm supports `random` / `trained` on top of `none` / `fixed` / `self`;
the base arm supports `none` / `fixed` / `self`. Always build the shared Qwen2
view before fanning out, so parallel jobs don't race on the same output dir.

Speaker retrieval is on by default. The adapter arm requires
`ADAPTER=/path/to/adapter`; `out/` is git-ignored.

Results land in `out/ablation/<tag>/{v1,v2}_result/final_result.txt`. `<tag>` is
`ablate_<mode>_<run_tag>` (adapter arm) or
`base_<mode>` (base arm).

## Extra probes

Beyond `--critique_mode`, `infer_ablation_pipeline_vllm.py` carries a few
independent knobs:

- `--fixed_v1_text` / `--fixed_v1_max_codes` — v1 is no longer a per-item first
  attempt: synthesize ONE style-neutral utterance of a fixed sentence (optionally
  truncated to N audio codes, i.e. a deliberately degenerate first draft) and
  reuse it for every item. Isolates how much a *good v1* is worth.
- `--v2_len_cap` / `--v2_max_retries` — inference-time guard against off-text
  drift: reject and resample v2 whose audio runs longer than `cap × v1`, keeping
  the shortest. Disabled by default.
- `--num_shards` / `--shard_id` — round-robin row sharding (`rows[shard_id::num_shards]`)
  for splitting one eval across parallel jobs.

## Files

- `infer_ablation_pipeline_vllm.py` — adapter arm; `--critique_mode {self,none,fixed,random,trained}`.
- `infer_ablation_vllm.py` — base arm; `--critique_mode {self,none,fixed}` (required).
- `run_ablation_adapter.sbatch` — one adapter-arm job for one mode (merge view → sidecar → infer → score).
- `run_ablation_base.sbatch` — same for the base arm (no adapter merge; plain Qwen2-view).
