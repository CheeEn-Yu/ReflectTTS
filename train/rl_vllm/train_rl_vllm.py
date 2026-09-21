"""vLLM-rollout GRPO smoke trainer (step env). Two-process design:

  * This process (step env) owns the policy (StepAudio2+LoRA), the reward scorer,
    the GRPO loss + backward — the proven hand-rolled `train.rl.trainer.GRPOTrainer`,
    UNCHANGED except that rollout is swapped to vLLM.
  * A sidecar (vllm env) runs the stock-vLLM Qwen2-view engine and serves rollouts
    over a file-RPC (train.rl_vllm.vllm_rollout_server).

Per step: merge LoRA -> dump merged weights to /dev/shm -> tell the sidecar to
hot-load them (on-policy) -> sidecar generates G rollouts per prompt -> we score +
GRPO-update on the HF policy exactly as before. Phase timing is collected and, at
the end, the smoke summary + profile is pushed to Discord.

Run via scripts/train/run_rl_twohop.sbatch (which also launches the sidecar).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

# side effects: torchaudio bypass + sys.path for stepaudio2; reuse builders.
from train.rl.train_rl import _build_policy, _build_scorer  # noqa: E402
from train.rl.trainer import GRPOTrainer, GRPOConfig
from train.rl.prompt import (build_twohop_prompt_chat, build_twohop_selfcritique_chat,
                             build_singlepass_prompt_chat, build_critique_prompt_chat)
from train.rl.rollout import (Rollout, _build_prompt_ids, _trim_at_eos,
                              AUDIO_TOKEN_OFFSET, AUDIO_TOKEN_VOCAB_SIZE, TEXT_TOKEN_MAX)


# --------------------------------------------------------------------------- #
# phase timing
# --------------------------------------------------------------------------- #
class Timer:
    def __init__(self):
        self.sum = defaultdict(float)
        self.cnt = defaultdict(int)
        self.steps = 0

    @contextlib.contextmanager
    def t(self, name, cuda=False):
        if cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.sum[name] += time.perf_counter() - t0
            self.cnt[name] += 1

    def add(self, name, dt):
        self.sum[name] += dt
        self.cnt[name] += 1

    def report(self) -> str:
        n = max(1, self.steps)
        lines = [f"per-step means over {self.steps} steps:"]
        for k in sorted(self.sum, key=lambda k: -self.sum[k]):
            lines.append(f"  {k:18s} {self.sum[k]/n:7.2f}s  ({self.sum[k]:7.1f}s total)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# file-RPC client to the vLLM sidecar
# --------------------------------------------------------------------------- #
class SidecarClient:
    def __init__(self, workdir: str, timeout: float = 1200.0):
        self.wd = Path(workdir)
        self.req = self.wd / "request.json"
        self.resp = self.wd / "response.json"
        self.timeout = timeout
        self._id = 0

    def wait_ready(self):
        ready = self.wd / "ready"
        t0 = time.time()
        while not ready.exists():
            if time.time() - t0 > self.timeout:
                raise TimeoutError("sidecar never became ready")
            time.sleep(0.2)

    def send(self, obj: dict) -> int:
        """Write a request and return its id WITHOUT waiting (lets the trainer do
        concurrent work, e.g. NCCL broadcast, before reading the response)."""
        self._id += 1
        obj["id"] = self._id
        tmp = self.req.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj))
        os.replace(tmp, self.req)
        return self._id

    def wait(self, rid: int) -> dict:
        t0 = time.time()
        while True:
            if self.resp.exists():
                try:
                    r = json.loads(self.resp.read_text())
                except Exception:
                    time.sleep(0.02); continue
                if r.get("id") == rid:
                    os.remove(self.resp)
                    if not r.get("ok"):
                        raise RuntimeError(f"sidecar error: {r}")
                    return r
            if time.time() - t0 > self.timeout:
                raise TimeoutError(f"sidecar call id={rid} timed out")
            time.sleep(0.02)

    def _call(self, obj: dict) -> dict:
        return self.wait(self.send(obj))

    def sync(self, merged_path: str) -> dict:
        return self._call({"op": "sync", "merged_path": merged_path})

    def compare(self, merged_path: str) -> dict:
        return self._call({"op": "compare", "merged_path": merged_path})["diff"]

    def rollout(self, prompts, sampling) -> list:
        return self._call({"op": "rollout", "prompts": prompts, "sampling": sampling})["completions"]

    def stop(self):
        with contextlib.suppress(Exception):
            self._call({"op": "stop"})


# --------------------------------------------------------------------------- #
# merged-weight extraction (clean Qwen2-view names, no encoder/adapter)
# --------------------------------------------------------------------------- #
def merged_items(peft_llm, to_cpu: bool):
    """After merge_adapter(), the lora.Linear base_layer.weight holds the merged
    values. Yield (clean_qwen2_name, tensor) for {model.* , lm_head.weight},
    dropping peft wrappers + the audio encoder/adapter. to_cpu=True for the disk
    path (bf16 on CPU); to_cpu=False keeps tensors on GPU for NCCL broadcast.
    Deterministic order (named_parameters) so trainer-send and worker-recv pair up."""
    base = peft_llm.base_model.model  # StepAudio2ForCausalLM
    for name, p in base.named_parameters():
        if ".lora_" in name:
            continue
        clean = name.replace(".base_layer.", ".")
        if clean.startswith("model.") or clean.startswith("lm_head."):
            t = p.detach()
            if to_cpu:
                t = t.to("cpu", torch.bfloat16)
            else:
                t = t.to(torch.bfloat16)
            yield clean, t.contiguous()


def merged_state_dict(peft_llm) -> dict:
    return {k: v for k, v in merged_items(peft_llm, to_cpu=True)}


# --------------------------------------------------------------------------- #
# vLLM rollout_fn (drop-in for sample_rollouts_twohop) + timed scorer
# --------------------------------------------------------------------------- #
def make_rollout_fn(client: SidecarClient, timer: Timer, seed_base=None):
    counter = [0]

    def rollout_fn(policy, messages, *, G, max_new_tokens, temperature, top_p,
                   repetition_penalty, eos_token_id=None, do_sample=True, **_):
        eos = eos_token_id if eos_token_id is not None else policy.eos_token_id
        prompt_ids = _build_prompt_ids(policy, messages).squeeze(0).cpu()
        # deterministic per-call seed so disk vs nccl runs are directly comparable
        seed = None if seed_base is None else seed_base + counter[0]
        counter[0] += 1
        sampling = dict(n=G, temperature=(temperature if do_sample else 0.0),
                        top_p=top_p, max_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty, eos=int(eos), seed=seed)
        with timer.t("rollout"):
            comps = client.rollout([prompt_ids.tolist()], sampling)
        rollouts = []
        for g in comps[0]:
            gen = torch.tensor(g, dtype=torch.long)
            gen, truncated = _trim_at_eos(gen, eos)
            codes = [t - AUDIO_TOKEN_OFFSET for t in gen.tolist()
                     if t >= AUDIO_TOKEN_OFFSET and (t - AUDIO_TOKEN_OFFSET) < AUDIO_TOKEN_VOCAB_SIZE]
            rollouts.append(Rollout(prompt_ids=prompt_ids, gen_ids=gen, think_text="",
                                    audio_codes=codes, truncated=truncated))
        return rollouts
    return rollout_fn


def make_selfcritique_rollout_fn(client: SidecarClient, timer: Timer, seed_base=None,
                                 max_think_tokens: int = 256):
    """Two-pass SELF-critique rollout over the vLLM sidecar (Fix 1 on vLLM).

    Mirrors train.rl.rollout.sample_rollouts_twohop_selfcritique but runs both
    passes on vLLM. The policy generates its OWN critique, so the critique tokens
    land in gen_ids and the GRPO gradient shapes them — unlike make_rollout_fn,
    which conditions on a frozen critique baked into the prompt (zero gradient).

      Pass 1: n=G critiques from the shared pass-1 prompt (stop at </think>).
      Pass 2: build each pass-2 prompt as p1_ids + critique_ids + bridge_ids and
              generate v2 audio (n=1 each). Building the prompt from token ids
              (not re-tokenized text) makes prompt_ids + gen_ids EXACTLY the
              sequence vLLM scored — no boundary re-tokenization drift.

    gen_ids = critique_ids ++ bridge_ids ++ audio_ids; prompt_ids = p1_ids.
    """
    counter = [0]
    bridge_cache: dict = {}

    def rollout_fn(policy, messages, *, G, max_new_tokens, temperature, top_p,
                   repetition_penalty, eos_token_id=None, do_sample=True, **_):
        eos = eos_token_id if eos_token_id is not None else policy.eos_token_id
        if "ids" not in bridge_cache:
            bridge_cache["ids"] = policy.llm_tokenizer(
                "\n</think>\n<tts_start>", return_tensors="pt", add_special_tokens=False
            )["input_ids"].squeeze(0)
        bridge_ids = bridge_cache["ids"]
        temp = temperature if do_sample else 0.0
        p1_ids = _build_prompt_ids(policy, messages).squeeze(0).cpu()  # [T1]

        # ---- Pass 1: critique (G samples from shared prompt, stop at </think>) ----
        seed = None if seed_base is None else seed_base + counter[0]
        counter[0] += 1
        s1 = dict(n=G, temperature=temp, top_p=top_p, max_tokens=max_think_tokens,
                  repetition_penalty=repetition_penalty, eos=int(eos), seed=seed,
                  stop=["</think>"])
        with timer.t("rollout.critique"):
            crit_comps = client.rollout([p1_ids.tolist()], s1)[0]  # list of G token-id lists

        # ---- Build pass-2 prompts directly from token ids: p1 + critique + bridge ----
        crit_ids_list: list[torch.Tensor] = []
        p2_prompts: list[list[int]] = []
        for g in crit_comps:
            c = torch.tensor(g, dtype=torch.long)
            if c.numel() and c[-1] == eos:   # drop a trailing eos if the model emitted one
                c = c[:-1]
            crit_ids_list.append(c)
            p2_prompts.append(torch.cat([p1_ids, c, bridge_ids]).tolist())

        # ---- Pass 2: audio (one completion per pass-2 prompt) ----
        seed2 = None if seed_base is None else seed_base + counter[0]
        counter[0] += 1
        s2 = dict(n=1, temperature=temp, top_p=top_p, max_tokens=max_new_tokens,
                  repetition_penalty=repetition_penalty, eos=int(eos), seed=seed2)
        with timer.t("rollout.audio"):
            audio_comps = client.rollout(p2_prompts, s2)  # list of G, each [[ids]]

        rollouts = []
        for g in range(G):
            c = crit_ids_list[g]
            a = torch.tensor(audio_comps[g][0], dtype=torch.long)
            a, truncated = _trim_at_eos(a, eos)
            gen_ids = torch.cat([c, bridge_ids, a])
            codes = [t - AUDIO_TOKEN_OFFSET for t in a.tolist()
                     if t >= AUDIO_TOKEN_OFFSET and (t - AUDIO_TOKEN_OFFSET) < AUDIO_TOKEN_VOCAB_SIZE]
            think_text = policy.llm_tokenizer.decode(
                [t for t in c.tolist() if t < TEXT_TOKEN_MAX], skip_special_tokens=False
            ).strip()
            rollouts.append(Rollout(prompt_ids=p1_ids, gen_ids=gen_ids, think_text=think_text,
                                    audio_codes=codes, truncated=truncated))
        return rollouts
    return rollout_fn


class TimedScorer:
    def __init__(self, scorer, timer):
        self._s, self._t = scorer, timer

    def __call__(self, *a, **k):
        with self._t.t("scoring", cuda=True):
            return self._s(*a, **k)

    def __getattr__(self, n):
        return getattr(self._s, n)


# --------------------------------------------------------------------------- #
# trainer subclass: sync weights to the sidecar each step, time phases
# --------------------------------------------------------------------------- #
class VLLMGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, client: SidecarClient, timer: Timer, merged_path: str,
                 sync_mode: str = "disk", nccl_host: str = "127.0.0.1",
                 nccl_port: int = 29555, verify_first: bool = True,
                 on_policy_critique: bool = False, critic_temperature: float = 0.3,
                 critic_max_new_tokens: int = 256, critique_dir: str = "/dev/shm/rlvllm_crit",
                 on_policy_v1: bool = False, v1_temperature: float = 0.9,
                 v1_top_p: float = 0.9, v1_max_new_tokens: int = 512,
                 **kw):
        super().__init__(*args, **kw)
        self.client = client
        self.timer = timer
        self.merged_path = merged_path
        self.sync_mode = sync_mode
        self.nccl_host, self.nccl_port = nccl_host, nccl_port
        self.verify_first = verify_first
        self.on_policy_critique = on_policy_critique
        self.critic_temperature = critic_temperature
        self.critic_max_new_tokens = critic_max_new_tokens
        # on-policy v1: generate the first-draft audio each step with the current
        # policy (single-pass) instead of reading precomputed v1_tokens. v1 is
        # NEVER trained — it only seeds the self-critique prompt context and the
        # reward's CLSP baseline; the trained completion is critique+v2 (gen_ids).
        self.on_policy_v1 = on_policy_v1
        self.v1_temperature = v1_temperature
        self.v1_top_p = v1_top_p
        self.v1_max_new_tokens = v1_max_new_tokens
        self.critique_dir = Path(critique_dir)
        self.critique_dir.mkdir(parents=True, exist_ok=True)
        self.nccl = None
        self._verified = None  # weight-equality diff from the first nccl sync
        if sync_mode == "nccl":
            self._init_nccl()

    def _gen_v1(self, batch):
        """On-policy v1 via the vLLM sidecar: single-pass instruction+text -> audio.

        Overwrites batch['v1_tokens'] with the freshly generated first draft and
        clears any precomputed clsp_v1/wer_v1 so the reward recomputes the
        per-prompt baseline from THIS v1. Runs the whole flow on-policy
        (v1->critique->v2) while keeping v1 OUT of the gradient: the v1 tokens
        enter only as prompt context for build_twohop_selfcritique_chat and as the
        reward's CLSP baseline; the trained completion (critique+v2) lives in each
        rollout's gen_ids, so policy_logprobs never sees the v1 tokens.

        Mirrors _gen_critiques: best-effort, batched on the same sidecar engine.
        On a per-prompt build/gen failure the existing batch v1_tokens (data, if
        any) is kept as fallback.
        """
        n = len(batch["uids"])
        v1_list = list(batch.get("v1_tokens") or [None] * n)   # fallback = data v1
        eos = int(self.policy.eos_token_id)
        prompt_ids_list, idx_map = [], []
        for i in range(n):
            try:
                msgs = build_singlepass_prompt_chat(
                    batch["instructions"][i], batch["texts"][i], batch["langs"][i])
                ids = _build_prompt_ids(self.policy, msgs).squeeze(0).cpu().tolist()
                prompt_ids_list.append(ids)
                idx_map.append(i)
            except Exception as e:
                print(f"[trainer] v1 prompt build failed (keep fallback): {e}", flush=True)
        if prompt_ids_list:
            sampling = dict(n=1, temperature=self.v1_temperature, top_p=self.v1_top_p,
                            max_tokens=self.v1_max_new_tokens, repetition_penalty=1.0,
                            eos=eos, seed=None)
            comps = self.client.rollout(prompt_ids_list, sampling)
            for j, i in enumerate(idx_map):
                gen = torch.tensor(comps[j][0] if comps[j] else [], dtype=torch.long)
                gen, _ = _trim_at_eos(gen, eos)
                # raw audio LM token ids (>=offset, in-codebook) — the format
                # build_twohop_selfcritique_chat splices and reward._v1_clsp_baseline reads.
                raw_audio_ids = [t for t in gen.tolist()
                                 if t >= AUDIO_TOKEN_OFFSET
                                 and (t - AUDIO_TOKEN_OFFSET) < AUDIO_TOKEN_VOCAB_SIZE]
                if len(raw_audio_ids) >= self.cfg_min_audio_codes:
                    v1_list[i] = raw_audio_ids
        batch["v1_tokens"] = v1_list
        # the baseline must match the on-policy v1, not a stale precomputed value.
        batch["clsp_v1"] = [None] * n
        batch["wer_v1"] = [None] * n

    @property
    def cfg_min_audio_codes(self) -> int:
        # reward's min-audio-codes threshold (scorer may be TimedScorer-wrapped).
        return getattr(getattr(self.scorer, "cfg", None), "min_audio_codes", 5)

    def _gen_critiques(self, batch):
        """On-policy critique via vLLM sidecar: build critique prompts with v1
        audio tokens inline, batch-generate text critiques on the same vLLM engine
        that does rollouts. Paper mode requires every critique to come from the
        synchronized current policy; generation failures therefore stop the run
        instead of silently falling back to a stored static critique."""
        n = len(batch["uids"])
        crits = [None] * n
        prompt_ids_list = []
        idx_map = []  # maps position in prompt_ids_list -> batch index
        for i in range(n):
            v1 = (batch.get("v1_tokens") or [None] * n)[i]
            if not v1:
                continue
            try:
                msgs = build_critique_prompt_chat(
                    batch["instructions"][i], batch["texts"][i],
                    batch["langs"][i], v1_tokens=v1,
                )
                ids = _build_prompt_ids(self.policy, msgs).squeeze(0).cpu().tolist()
                prompt_ids_list.append(ids)
                idx_map.append(i)
            except Exception as e:
                raise RuntimeError(
                    f"on-policy critique prompt failed for {batch['uids'][i]}"
                ) from e
        if prompt_ids_list:
            sampling = dict(n=1, temperature=self.critic_temperature,
                            top_p=0.9, max_tokens=self.critic_max_new_tokens,
                            repetition_penalty=1.0,
                            eos=int(self.policy.eos_token_id), seed=None)
            comps = self.client.rollout(prompt_ids_list, sampling)
            for j, i in enumerate(idx_map):
                gen_ids = comps[j][0] if comps[j] else []
                if any(AUDIO_TOKEN_OFFSET <= t
                       < AUDIO_TOKEN_OFFSET + AUDIO_TOKEN_VOCAB_SIZE
                       for t in gen_ids):
                    raise RuntimeError(
                        f"critic emitted audio tokens for {batch['uids'][i]}"
                    )
                text_ids = [t for t in gen_ids if t < TEXT_TOKEN_MAX]
                critique_text = self.policy.llm_tokenizer.decode(
                    text_ids, skip_special_tokens=True).strip()
                # Empty is a valid current-policy result: the critic prompt
                # explicitly permits zero bullets when v1 already matches.
                crits[i] = critique_text
        if any(c is None for c in crits):
            missing = [batch["uids"][i] for i, c in enumerate(crits) if c is None]
            raise RuntimeError(f"missing on-policy critiques for: {missing}")
        batch["critiques"] = crits

    def _init_nccl(self):
        from train.rl_vllm.nccl_transport import NcclGroup
        dev = next(self.policy.llm.parameters()).device
        print(f"[trainer] NCCL handshake rank0 {dev} <-> worker rank1 "
              f"({self.nccl_host}:{self.nccl_port})", flush=True)
        rid = self.client.send({"op": "init_nccl", "host": self.nccl_host,
                                "port": self.nccl_port, "rank": 1, "world_size": 2})
        self.nccl = NcclGroup(self.nccl_host, self.nccl_port, 0, 2, dev)  # blocks until worker joins
        self.client.wait(rid)
        print("[trainer] NCCL group up", flush=True)

    def _sync_weights(self):
        if self.sync_mode == "nccl":
            self._sync_weights_nccl()
        else:
            self._sync_weights_disk()

    def _sync_weights_disk(self):
        with self.timer.t("sync_merge_save", cuda=True):
            self.policy.llm.merge_adapter()
            try:
                save_file(merged_state_dict(self.policy.llm), self.merged_path)
            finally:
                self.policy.llm.unmerge_adapter()
        r = self.client.sync(self.merged_path)
        self.timer.add("sync_rpc_load", r.get("t", 0.0))

    def _sync_weights_nccl(self, chunk_bytes: int = 1 << 30):
        with self.timer.t("sync_nccl", cuda=True):
            with self.timer.t("sync.merge", cuda=True):
                self.policy.llm.merge_adapter()
            try:
                with self.timer.t("sync.extract", cuda=True):
                    items = list(merged_items(self.policy.llm, to_cpu=False))  # GPU tensors
                    chunks, cur, cur_b = [], [], 0
                    for name, t in items:
                        b = t.numel() * 2
                        if cur and cur_b + b > chunk_bytes:
                            chunks.append(cur); cur, cur_b = [], 0
                        cur.append((name, t)); cur_b += b
                    if cur:
                        chunks.append(cur)
                    specs = [[[name, list(t.shape)] for name, t in c] for c in chunks]
                rid = self.client.send({"op": "sync_nccl", "specs": specs})
                with self.timer.t("sync.broadcast", cuda=True):
                    for c in chunks:
                        flat = torch.cat([t.reshape(-1) for _, t in c])
                        self.nccl.broadcast(flat, src=0)
                        del flat
                    r = self.client.wait(rid)
            finally:
                self.policy.llm.unmerge_adapter()
        self.timer.add("sync_rpc_load", r.get("t", 0.0))
        # one-time weight-equality verification (NOT in the steady-state sync timer)
        if self.verify_first and self._verified is None:
            with self.timer.t("verify_once", cuda=True):
                self.policy.llm.merge_adapter()
                try:
                    save_file(merged_state_dict(self.policy.llm), self.merged_path)
                finally:
                    self.policy.llm.unmerge_adapter()
                self._verified = self.client.compare(self.merged_path)
            print(f"[trainer] NCCL-sync weight check vs merged file: {self._verified}", flush=True)

    def step(self, batch):
        with self.timer.t("step_total", cuda=True):
            self._sync_weights()
            # on-policy v1 first (its tokens seed the critique prompt + baseline);
            # must run AFTER the weight sync so v1 reflects the current policy.
            if self.on_policy_v1:
                with self.timer.t("v1_gen", cuda=True):
                    self._gen_v1(batch)
            if self.on_policy_critique:
                with self.timer.t("critique", cuda=True):
                    self._gen_critiques(batch)
            agg = super().step(batch)
        return agg


# --------------------------------------------------------------------------- #
def load_rows(jsonl: str, limit: int | None):
    rows = [json.loads(l) for l in open(jsonl) if l.strip()]
    return rows[:limit] if limit else rows


def validate_rows(rows: list[dict], args) -> None:
    """Fail before model loading when the JSONL cannot support this RL mode."""
    if not rows:
        raise SystemExit(f"training JSONL is empty: {args.train_jsonl}")

    problems = []
    for i, row in enumerate(rows, 1):
        uid = row.get("uid", f"row-{i}")
        missing = [k for k in ("uid", "instruction", "text") if not row.get(k)]
        if args.critique_mode != "none":
            if not row.get("v1_tokens") and not args.on_policy_v1:
                missing.append("v1_tokens")
            if (args.critique_mode == "static"
                    and not args.on_policy_critique
                    and row.get("critique") is None):
                missing.append("critique")
            if not args.on_policy_v1:
                for k in ("clsp_v1", "wer_v1"):
                    if row.get(k) is None:
                        missing.append(k)
                if args.speaker_rag and not row.get("clsp_v1_reference"):
                    missing.append("clsp_v1_reference")
        if missing:
            problems.append(f"{uid}: {', '.join(missing)}")
        if len(problems) == 5:
            break

    if problems:
        details = "\n  ".join(problems)
        raise SystemExit(
            "training JSONL is incompatible with the selected two-hop mode; "
            f"first problems:\n  {details}"
        )


def batches(rows, micro_batch):
    for i in range(0, len(rows), micro_batch):
        c = rows[i:i + micro_batch]
        yield {
            "uids": [r["uid"] for r in c],
            "instructions": [r["instruction"] for r in c],
            "texts": [r["text"] for r in c],
            "langs": [r.get("lang", "en") for r in c],
            "v1_tokens": [r.get("v1_tokens") for r in c],
            "critiques": [r.get("critique") for r in c],
            "clsp_v1": [r.get("clsp_v1") for r in c],
            "wer_v1": [r.get("wer_v1") for r in c],
            "clsp_v1_reference": [r.get("clsp_v1_reference") for r in c],
        }


def plot_reward_breakdown(metrics, rows, out_dir: Path) -> str:
    """4-panel reward-detail figure: reward, CLSP (v2 vs v1 baseline), WER (v2 vs
    v1 baseline), KL. v1 baselines are precomputed per-prompt constants -> drawn
    as horizontal reference lines (mean over the training rows)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def col(k):
        return [(m["step"], m[k]) for m in metrics if m.get(k) is not None and m.get("step")]

    def moving_avg(xs, ys, w=50):
        ma = []
        for i in range(len(ys)):
            s = max(0, i - w + 1)
            ma.append(sum(ys[s:i+1]) / (i - s + 1))
        return ma

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    clsp_v1 = mean([r.get("clsp_v1") for r in rows])
    wer_v1 = mean([r.get("wer_v1") for r in rows])
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for (axx, key, title, hline) in [
        (ax[0, 0], "reward_mean", "reward", None),
        (ax[0, 1], "clsp_mean", "CLSP (v2 vs v1)", clsp_v1),
        (ax[0, 2], "clsp_delta_mean", "CLSP delta (v2-v1)", 0.0),
        (ax[1, 0], "wer_mean", "WER (v2 vs v1)", wer_v1),
        (ax[1, 1], "kl", "KL", None),
        (ax[1, 2], "improve_transformed_mean", "improve_fn(delta)", 0.0),
    ]:
        pts = col(key)
        if pts:
            xs, ys = zip(*pts)
            axx.plot(xs, ys, alpha=0.15, color="C0", linewidth=0.5)
            ma = moving_avg(xs, ys, w=50)
            axx.plot(xs, ma, color="C0", linewidth=2, label="MA50")
        if hline is not None:
            axx.axhline(hline, color="r", ls="--", label="baseline")
        axx.legend(fontsize=8); axx.set_title(title); axx.set_xlabel("step"); axx.grid(alpha=0.3)
    fig.suptitle("RL reward breakdown (vLLM rollout, tanh improve_fn)")
    fig.tight_layout()
    figdir = out_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    p = str(figdir / "reward_breakdown.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/rl/paraspeechcaps_en_twohop/train.jsonl")
    ap.add_argument("--output_dir", default="out/rl/vllm_smoke")
    ap.add_argument("--workdir", required=True, help="sidecar file-RPC dir")
    ap.add_argument("--merged_path", default="/dev/shm/rlvllm_merged.safetensors")
    ap.add_argument("--sync", choices=["disk", "nccl"], default="disk")
    ap.add_argument("--nccl_host", default="127.0.0.1")
    ap.add_argument("--nccl_port", type=int, default=29555)
    ap.add_argument("--policy_path", default="Step-Audio-2-mini")
    ap.add_argument("--scorer_device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="cap train rows (default: all)")
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup_steps", type=int, default=0,
                    help="linear warmup steps; 0 = no warmup")
    ap.add_argument("--lr_min_ratio", type=float, default=0.1,
                    help="cosine decay floor as fraction of peak lr")
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="grad-norm clip; 0.3 tames the KL blow-up under small effective batch")
    ap.add_argument("--clip_advantage", type=float, default=5.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--grad_checkpointing", action="store_true", default=True)
    ap.add_argument("--resume_adapter", default=None)
    # reward cfg passthrough (defaults match the TRL run)
    ap.add_argument("--alpha_wer", type=float, default=0.5)
    ap.add_argument("--beta_clsp", type=float, default=0.5)
    ap.add_argument("--wer_floor", type=float, default=0.0)
    ap.add_argument("--wer_cap", type=float, default=0.5,
                    help="one-hop absolute reward (critique_mode none): cap on the "
                         "monotone WER penalty alpha_wer*min(WER, wer_cap). Bounds "
                         "group-advantage variance while keeping gradient for WER<cap.")
    ap.add_argument("--invalid_penalty", type=float, default=-1.0)
    ap.add_argument("--min_audio_codes", type=int, default=5)
    ap.add_argument("--lambda_improve", type=float, default=1.0)
    ap.add_argument("--improve_fn", type=str, default="tanh",
                    choices=["none", "tanh", "exp"])
    ap.add_argument("--improve_scale", type=float, default=10.0)
    ap.add_argument("--wer_penalty", type=float, default=0.5)
    ap.add_argument("--wer_gate", type=float, default=0.10)
    ap.add_argument("--gate_penalty", type=float, default=-1.0)
    ap.add_argument("--wer_absolute", action="store_true", default=False,
                    help="two-hop: penalize WER_v2 absolutely (min(WER,wer_cap)), "
                         "dropping the v1-relative margin.")
    ap.add_argument("--wer_hard_gate", action="store_true", default=False,
                    help="two-hop: add gate_penalty whenever WER_v2 > wer_gate "
                         "(hard cliff to crush off-text drift rollouts).")
    ap.add_argument("--prompt_wav", default="assets/default_male.wav")
    ap.add_argument("--speaker_rag", action="store_true",
                    help="Per-prompt CLSP+gender reference selection for the reward "
                         "(speaker_rag pkg) instead of the fixed --prompt_wav.")
    ap.add_argument("--speaker_emb_cache", default=None,
                    help="Cache path for reference audio embeddings (--speaker_rag).")
    ap.add_argument("--asr_model", default="openai/whisper-large-v3")
    ap.add_argument("--clsp_model", default="yfyeung/CLSP")
    ap.add_argument("--seed", type=int, default=0, help="fixed rollout seed (disk vs nccl comparable)")
    ap.add_argument("--critique_mode", choices=["static", "self", "none"], default="static",
                    help="static: condition v2 on a precomputed/injected critique "
                         "(frozen prompt context — never trained). self: the policy "
                         "generates the critique on-policy (two-pass vLLM rollout) so "
                         "it lands in gen_ids and the GRPO gradient shapes it (Fix 1). "
                         "none: one-hop single-pass baseline (instruction+text -> audio, "
                         "no v1/critique/refine); absolute CLSP+WER reward.")
    ap.add_argument("--max_think_tokens", type=int, default=256,
                    help="critique token budget for --critique_mode self (pass-1).")
    ap.add_argument("--on_policy_critique", action="store_true", default=False,
                    help="regenerate critique each step with the current policy (option A; "
                         "static mode only — ignored when --critique_mode self)")
    ap.add_argument("--critic_temperature", type=float, default=0.3)
    ap.add_argument("--critic_max_new_tokens", type=int, default=256)
    ap.add_argument("--on_policy_v1", action="store_true", default=False,
                    help="generate v1 on-policy each step (single-pass instruction+text "
                         "-> audio) instead of reading precomputed v1_tokens from data. "
                         "v1 is NOT trained (no gradient): it only seeds the self-critique "
                         "prompt context + the reward's CLSP baseline. Use with "
                         "--critique_mode self for fully on-policy v1->critique->v2 where "
                         "only critique+v2 receive gradient.")
    ap.add_argument("--v1_temperature", type=float, default=None,
                    help="sampling temperature for on-policy v1 (default: --temperature)")
    ap.add_argument("--v1_top_p", type=float, default=0.9)
    ap.add_argument("--v1_max_new_tokens", type=int, default=None,
                    help="token budget for on-policy v1 (default: --max_new_tokens)")
    ap.add_argument("--ckpt_every", type=int, default=20, help="save adapter every N steps")
    ap.add_argument("--verify_sync", action="store_true", default=False,
                    help="one-time bit-exact weight-sync check (debug; skipped in training)")
    ap.add_argument("--mixed_group", action="store_true", default=False,
                    help="mixed-group GRPO: half rollouts are single-pass (no v1/critique), "
                         "half are two-hop. Group advantage directly rewards 'refine > blind'. "
                         "Reward switches to absolute CLSP (no relative term).")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer()

    # Auto-resume across `normal`-partition preemption (REQUEUE). Pick the
    # highest-numbered adapter and restore its optimizer/scheduler state too.
    import re
    base_done = 0
    if args.resume_adapter:
        mm = re.search(r"ckpt-step(\d+)", str(args.resume_adapter))
        base_done = int(mm.group(1)) if mm else 0
    else:
        cands = []
        for d in out.glob("ckpt-step*/adapter"):
            if (d / "adapter_model.safetensors").exists():
                mm = re.search(r"ckpt-step(\d+)", str(d))
                cands.append((int(mm.group(1)) if mm else 0, d))
        if cands:
            base_done, latest = max(cands, key=lambda t: t[0])
            args.resume_adapter = str(latest)
            print(f"[trainer] RESUME from {latest} (cumulative done: {base_done})", flush=True)
    (out / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2)
    )
    remaining = max(0, args.steps - base_done)
    print(f"[trainer] target {args.steps}, done {base_done}, running {remaining} more", flush=True)
    if remaining <= 0:
        # Target already reached after a requeue: do not reload the models.
        print(f"[trainer] target reached; exiting without work.", flush=True)
        return 0

    rows = load_rows(args.train_jsonl, args.limit)
    validate_rows(rows, args)
    if args.critique_mode == "none":
        # One-hop baseline uses the same source JSONL but ignores all two-hop
        # context and switches the scorer to its absolute reward branch.
        for r in rows:
            for k in ("v1_tokens", "critique", "clsp_v1", "wer_v1"):
                r.pop(k, None)

    client = SidecarClient(args.workdir)
    print("[trainer] waiting for sidecar ready ...", flush=True)
    client.wait_ready()
    print("[trainer] sidecar ready; building policy + scorer", flush=True)

    policy = _build_policy(args)
    scorer = TimedScorer(_build_scorer(args), timer)

    cfg = GRPOConfig(G=args.G, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                     lr=args.lr, warmup_steps=args.warmup_steps,
                     total_steps=args.steps, lr_min_ratio=args.lr_min_ratio,
                     kl_coef=args.kl_coef, grad_clip=args.grad_clip,
                     clip_advantage=args.clip_advantage, max_new_tokens=args.max_new_tokens,
                     temperature=args.temperature, log_every=1)

    # Select critique source. `self` puts critique generation inside the rewarded
    # trajectory via a two-pass vLLM rollout; `static` injects a frozen critique
    # into the prompt (optionally regenerated each step by --on_policy_critique,
    # which still never trains the critique tokens).
    mixed_group = args.mixed_group
    # the single-pass (baseline) arm is ALWAYS a plain one-pass rollout
    singlepass_rollout_fn = make_rollout_fn(client, timer, seed_base=args.seed)
    if args.critique_mode == "none":
        # One-hop / single-pass BASELINE: instruction+text -> audio directly, no
        # v1, no critique, no refine. The reward is absolute CLSP+WER
        # (beta_clsp*CLSP - alpha_wer*min(WER, wer_cap)); the two-hop v1 fields are stripped
        # from the rows below so the scorer takes the absolute path (clsp_v1=None).
        # GRPO trains the single-pass generation directly -> a clean RL comparison
        # point for the two-hop refine paradigm on the SAME training data.
        if mixed_group:
            raise SystemExit("--critique_mode none is the one-hop baseline; "
                             "it is incompatible with --mixed_group.")
        build_messages = build_singlepass_prompt_chat
        rollout_fn = singlepass_rollout_fn
        on_policy_critique = False
    elif args.critique_mode == "self":
        # A arm = two-pass self-critique (critique in <think>, then v2); its tokens
        # are in gen_ids -> GRPO trains the critique. Works WITH mixed_group now:
        # baseline arm = single-pass (no-grad), A arm = self-critique refine.
        build_messages = build_twohop_selfcritique_chat
        rollout_fn = make_selfcritique_rollout_fn(
            client, timer, seed_base=args.seed, max_think_tokens=args.max_think_tokens)
        on_policy_critique = False  # critique is in-trajectory; no separate critic
    else:
        build_messages = build_twohop_prompt_chat
        rollout_fn = singlepass_rollout_fn   # static two-hop is also one-pass (critique in prompt)
        on_policy_critique = args.on_policy_critique

    # on-policy v1: generate the first draft each step instead of reading it from
    # data. Only meaningful when there IS a v1 to refine (self/static critique);
    # the one-hop baseline has no v1 stage.
    if args.on_policy_v1 and args.critique_mode == "none":
        raise SystemExit("--on_policy_v1 needs a refine stage; use --critique_mode "
                         "self (or static), not none.")
    if args.on_policy_v1 and args.critique_mode != "self":
        print("[trainer] WARN: --on_policy_v1 with --critique_mode "
              f"{args.critique_mode}: v1 is on-policy but the critique is NOT "
              "self-generated; for full on-policy v1->critique->v2 use "
              "--critique_mode self.", flush=True)
    v1_temp = args.v1_temperature if args.v1_temperature is not None else args.temperature
    v1_max = args.v1_max_new_tokens if args.v1_max_new_tokens is not None else args.max_new_tokens

    print(f"[trainer] critique_mode={args.critique_mode} "
          f"(on_policy_critique={on_policy_critique}, on_policy_v1={args.on_policy_v1}, "
          f"mixed_group={mixed_group}, baseline_nograd={mixed_group})", flush=True)

    trainer = VLLMGRPOTrainer(policy, scorer, build_messages, cfg, out,
                              rollout_fn=rollout_fn,
                              mixed_group=mixed_group,
                              build_singlepass_messages=(build_singlepass_prompt_chat
                                                        if mixed_group else None),
                              singlepass_rollout_fn=singlepass_rollout_fn,
                              baseline_nograd=mixed_group,
                              client=client, timer=timer, merged_path=args.merged_path,
                              sync_mode=args.sync, nccl_host=args.nccl_host,
                              nccl_port=args.nccl_port,
                              on_policy_critique=on_policy_critique,
                              critic_temperature=args.critic_temperature,
                              critic_max_new_tokens=args.critic_max_new_tokens,
                              on_policy_v1=args.on_policy_v1,
                              v1_temperature=v1_temp, v1_top_p=args.v1_top_p,
                              v1_max_new_tokens=v1_max,
                              verify_first=args.verify_sync)

    if args.resume_adapter:
        restored = trainer.load_training_state(args.resume_adapter)
        if base_done and not restored:
            print("[trainer] WARN: legacy checkpoint has no trainer_state.pt; "
                  "optimizer and LR scheduler start fresh", flush=True)

    print(f"[trainer] {len(rows)} rows; running {remaining} steps (G={args.G}, "
          f"on_policy_critique={args.on_policy_critique})", flush=True)
    # Make the base trainer count from base_done so train_metrics.jsonl steps stay
    # cumulative/monotonic across preemption legs.
    trainer.global_step = base_done
    # cycle through the dataset offset by base_done so a resumed leg sees new prompts
    it = batches(rows, args.micro_batch)
    for _ in range(base_done % max(1, len(rows))):
        try:
            next(it)
        except StopIteration:
            it = batches(rows, args.micro_batch)
    metrics = []
    best = {"reward": -1e9, "step": 0, "ckpt": None}
    bf = out / "best_ckpt.json"
    if bf.exists():
        try:
            best = json.loads(bf.read_text())
        except Exception:
            pass
    t_run0 = time.time()
    for _ in range(remaining):
        try:
            batch = next(it)
        except StopIteration:
            it = batches(rows, args.micro_batch); batch = next(it)
        m = trainer.step(batch)
        timer.steps += 1
        cum = m.get("step") or (base_done + timer.steps)  # cumulative global step
        metrics.append(m)
        lr_str = f" lr={m['lr']:.2e}" if m.get("lr") is not None else ""
        msg = (f"[trainer] step {cum}: reward={m['reward_mean']:+.3f} "
               f"kl={m['kl']:+.4f} clsp={m['clsp_mean']} wer={m['wer_mean']} "
               f"invalid={m['n_invalid']}{lr_str}")
        if m.get("clsp_sp") is not None and m.get("clsp_th") is not None:
            delta = m["clsp_th"] - m["clsp_sp"]
            msg += f" sp={m['clsp_sp']:.4f} th={m['clsp_th']:.4f} delta={delta:+.4f}"
        print(msg, flush=True)
        # checkpoint + track best-reward ckpt (cumulative numbering, persisted).
        # NOTE: under grad_accum>1, `cum` (=m["step"]) reports the optimizer-step
        # counter on accumulation boundaries, so it can never hit ckpt_every/steps
        # (which are expressed in micro-steps) — guard the final leg on timer.steps
        # so a checkpoint is always saved at the end regardless of grad_accum.
        if cum % args.ckpt_every == 0 or cum >= args.steps or timer.steps >= remaining:
            ckpt = out / f"ckpt-step{cum}" / "adapter"
            trainer.save_adapter(ckpt)
            (out / ".cumulative_steps").write_text(str(cum))
            if m["reward_mean"] > best["reward"]:
                best = {"reward": m["reward_mean"], "step": cum, "ckpt": str(ckpt)}
                bf.write_text(json.dumps(best, ensure_ascii=False, indent=2))
    run_s = time.time() - t_run0
    bf.write_text(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"[trainer] BEST ckpt: {best}", flush=True)

    # derive loss/backward remainder per step
    tsum = timer.sum
    # top-level phases only (sync.* are sub-timers of sync_nccl -> exclude to avoid double count)
    accounted = (tsum.get("sync_merge_save", 0) + tsum.get("sync_nccl", 0)
                 + tsum.get("verify_once", 0) + tsum.get("rollout", 0)
                 + tsum.get("scoring", 0) + tsum.get("critique", 0))
    loss_bwd = max(0.0, tsum.get("step_total", run_s) - accounted)
    timer.sum["loss_fwd_bwd_opt"] = loss_bwd

    prof = timer.report()
    # Full cross-leg history from train_metrics.jsonl (base trainer writes it,
    # appended over preemption legs) -> the plot + summary cover ALL steps.
    all_metrics = []
    mp = out / "train_metrics.jsonl"
    if mp.exists():
        for ln in mp.read_text().splitlines():
            ln = ln.strip()
            if ln:
                try:
                    all_metrics.append(json.loads(ln))
                except Exception:
                    pass
    if not all_metrics:
        all_metrics = metrics

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    rew = [m.get("reward_mean") for m in all_metrics if m.get("reward_mean") is not None]
    cls = [m.get("clsp_mean") for m in all_metrics if m.get("clsp_mean") is not None]
    wer = [m.get("wer_mean") for m in all_metrics if m.get("wer_mean") is not None]
    last = all_metrics[-50:]
    wcheck = getattr(trainer, "_verified", None)
    summary = (
        f"✅ RL training DONE [sync={args.sync}, on_policy_critique={args.on_policy_critique}] — "
        f"target {args.steps} steps, this leg {timer.steps} ({run_s/60:.1f} min, "
        f"{timer.sum.get('step_total',run_s)/max(1,timer.steps):.1f}s/step)\n"
        f"reward: overall mean {avg(rew):.4f}, last-50 mean {avg([m.get('reward_mean') for m in last]):.4f}\n"
        f"clsp_v2: overall {avg(cls):.4f}, last-50 {avg([m.get('clsp_mean') for m in last]):.4f}\n"
        f"clsp_delta: last-50 {avg([m.get('clsp_delta_mean') for m in last]):.4f} "
        f"(improve_fn={args.improve_fn}, scale={args.improve_scale})\n"
        f"wer_v2: overall {avg(wer):.4f}, last-50 {avg([m.get('wer_mean') for m in last]):.4f}\n"
        f"BEST ckpt: step {best['step']} reward {best['reward']:.4f}\n"
        + (f"NCCL-sync weight-equality vs disk merged: {wcheck}\n" if wcheck else "")
        + f"--- time profile (G={args.G}) ---\n{prof}\n"
        f"(HF policy loss/backward; vLLM rollout via sidecar; {args.sync} weight-sync)"
    )
    (out / "smoke_summary.txt").write_text(summary)
    (out / "smoke_metrics.json").write_text(json.dumps(
        {"timing": dict(timer.sum), "run_s": run_s, "this_leg_steps": timer.steps,
         "total_steps": len(all_metrics), "best": best,
         "reward_mean": avg(rew), "clsp_mean": avg(cls), "wer_mean": avg(wer)},
        ensure_ascii=False, indent=2))
    print("\n" + summary, flush=True)

    client.stop()

    # reward-breakdown figure (full cross-leg history)
    fig_path = None
    try:
        fig_path = plot_reward_breakdown(all_metrics, rows, out)
        print(f"[trainer] wrote {fig_path}", flush=True)
    except Exception as e:
        print(f"[trainer] plot failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
