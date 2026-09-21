"""GRPO trainer with LoRA on Step-Audio-2 LM (audio encoder + token2wav frozen).

Per step:
  1. For each of `micro_batch` prompts, sample G rollouts under the current
     policy (no grad).
  2. Reward each rollout (token2wav -> wav -> ASR + CLSP).
  3. advantage_i = (r_i - mean_g) / (std_g + eps), optionally clipped.
  4. Re-score gen_ids under the current policy WITH grad ->  log pi_theta.
  5. Re-score gen_ids under the frozen base (LoRA off) ->   log pi_ref.
  6. loss = -mean(adv * log_pi * mask) + kl_coef * mean((log_pi - log_ref) * mask)
  7. Backward through LoRA params only; AdamW step.

There is no value network or PPO clipping. The KL term
keeps the policy from drifting away from the base distribution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .rollout import (
    Rollout,
    policy_logprobs,
    reference_logprobs,
    sample_rollouts,
    sample_rollouts_twohop,
)


@dataclass
class GRPOConfig:
    G: int = 4
    micro_batch: int = 1
    grad_accum: int = 4
    lr: float = 1e-5
    warmup_steps: int = 0
    total_steps: int = 0          # 0 = no cosine decay (constant LR)
    lr_min_ratio: float = 0.1     # cosine decays to lr * lr_min_ratio
    kl_coef: float = 0.05
    clip_advantage: float = 5.0
    grad_clip: float = 1.0
    max_new_tokens: int = 1024
    temperature: float = 0.9
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    log_every: int = 10
    eval_every: int = 200
    ckpt_every: int = 500
    eval_max_batches: int = 20
    eval_temperature: float = 0.4
    eval_save_audio_samples: int = 4
    seed: int = 0


class GRPOTrainer:
    def __init__(
        self,
        policy,
        scorer,
        build_messages: Callable,
        cfg: GRPOConfig,
        output_dir: str | Path,
        rollout_fn: Callable = sample_rollouts,
        mixed_group: bool = False,
        build_singlepass_messages: Callable | None = None,
        singlepass_rollout_fn: Callable | None = None,
        baseline_nograd: bool = False,
    ):
        self.policy = policy
        self.scorer = scorer
        self.build_messages = build_messages
        self.rollout_fn = rollout_fn
        self.mixed_group = mixed_group
        self.build_singlepass_messages = build_singlepass_messages
        # single-pass (baseline) arm may need a DIFFERENT rollout fn than the A arm:
        # in critique_mode=self the A arm is a two-pass self-critique rollout, but the
        # single-pass baseline must be a plain one-pass gen. Defaults to rollout_fn
        # (correct for static mixed, where both arms are one-pass).
        self.singlepass_rollout_fn = singlepass_rollout_fn or rollout_fn
        # baseline_nograd: single-pass rewards still enter the group mean/std (so they
        # set the bar the A arm must beat) but their rollouts are NOT backpropped —
        # otherwise GRPO pushes the (below-mean) baseline down = "sandbag without critique".
        self.baseline_nograd = baseline_nograd
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        params = [p for p in self.policy.llm.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable params on policy.llm — did you forget LoRA?")
        self.opt = torch.optim.AdamW(params, lr=cfg.lr)
        self.scheduler = self._build_scheduler(cfg)
        self.global_step = 0
        self._accum = 0
        self._train_log = (self.output_dir / "train_metrics.jsonl").open("a", buffering=1)
        self._dev_log = (self.output_dir / "dev_metrics.jsonl").open("a", buffering=1)
        self._sample_log = (self.output_dir / "rollout_samples.jsonl").open("a", buffering=1)

    def _build_scheduler(self, cfg: GRPOConfig):
        if cfg.total_steps <= 0 and cfg.warmup_steps <= 0:
            return None
        import math
        total = max(cfg.total_steps, 1)
        warmup = cfg.warmup_steps

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return cfg.lr_min_ratio + (1 - cfg.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    # ----- rollout + reward + per-prompt loss -----
    def _process_prompt(
        self,
        instruction: str,
        text: str,
        lang: str,
        v1_tokens: list[int] | None = None,
        critique: str | None = None,
        clsp_v1: float | None = None,
        wer_v1: float | None = None,
        clsp_v1_reference: str | None = None,
        loss_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        rollout_kw = dict(
            max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
            top_p=cfg.top_p, repetition_penalty=cfg.repetition_penalty,
            eos_token_id=self.policy.eos_token_id,
        )

        # ---- Generate rollouts (mixed-group or standard) ----
        n_singlepass = 0
        if self.mixed_group:
            # critique_mode=self carries no static critique; the A arm self-generates
            # it, so a missing `critique` must NOT collapse us to all-single-pass.
            can_refine = bool(v1_tokens) and (bool(critique) or self.build_messages.__name__ == "build_twohop_selfcritique_chat")
            if can_refine:
                G_sp = cfg.G // 2
                G_th = cfg.G - G_sp
                # baseline arm: plain single-pass (instruction+text, no v1/critique)
                sp_msgs = self.build_singlepass_messages(instruction, text, lang)
                sp_rollouts = self.singlepass_rollout_fn(
                    self.policy, sp_msgs, G=G_sp, **rollout_kw)
                # A arm: refine (static two-hop OR self-critique two-pass, per build_messages/rollout_fn)
                th_msgs = self.build_messages(
                    instruction, text, lang,
                    v1_tokens=v1_tokens, critique=critique)
                th_rollouts = self.rollout_fn(
                    self.policy, th_msgs, G=G_th, **rollout_kw)
                rollouts = sp_rollouts + th_rollouts   # baseline FIRST (indices < n_singlepass)
                n_singlepass = len(sp_rollouts)
            else:
                sp_msgs = self.build_singlepass_messages(instruction, text, lang)
                rollouts = self.singlepass_rollout_fn(
                    self.policy, sp_msgs, G=cfg.G, **rollout_kw)
                n_singlepass = len(rollouts)
        else:
            messages = self.build_messages(
                instruction, text, lang,
                v1_tokens=v1_tokens, critique=critique,
            )
            rollouts = self.rollout_fn(
                self.policy, messages, G=cfg.G, **rollout_kw)

        if not rollouts:
            return None, {"reward_mean": 0.0, "reward_std": 0.0, "kl": 0.0,
                          "pg_loss": 0.0, "n_invalid": cfg.G, "skipped": True}

        # Score: mixed-group uses absolute reward so sp and th are on the same
        # scale and the group advantage directly compares them.
        if self.mixed_group:
            scored = self.scorer(rollouts, ref_text=text, instruction=instruction,
                                 lang=lang, clsp_v1=None, wer_v1=None)
        else:
            scored = self.scorer(rollouts, ref_text=text, instruction=instruction,
                                 lang=lang, v1_tokens=v1_tokens, clsp_v1=clsp_v1,
                                 wer_v1=wer_v1,
                                 clsp_v1_reference=clsp_v1_reference)

        rewards = torch.tensor([s["reward"] for s in scored], dtype=torch.float32)

        # Group-relative advantage. A zero-variance group has no policy-gradient
        # signal, but Eq. (6)'s KL regularizer still applies; keep the backward
        # pass with zero advantages instead of returning early.
        low_variance = (rewards.numel() < 2
                        or rewards.std(unbiased=False).item() < 1e-6)
        if low_variance:
            adv = torch.zeros_like(rewards)
        else:
            adv = (rewards - rewards.mean()) / (
                rewards.std(unbiased=False) + 1e-8
            )
            adv = adv.clamp(-cfg.clip_advantage, cfg.clip_advantage)

        # Per-rollout logprob + incremental backward. Scoring each rollout's
        # gen_ids on its own (batch=1) keeps the [seq, vocab] logits tensor —
        # the dominant memory cost for step-audio's ~158k vocab — independent of
        # G. We backward each rollout's loss immediately so its autograd graph
        # frees before the next forward; gradients accumulate in .grad as usual.
        # Match Eq. (6) in the paper: normalize each rollout by its own sequence
        # length, then average the G rollout losses. Normalizing by the total
        # token count instead would give longer utterances more weight.
        device = next(self.policy.llm.parameters()).device
        adv_l = adv.tolist()
        # no-grad baseline: single-pass rollouts (indices < n_singlepass) stay in the
        # advantage (they set the bar via the group mean above) but are NOT trained.
        skip_baseline = self.mixed_group and self.baseline_nograd and n_singlepass > 0
        trained_idx = [j for j in range(len(rollouts))
                       if not (skip_baseline and j < n_singlepass)]
        n_trained = max(1, len(trained_idx))
        pg_sum = 0.0
        kl_sum = 0.0
        for j in trained_idx:
            r = rollouts[j]
            prompt_ids_j = r.prompt_ids.unsqueeze(0).to(device)     # per-rollout prompt
            gen_j = r.gen_ids.unsqueeze(0).to(device)               # [1, L_j]
            attn_j = torch.ones(1, prompt_ids_j.shape[-1] + gen_j.shape[-1],
                                dtype=torch.long, device=device)
            logp_pol_j = policy_logprobs(self.policy.llm, prompt_ids_j, gen_j, attn_j)
            logp_ref_j = reference_logprobs(self.policy.llm, prompt_ids_j, gen_j, attn_j)
            pg_j = -(logp_pol_j * adv_l[j]).sum()
            # Schulman k3 KL estimator: exp(d) - d - 1 with d = logp_ref - logp_pol.
            # Always >= 0 and its gradient pulls the policy toward the reference.
            d_j = (logp_ref_j - logp_pol_j).clamp(-10.0, 10.0)
            kl_j = (torch.exp(d_j) - d_j - 1.0).sum()
            seq_len = max(1, r.gen_ids.numel())
            rollout_scale = loss_scale / (n_trained * seq_len)
            loss_j = (pg_j + cfg.kl_coef * kl_j) * rollout_scale
            loss_j.backward()
            pg_sum += float(pg_j.detach()) / seq_len
            kl_sum += float(kl_j.detach()) / seq_len

        stats = self._stats(rewards, scored,
                            kl=kl_sum / n_trained,
                            pg=pg_sum / n_trained,
                            skipped=low_variance,
                            n_singlepass=n_singlepass)
        self._log_sample(scored, rollouts)
        return None, stats

    def _stats(self, rewards: torch.Tensor, scored: list[dict], *,
               kl: float, pg: float, skipped: bool = False,
               n_singlepass: int = 0) -> dict:
        valid = [s for s in scored if s["valid"]]
        out = {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)) if rewards.numel() > 1 else 0.0,
            "wer_mean": (sum(s["wer"] for s in valid) / len(valid)) if valid else None,
            "clsp_mean": (sum(s["clsp"] for s in valid) / len(valid)) if valid else None,
            "clsp_delta_mean": (sum(s["clsp_delta"] for s in valid if "clsp_delta" in s)
                                / max(1, sum(1 for s in valid if "clsp_delta" in s))) if valid else None,
            "improve_transformed_mean": (sum(s["improve_transformed"] for s in valid if "improve_transformed" in s)
                                         / max(1, sum(1 for s in valid if "improve_transformed" in s))) if valid else None,
            "n_invalid": sum(1 for s in scored if not s["valid"]),
            "kl": kl,
            "pg_loss": pg,
            "skipped": skipped,
        }
        if 0 < n_singlepass < len(scored):
            sp_valid = [s for s in scored[:n_singlepass] if s["valid"]]
            th_valid = [s for s in scored[n_singlepass:] if s["valid"]]
            out["clsp_sp"] = (sum(s["clsp"] for s in sp_valid) / len(sp_valid)) if sp_valid else None
            out["clsp_th"] = (sum(s["clsp"] for s in th_valid) / len(th_valid)) if th_valid else None
        return out

    def _log_sample(self, scored: list[dict], rollouts: list[Rollout]) -> None:
        if not scored:
            return
        # Pick the highest-reward rollout for this prompt
        best = max(range(len(scored)), key=lambda i: scored[i]["reward"])
        s = scored[best]
        r = rollouts[best]
        self._sample_log.write(json.dumps({
            "step": self.global_step,
            "think_text": r.think_text[:500],
            "n_audio_codes": len(r.audio_codes),
            "truncated": r.truncated,
            "reward": s["reward"],
            "wer": s.get("wer"),
            "clsp": s.get("clsp"),
            "clsp_v1": s.get("clsp_v1"),
            "clsp_delta": s.get("clsp_delta"),
            "improve_transformed": s.get("improve_transformed"),
            "wer_v1": s.get("wer_v1"),
            "hyp": s.get("hyp", "")[:200],
            "valid": s["valid"],
        }, ensure_ascii=False) + "\n")

    # ----- public step / fit / eval -----
    def step(self, batch: dict) -> dict:
        cfg = self.cfg
        agg: dict = {"step": None, "uids": batch["uids"],
                     "reward_mean": 0.0, "reward_std": 0.0,
                     "kl": 0.0, "pg_loss": 0.0, "n_invalid": 0,
                     "wer_mean": None, "clsp_mean": None,
                     "clsp_delta_mean": None, "improve_transformed_mean": None,
                     "clsp_sp": None, "clsp_th": None,
                     "n_processed": 0, "wall": 0.0}
        t0 = time.time()
        n = len(batch["uids"])
        v1_tokens_list = batch.get("v1_tokens")
        critiques_list = batch.get("critiques")
        clsp_v1_list = batch.get("clsp_v1")
        wer_v1_list = batch.get("wer_v1")
        clsp_v1_reference_list = batch.get("clsp_v1_reference")
        # _process_prompt backwards per-rollout internally; fold the grad-accum
        # normalization into loss_scale so gradients land at the right magnitude.
        loss_scale = 1.0 / max(1, n * cfg.grad_accum)
        for i in range(n):
            _loss, stats = self._process_prompt(
                batch["instructions"][i], batch["texts"][i], batch["langs"][i],
                v1_tokens=v1_tokens_list[i] if v1_tokens_list else None,
                critique=critiques_list[i] if critiques_list else None,
                clsp_v1=clsp_v1_list[i] if clsp_v1_list else None,
                wer_v1=wer_v1_list[i] if wer_v1_list else None,
                clsp_v1_reference=(clsp_v1_reference_list[i]
                                   if clsp_v1_reference_list else None),
                loss_scale=loss_scale,
            )
            for k in ("reward_mean", "reward_std", "kl", "pg_loss"):
                agg[k] += stats[k]
            agg["n_invalid"] += stats["n_invalid"]
            for k in ("wer_mean", "clsp_mean", "clsp_delta_mean",
                      "improve_transformed_mean", "clsp_sp", "clsp_th"):
                v = stats.get(k)
                if v is not None:
                    agg[k] = (agg[k] or 0.0) + v
            agg["n_processed"] += 1

        for k in ("reward_mean", "reward_std", "kl", "pg_loss"):
            agg[k] /= max(1, agg["n_processed"])
        for k in ("wer_mean", "clsp_mean", "clsp_delta_mean",
                  "improve_transformed_mean", "clsp_sp", "clsp_th"):
            if agg[k] is not None:
                agg[k] /= max(1, agg["n_processed"])

        self._accum += 1
        if self._accum >= cfg.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy.llm.parameters() if p.requires_grad],
                cfg.grad_clip,
            )
            self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.opt.zero_grad(set_to_none=True)
            self._accum = 0
            self.global_step += 1
            agg["step"] = self.global_step
            agg["lr"] = self.opt.param_groups[0]["lr"]
            agg["wall"] = time.time() - t0
            self._train_log.write(json.dumps(agg, ensure_ascii=False) + "\n")
        return agg

    def fit(self, train_loader, dev_loader, max_steps: int) -> None:
        cfg = self.cfg
        for batch in _infinite(train_loader):
            metrics = self.step(batch)
            if metrics["step"] is None:
                continue  # mid-accumulation, no opt step yet
            if self.global_step % cfg.log_every == 0:
                print(
                    f"[step {self.global_step}] "
                    f"reward={metrics['reward_mean']:+.3f}±{metrics['reward_std']:.3f} "
                    f"kl={metrics['kl']:+.4f} pg={metrics['pg_loss']:+.4f} "
                    f"invalid={metrics['n_invalid']} "
                    f"wer={metrics['wer_mean']} clsp={metrics['clsp_mean']} "
                    f"({metrics['wall']:.1f}s)"
                )
            if dev_loader is not None and self.global_step % cfg.eval_every == 0:
                self.evaluate(dev_loader)
            if self.global_step % cfg.ckpt_every == 0:
                self.save_adapter(self.output_dir / f"ckpt-step{self.global_step}" / "adapter")
            if self.global_step >= max_steps:
                self.save_adapter(self.output_dir / "adapter")
                return
        self.save_adapter(self.output_dir / "adapter")

    @torch.no_grad()
    def evaluate(self, dev_loader, max_batches: int | None = None) -> dict:
        max_batches = max_batches if max_batches is not None else self.cfg.eval_max_batches
        rewards: list[float] = []
        wers: list[float] = []
        clsps: list[float] = []
        n_invalid = 0
        n_total = 0
        n_audio_saved = 0
        for n_seen, batch in enumerate(dev_loader):
            if n_seen >= max_batches:
                break
            v1_tokens_list = batch.get("v1_tokens")
            critiques_list = batch.get("critiques")
            clsp_v1_list = batch.get("clsp_v1")
            wer_v1_list = batch.get("wer_v1")
            clsp_v1_reference_list = batch.get("clsp_v1_reference")
            for i in range(len(batch["uids"])):
                messages = self.build_messages(
                    batch["instructions"][i], batch["texts"][i], batch["langs"][i],
                    v1_tokens=v1_tokens_list[i] if v1_tokens_list else None,
                    critique=critiques_list[i] if critiques_list else None,
                )
                rollouts = self.rollout_fn(
                    self.policy, messages,
                    G=1,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.eval_temperature,
                    top_p=self.cfg.top_p,
                    repetition_penalty=self.cfg.repetition_penalty,
                    eos_token_id=self.policy.eos_token_id,
                    do_sample=self.cfg.eval_temperature > 0,
                )
                scored = self.scorer(rollouts, ref_text=batch["texts"][i],
                                     instruction=batch["instructions"][i],
                                     lang=batch["langs"][i],
                                     v1_tokens=v1_tokens_list[i] if v1_tokens_list else None,
                                     clsp_v1=clsp_v1_list[i] if clsp_v1_list else None,
                                     wer_v1=wer_v1_list[i] if wer_v1_list else None,
                                     clsp_v1_reference=(clsp_v1_reference_list[i]
                                                        if clsp_v1_reference_list else None))
                for r, s in zip(rollouts, scored):
                    n_total += 1
                    rewards.append(s["reward"])
                    if s["valid"]:
                        wers.append(s["wer"])
                        clsps.append(s["clsp"])
                        n_audio_saved = self._save_eval_audio_sample(
                            uid=batch["uids"][i],
                            rollout=r,
                            score=s,
                            saved_count=n_audio_saved,
                        )
                    else:
                        n_invalid += 1
        out = {
            "step": self.global_step,
            "n_total": n_total,
            "n_invalid": n_invalid,
            "n_audio_saved": n_audio_saved,
            "reward_mean": (sum(rewards) / len(rewards)) if rewards else None,
            "wer_mean":   (sum(wers)   / len(wers))   if wers   else None,
            "clsp_mean":  (sum(clsps)  / len(clsps))  if clsps  else None,
        }
        self._dev_log.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"[eval step {self.global_step}] "
              f"n={n_total} reward={out['reward_mean']} "
              f"wer={out['wer_mean']} clsp={out['clsp_mean']} "
              f"invalid={n_invalid} audio_saved={n_audio_saved}")
        return out

    def _save_eval_audio_sample(
        self,
        *,
        uid: str,
        rollout: Rollout,
        score: dict,
        saved_count: int,
    ) -> int:
        if saved_count >= self.cfg.eval_save_audio_samples:
            return saved_count
        if not score.get("valid") or not rollout.audio_codes:
            return saved_count

        step_dir = self.output_dir / "eval_audio" / f"step{self.global_step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        safe_uid = "".join(c if c.isalnum() or c in "._-" else "_" for c in uid)[:120]
        stem = f"{saved_count:02d}_{safe_uid}"
        wav_path = step_dir / f"{stem}.wav"
        meta_path = step_dir / f"{stem}.json"
        self.scorer.save_audio_codes(rollout.audio_codes, str(wav_path))
        meta_path.write_text(json.dumps({
            "step": self.global_step,
            "uid": uid,
            "wav": str(wav_path),
            "reward": score.get("reward"),
            "wer": score.get("wer"),
            "clsp": score.get("clsp"),
            "hyp": score.get("hyp", ""),
            "n_audio_codes": len(rollout.audio_codes),
            "truncated": rollout.truncated,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return saved_count + 1

    def save_adapter(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.policy.llm, "save_pretrained"):
            self.policy.llm.save_pretrained(str(path))
            torch.save({
                "optimizer": self.opt.state_dict(),
                "scheduler": (self.scheduler.state_dict()
                              if self.scheduler is not None else None),
                "global_step": self.global_step,
                "grad_accum_position": self._accum,
            }, path / "trainer_state.pt")
            print(f"[ckpt] saved adapter -> {path}")
        else:
            print(f"[ckpt] policy.llm has no save_pretrained; skipped {path}")

    def load_training_state(self, adapter_path: str | Path) -> bool:
        """Restore optimizer/scheduler state saved beside a LoRA adapter."""
        state_path = Path(adapter_path) / "trainer_state.pt"
        if not state_path.exists():
            return False
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.opt.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state.get("global_step", 0))
        self._accum = int(state.get("grad_accum_position", 0))
        print(f"[ckpt] restored trainer state from {state_path}")
        return True


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch
