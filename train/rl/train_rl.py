"""GRPO RL training for Step-Audio-2 reasoning.

CLI entry. Wires:
  - StepAudio2 (chat variant, with <|BOT|>/<|EOT|>) as the policy
  - PEFT LoRA on model.llm
  - RewardScorer (token2wav + Whisper + CLSP)
  - GRPOTrainer.fit over data/rl/<split>/{train,dev}.jsonl

Server-only (CUDA + Step-Audio-2 weights). ``--dry_run`` produces a
single-batch rollout-and-reward smoke test without backward.

Usage:
    python -m train.rl.train_rl \
        --train_jsonl data/rl/paraspeechcaps_en/train.jsonl \
        --dev_jsonl   data/rl/paraspeechcaps_en/dev.jsonl \
        --output_dir  out/rl/psc_en_v1 \
        --policy_path Step-Audio-2-mini \
        --steps 2000 --G 4 --kl_coef 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --- torchcodec bypass (must run before importing stepaudio2 / token2wav / s3tokenizer) ---
import torch as _torch  # noqa: E402
import torchaudio as _torchaudio  # noqa: E402
import soundfile as _sf  # noqa: E402


def _sf_load(uri, *_, **__):
    data, sr = _sf.read(uri, always_2d=True, dtype="float32")
    return _torch.from_numpy(data.T.copy()), sr


def _sf_save(uri, src, sample_rate, *, format=None, channels_first=True, **__):
    if isinstance(src, _torch.Tensor):
        src = src.detach().cpu().numpy()
    if src.ndim == 2 and channels_first:
        src = src.T
    _sf.write(uri, src, sample_rate, format=(format.upper() if format else None))


_torchaudio.load = _sf_load
_torchaudio.save = _sf_save
# --- end bypass ---

# Make Step-Audio2/ importable.
THIS_DIR = Path(__file__).resolve().parent     # .../train/rl/
REPO_ROOT = THIS_DIR.parent.parent
STEP_DIR = REPO_ROOT / "Step-Audio2"
sys.path.insert(0, str(STEP_DIR))
sys.path.insert(0, str(REPO_ROOT / "train"))
sys.path.insert(0, str(REPO_ROOT))            # so `import speaker_rag` works


def _resolve_step(path: str) -> str:
    return path if os.path.isabs(path) else str(STEP_DIR / path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--dev_jsonl", default=None)
    p.add_argument("--output_dir", required=True)

    # Policy
    p.add_argument("--policy_path", default="Step-Audio-2-mini",
                   help="StepAudio2 chat variant (resolved relative to Step-Audio2/).")
    p.add_argument("--prompt_wav", default="assets/default_male.wav",
                   help="Speaker reference for token2wav. Pin one for the whole run "
                        "(used as fallback when --speaker_rag is on).")
    p.add_argument("--speaker_rag", action="store_true",
                   help="Per-prompt, pick the token2wav reference whose voice best "
                        "matches the instruction (gender filter + CLSP, speaker_rag "
                        "pkg) instead of the fixed --prompt_wav. Applied to BOTH the "
                        "v1 baseline and v2 rollouts so the CLSP improvement stays "
                        "comparable.")
    p.add_argument("--speaker_emb_cache", default=None,
                   help="Cache path for reference audio embeddings (--speaker_rag). "
                        "Default recomputes each run (only a few refs).")

    # Reward / scorer
    p.add_argument("--asr_model", default="openai/whisper-large-v3")
    p.add_argument("--clsp_model", default="yfyeung/CLSP")
    p.add_argument("--scorer_device", default="cuda:1",
                   help="Device for token2wav/Whisper/CLSP. Use 'cuda:1' on a "
                        "2-GPU box to keep the policy alone on cuda:0. "
                        "Falls back to cuda:0 / cpu when unavailable.")
    p.add_argument("--alpha_wer", type=float, default=0.5)
    p.add_argument("--beta_clsp", type=float, default=0.5)
    p.add_argument("--wer_floor", type=float, default=0.0)
    p.add_argument("--wer_cap", type=float, default=0.5,
                   help="one-hop absolute reward: cap on monotone WER penalty alpha_wer*min(WER, wer_cap)")
    p.add_argument("--invalid_penalty", type=float, default=-1.0)
    p.add_argument("--min_audio_codes", type=int, default=5)
    # two-hop relative reward
    p.add_argument("--lambda_improve", type=float, default=1.0,
                   help="weight on the (possibly non-linear) improvement term")
    p.add_argument("--improve_fn", type=str, default="tanh",
                   choices=["none", "tanh", "exp"],
                   help="non-linear transform on (CLSP_v2 - CLSP_v1); 'none' = linear")
    p.add_argument("--improve_scale", type=float, default=10.0,
                   help="scale factor inside the non-linear fn; higher = sharper")
    p.add_argument("--wer_penalty", type=float, default=0.5,
                   help="weight on soft WER penalty max(0, WER_v2 - wer_ref)")
    p.add_argument("--wer_gate", type=float, default=0.10,
                   help="absolute WER reference used only when wer_v1 is unknown")
    p.add_argument("--gate_penalty", type=float, default=-1.0,
                   help="(legacy) hard-gate penalty; unused in soft mode")

    p.add_argument("--grad_checkpointing", action="store_true", default=True,
                   help="enable gradient checkpointing on the policy (saves VRAM)")
    p.add_argument("--no_grad_checkpointing", dest="grad_checkpointing",
                   action="store_false")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

    # GRPO
    p.add_argument("--G", type=int, default=4)
    p.add_argument("--micro_batch", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl_coef", type=float, default=0.05)
    p.add_argument("--clip_advantage", type=float, default=5.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--eval_max_batches", type=int, default=20)
    p.add_argument("--eval_save_audio_samples", type=int, default=4,
                   help="Save up to this many valid eval wavs per eval step.")
    p.add_argument("--seed", type=int, default=0)

    # Mode
    p.add_argument("--mode", choices=["reasoning", "twohop"], default="twohop",
                   help="reasoning: instruction+text→think→audio (single-hop). "
                        "twohop: v1_tokens+critique→v2_audio (refine mode, recommended).")

    # Workflow
    p.add_argument("--dry_run", action="store_true",
                   help="Sample one batch of rollouts + score; skip backward. "
                        "Smoke-test the reward signal before adding gradients.")
    p.add_argument("--resume_adapter", default=None,
                   help="Path to a previously saved LoRA adapter dir to load.")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. Keep 0 to avoid CUDA fork issues.")
    return p.parse_args()


def _enable_kv_cache(llm) -> None:
    """Monkey-patch StepAudio2ForCausalLM to thread a KV cache through generation.

    The upstream `forward` (Step-Audio2/, do-not-modify) calls the inner Qwen2Model
    without `past_key_values`/`use_cache` and always returns `past_key_values=None`,
    so HF `generate()` recomputes the full sequence every decode step (O(n^2) — and
    twohop prompts embed ~2k v1-audio tokens). We patch the *class* (before PEFT
    wraps it, so peft captures the patched `prepare_inputs_for_generation` at init)
    to:
      * forward: pass past_key_values/use_cache/cache_position/position_ids to the
        inner model and return the real cache (Qwen2Model auto-creates a
        DynamicCache when use_cache=True and past_key_values is None);
      * prepare_inputs_for_generation: on decode steps feed only the new token(s).

    Training forwards (full prompt+gen, use_cache unset -> coerced False) are
    unaffected: cache only accelerates the autoregressive generation loop.
    """
    if os.environ.get("REFINETTS_DISABLE_KV_CACHE") == "1":
        print("[rl] KV cache patch DISABLED via REFINETTS_DISABLE_KV_CACHE=1")
        return

    from transformers.modeling_outputs import CausalLMOutputWithPast

    cls = type(llm)
    if getattr(cls, "_kv_cache_patched", False):
        return

    AUDIO_PLACEHOLDER_ID = 151688  # input_ids marker where audio mels get spliced

    def forward(self, input_ids=None, wavs=None, wav_lens=None, attention_mask=None,
                past_key_values=None, use_cache=None, cache_position=None,
                position_ids=None, inputs_embeds=None, **kwargs):
        if use_cache is None:
            use_cache = False  # training/logprob forwards: single full pass, no cache
        if inputs_embeds is None:
            hidden_states = self.model.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        if wavs is not None:
            if self.bf16:
                wavs = wavs.bfloat16()
            out, feat_lens = self.encoder(wavs, wav_lens)
            out = self.adapter(out)
            feat_lens = (feat_lens - 1) // 2 + 1
            insert_location = _torch.nonzero(input_ids == AUDIO_PLACEHOLDER_ID)
            insert_location[:, 1] += 1
            for idx in range(len(insert_location)):
                i, s = insert_location[idx]
                hidden_states[i][s : s + feat_lens[idx]] = out[idx][:feat_lens[idx]]
        outputs = self.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        logits = self.lm_head(outputs[0])
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, cache_position=None,
                                      **kwargs):
        if past_key_values is not None:
            # decode step: only the newest token(s) are not yet cached
            if cache_position is not None:
                input_ids = input_ids[:, cache_position]
            else:
                input_ids = input_ids[:, -1:]
            return {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "use_cache": True,
            }
        # prefill: full prompt (+ audio mels if any)
        return {
            "input_ids": input_ids,
            "past_key_values": None,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "use_cache": kwargs.get("use_cache", True),
            "wavs": kwargs.get("wavs"),
            "wav_lens": kwargs.get("wav_lens"),
        }

    cls.forward = forward
    cls.prepare_inputs_for_generation = prepare_inputs_for_generation
    cls._kv_cache_patched = True
    print("[rl] KV cache enabled on policy (patched forward + prepare_inputs)")


def _build_policy(args):
    """Load StepAudio2 (chat variant) and attach a LoRA adapter on llm."""
    from stepaudio2 import StepAudio2  # type: ignore
    from peft import LoraConfig, get_peft_model, PeftModel

    model_dir = _resolve_step(args.policy_path)
    print(f"[rl] loading policy: {model_dir}")
    policy = StepAudio2(model_dir)

    # Patch the base class BEFORE PEFT wraps it (peft captures
    # prepare_inputs_for_generation at construction time).
    _enable_kv_cache(policy.llm)

    if args.resume_adapter:
        print(f"[rl] resuming LoRA adapter from {args.resume_adapter}")
        policy.llm = PeftModel.from_pretrained(policy.llm, args.resume_adapter, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy.llm = get_peft_model(policy.llm, lora_cfg)

    if hasattr(policy.llm, "print_trainable_parameters"):
        policy.llm.print_trainable_parameters()

    # Gradient checkpointing: trade compute for activation memory. The GRPO
    # re-scoring forward runs over prompt+gen for all G rollouts with
    # use_cache=False; two-hop prompts embed the full v1 audio (~2k tokens), so
    # activations dominate VRAM. Checkpointing makes an 80GB GPU fit comfortably.
    if getattr(args, "grad_checkpointing", True):
        try:
            policy.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(policy.llm, "enable_input_require_grads"):
                policy.llm.enable_input_require_grads()
            print("[rl] gradient checkpointing enabled (use_reentrant=False)")
        except Exception as e:
            print(f"[rl] gradient checkpointing not enabled: {e}")
    return policy


def _resolve_scorer_device(requested: str) -> _torch.device:
    """Pick the scorer device, falling back gracefully if unavailable."""
    if not _torch.cuda.is_available():
        return _torch.device("cpu")
    dev = _torch.device(requested)
    if dev.type == "cuda" and (dev.index or 0) >= _torch.cuda.device_count():
        print(f"[rl] scorer_device {requested} not present "
              f"(have {_torch.cuda.device_count()} GPU); using cuda:0")
        return _torch.device("cuda:0")
    return dev


def _build_scorer(args):
    from token2wav import Token2wav  # type: ignore
    from .reward import RewardScorer, RewardConfig, asr_load, clsp_load

    policy_dir = _resolve_step(args.policy_path)
    device = _resolve_scorer_device(args.scorer_device)
    cuda_idx = device.index if device.type == "cuda" else None
    print(f"[rl] scorer device: {device}")

    # token2wav hard-codes .cuda()/device='cuda' internally; build it under a
    # cuda-device context so those resolve to the chosen GPU.
    if cuda_idx is not None:
        with _torch.cuda.device(cuda_idx):
            token2wav = Token2wav(os.path.join(policy_dir, "token2wav"))
    else:
        token2wav = Token2wav(os.path.join(policy_dir, "token2wav"))

    print(f"[rl] loading ASR: {args.asr_model}")
    asr, asr_processor = asr_load(args.asr_model, device=device)
    print(f"[rl] loading CLSP: {args.clsp_model}")
    clsp = clsp_load(args.clsp_model, device=device)

    # Speaker RAG: embed the candidate references once (reusing the scorer's
    # CLSP + device). None -> disabled, fixed prompt_wav everywhere.
    references = ref_embeddings = None
    if getattr(args, "speaker_rag", False):
        from speaker_rag import DEFAULT_REFERENCES, embed_references
        references = DEFAULT_REFERENCES
        ref_embeddings = embed_references(
            references, clsp, device, cache_path=getattr(args, "speaker_emb_cache", None),
        )
        names = ", ".join(r.name for r in references)
        print(f"[rl] speaker_rag ON: {len(references)} references [{names}]")

    cfg = RewardConfig(
        alpha_wer=args.alpha_wer,
        beta_clsp=args.beta_clsp,
        wer_floor=args.wer_floor,
        wer_cap=args.wer_cap,
        invalid_penalty=args.invalid_penalty,
        min_audio_codes=args.min_audio_codes,
        lambda_improve=args.lambda_improve,
        improve_fn=args.improve_fn,
        improve_scale=args.improve_scale,
        wer_penalty=args.wer_penalty,
        wer_gate=args.wer_gate,
        gate_penalty=args.gate_penalty,
        wer_absolute=getattr(args, "wer_absolute", False),
        wer_hard_gate=getattr(args, "wer_hard_gate", False),
    )
    print(f"[rl] reward cfg: beta_clsp={cfg.beta_clsp} lambda_improve={cfg.lambda_improve} "
          f"improve_fn={cfg.improve_fn} improve_scale={cfg.improve_scale} | "
          f"wer_penalty={cfg.wer_penalty} wer_absolute={cfg.wer_absolute} "
          f"wer_hard_gate={cfg.wer_hard_gate} wer_gate={cfg.wer_gate} "
          f"gate_penalty={cfg.gate_penalty} wer_cap={cfg.wer_cap}", flush=True)
    return RewardScorer(
        token2wav=token2wav,
        asr_model=asr,
        asr_processor=asr_processor,
        clsp_model=clsp,
        prompt_wav=_resolve_step(args.prompt_wav),
        cfg=cfg,
        device=device,
        token2wav_cuda_idx=cuda_idx,
        references=references,
        ref_embeddings=ref_embeddings,
    )


def _write_run_config(args, output_dir: Path) -> None:
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (output_dir / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2)
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(args, output_dir)

    _torch.manual_seed(args.seed)

    from torch.utils.data import DataLoader
    from .data import RLPromptDataset, collate_rl
    from .prompt import build_rl_prompt_chat, build_twohop_prompt_chat
    from .rollout import sample_rollouts, sample_rollouts_twohop
    from .trainer import GRPOConfig, GRPOTrainer

    if args.mode == "twohop":
        build_messages = build_twohop_prompt_chat
        rollout_fn = sample_rollouts_twohop
        print("[rl] mode=twohop: v1_tokens+critique → v2 audio")
    else:
        build_messages = build_rl_prompt_chat
        rollout_fn = sample_rollouts
        print("[rl] mode=reasoning: instruction+text → think → audio")

    train_ds = RLPromptDataset(args.train_jsonl)
    print(f"[rl] train: {len(train_ds)} prompts ({args.train_jsonl})")
    train_loader = DataLoader(
        train_ds, batch_size=args.micro_batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_rl,
        drop_last=True,
    )
    dev_loader = None
    if args.dev_jsonl:
        dev_ds = RLPromptDataset(args.dev_jsonl)
        print(f"[rl] dev: {len(dev_ds)} prompts ({args.dev_jsonl})")
        dev_loader = DataLoader(
            dev_ds, batch_size=args.micro_batch, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_rl,
        )

    policy = _build_policy(args)
    scorer = _build_scorer(args)

    if args.dry_run:
        batch = next(iter(train_loader))
        print(f"[rl:dry_run] mode={args.mode} sampling G={args.G} rollouts for "
              f"uid={batch['uids'][0]} ...")
        v1_toks = batch["v1_tokens"][0] if "v1_tokens" in batch else None
        crit = batch["critiques"][0] if "critiques" in batch else None
        messages = build_messages(
            batch["instructions"][0], batch["texts"][0], batch["langs"][0],
            v1_tokens=v1_toks, critique=crit,
        )
        rollouts = rollout_fn(
            policy, messages,
            G=args.G,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=policy.eos_token_id,
        )
        scored = scorer(rollouts, ref_text=batch["texts"][0],
                        instruction=batch["instructions"][0],
                        lang=batch["langs"][0])
        for i, (r, s) in enumerate(zip(rollouts, scored)):
            print(f"  rollout[{i}] valid={s['valid']} reward={s['reward']:.3f} "
                  f"wer={s.get('wer')} clsp={s.get('clsp')} "
                  f"n_audio={len(r.audio_codes)} truncated={r.truncated}")
            print(f"    think: {r.think_text[:160]}")
            if s.get("hyp"):
                print(f"    hyp:   {s['hyp'][:160]}")
        return 0

    cfg = GRPOConfig(
        G=args.G,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        lr=args.lr,
        kl_coef=args.kl_coef,
        clip_advantage=args.clip_advantage,
        grad_clip=args.grad_clip,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        log_every=args.log_every,
        eval_every=args.eval_every,
        ckpt_every=args.ckpt_every,
        eval_max_batches=args.eval_max_batches,
        eval_save_audio_samples=args.eval_save_audio_samples,
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        policy=policy,
        scorer=scorer,
        build_messages=build_messages,
        cfg=cfg,
        output_dir=output_dir,
        rollout_fn=rollout_fn,
    )
    trainer.fit(train_loader, dev_loader, max_steps=args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
