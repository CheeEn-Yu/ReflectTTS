"""vLLM rollout sidecar (runs in the `vllm` env, separate process from the trainer).

The trainer (step env, transformers 4.49) and vLLM (vllm env, transformers 5.x)
have incompatible deps, so they run as two processes in one slurm job and talk
over a tiny file-RPC. This sidecar:
  * loads the Qwen2-view in stock vLLM (prefix caching OFF — KV cached under old
    weights is stale after a hot weight update; see vllm-weight-hotload finding),
  * serves two ops over <workdir>/request.json -> <workdir>/response.json:
      sync    : collective_rpc hot-load merged weights from a path (on-policy)
      rollout : generate G completions for each prompt (raw token ids)
  * writes <workdir>/ready when the engine is up.

One request in flight at a time (synchronous RPC); ids guard against stale reads.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _atomic_write(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view_dir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--enforce_eager", action="store_true", default=False)
    ap.add_argument("--poll", type=float, default=0.03)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    req_path, resp_path, ready_path = wd / "request.json", wd / "response.json", wd / "ready"

    print(f"[sidecar] loading vLLM view {args.view_dir}", flush=True)
    llm = LLM(model=args.view_dir, tokenizer=args.view_dir, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
              enforce_eager=args.enforce_eager, enable_prefix_caching=False,
              worker_extension_cls="train.rl_vllm.worker_ext.WorkerExt")
    _atomic_write(ready_path, {"ready": True})
    print("[sidecar] ready", flush=True)

    while True:
        if not req_path.exists():
            time.sleep(args.poll)
            continue
        try:
            req = json.loads(req_path.read_text())
        except Exception:
            time.sleep(args.poll)
            continue
        os.remove(req_path)  # consume
        rid, op = req.get("id"), req.get("op")

        if op == "stop":
            _atomic_write(resp_path, {"id": rid, "ok": True})
            print("[sidecar] stop", flush=True)
            break

        if op == "sync":
            t0 = time.perf_counter()
            n = llm.collective_rpc("hot_load_from_disk", args=(req["merged_path"],))
            _atomic_write(resp_path, {"id": rid, "ok": True, "n_loaded": n,
                                      "t": time.perf_counter() - t0})
            continue

        if op == "init_nccl":
            llm.collective_rpc("init_nccl", args=(req["host"], req["port"],
                                                  req["rank"], req["world_size"]))
            _atomic_write(resp_path, {"id": rid, "ok": True})
            continue

        if op == "sync_nccl":
            # worker recvs the broadcasts the trainer is about to send (same order)
            t0 = time.perf_counter()
            n = llm.collective_rpc("recv_and_load", args=(req["specs"],))
            _atomic_write(resp_path, {"id": rid, "ok": True, "n_loaded": n,
                                      "t": time.perf_counter() - t0})
            continue

        if op == "compare":
            d = llm.collective_rpc("compare_to_file", args=(req["merged_path"],))
            _atomic_write(resp_path, {"id": rid, "ok": True, "diff": d})
            continue

        if op == "rollout":
            s = req["sampling"]
            # Optional stop strings (e.g. "</think>" for the critique pass). vLLM
            # only checks stop strings when it detokenizes, so flip detokenize on
            # whenever stop is supplied; the audio pass passes no stop and stays on
            # the fast detokenize=False path. include_stop_str_in_output defaults
            # False, so the returned token_ids exclude the stop string.
            stop = s.get("stop")
            sp = SamplingParams(n=s["n"], temperature=s["temperature"], top_p=s["top_p"],
                                top_k=s.get("top_k", -1),
                                max_tokens=s["max_tokens"], repetition_penalty=s["repetition_penalty"],
                                stop_token_ids=[s["eos"]], stop=stop,
                                detokenize=bool(stop), seed=s.get("seed"))
            prompts = [{"prompt_token_ids": ids} for ids in req["prompts"]]
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp)
            comps = [[list(o.outputs[g].token_ids) for g in range(len(o.outputs))] for o in outs]
            _atomic_write(resp_path, {"id": rid, "ok": True, "completions": comps,
                                      "t": time.perf_counter() - t0})
            continue

        _atomic_write(resp_path, {"id": rid, "ok": False, "err": f"unknown op {op}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
