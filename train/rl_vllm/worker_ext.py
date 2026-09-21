"""vLLM worker-side extension: in-place weight hot-update for on-policy RL.

Mixed into every stock-vLLM worker process via `LLM(worker_extension_cls=...)`
and invoked by NAME through `collective_rpc("<method>", ...)` — a string method
name is serializable, unlike passing a raw function (vLLM 0.22 blocks that by
default). In each method `self` IS the worker, exposing `get_model()` -> the live
nn.Module, so `model.load_weights(...)` uses the SAME in-place loader vLLM runs
at startup (it fuses q/k/v->qkv_proj, gate/up->gate_up_proj, etc).

Two weight-sync transports:
  * disk : `hot_load_from_disk(path)` — load a merged single-file safetensors.
  * nccl : `init_nccl(...)` then `recv_and_load(chunks)` — receive chunked bf16
           broadcasts from the trainer (rank 0) and load_weights once per chunk.

This module imports nothing heavy at top level; each method imports what it needs
lazily so it is cheap to import in any env that only needs the dotted path.
"""
from __future__ import annotations


def _prod(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


class WorkerExt:
    def hot_load_from_disk(self, path):
        from safetensors import safe_open
        model = self.get_model()

        def gen():
            with safe_open(path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    yield k, f.get_tensor(k)

        loaded = model.load_weights(gen())
        try:
            return len(loaded)
        except TypeError:
            return -1

    def init_nccl(self, host, port, rank, world_size):
        """Join the trainer's standalone NCCL group (rank 1, consumer)."""
        from train.rl_vllm.nccl_transport import NcclGroup
        import torch
        dev = next(self.get_model().parameters()).device
        if dev.type != "cuda":
            dev = torch.device("cuda:0")
        self._nccl = NcclGroup(host, int(port), int(rank), int(world_size), dev)
        return True

    def recv_and_load(self, chunks):
        """Receive CHUNKED broadcasts (one big buffer per chunk, far fewer NCCL ops
        than per-tensor), slice each chunk into its tensors, and load_weights once
        per chunk. `chunks` = list of [ [name, shape], ... ]; tensors are bf16 and
        broadcast in chunk-then-name order, matching the trainer's send order."""
        import torch
        model = self.get_model()
        dev = self._nccl.device
        n = 0
        for chunk in chunks:
            numel = sum(_prod(shape) for _, shape in chunk)
            buf = torch.empty(numel, dtype=torch.bfloat16, device=dev)
            self._nccl.broadcast(buf, src=0)
            weights, off = [], 0
            for name, shape in chunk:
                k = _prod(shape)
                weights.append((name, buf[off:off + k].view(*shape)))
                off += k
            model.load_weights(weights)
            del buf
            n += len(chunk)
        return n

    def probe_param(self, name):
        d = dict(self.get_model().named_parameters())
        p = d[name].detach()
        return {"shape": list(p.shape), "norm": float(p.float().norm()),
                "row0_head": [round(x, 5) for x in p[0, :6].float().tolist()]}

    def compare_to_file(self, path):
        """Decisive weight-level check: is EVERY live param bit-equal to the merged
        file (reconstructing vLLM's qkv/gate_up fusion)? max_abs_diff~0 => the
        hot-load made the engine identical to a fresh load, independent of the
        chaotic greedy decode."""
        import torch
        from safetensors import safe_open
        live = dict(self.get_model().named_parameters())
        max_abs, n, nmis, worst = 0.0, 0, 0, None
        with safe_open(path, framework="pt", device="cpu") as f:
            have = set(f.keys())

            def cat(*names):
                return torch.cat([f.get_tensor(x).float() for x in names], dim=0)

            for name, param in live.items():
                p = param.detach().float().cpu()
                if name.endswith("qkv_proj.weight") or name.endswith("qkv_proj.bias"):
                    pre, suf = name[:name.index("qkv_proj")], name.rsplit(".", 1)[1]
                    exp = cat(f"{pre}q_proj.{suf}", f"{pre}k_proj.{suf}", f"{pre}v_proj.{suf}")
                elif name.endswith("gate_up_proj.weight"):
                    pre = name[:name.index("gate_up_proj")]
                    exp = cat(f"{pre}gate_proj.weight", f"{pre}up_proj.weight")
                elif name in have:
                    exp = f.get_tensor(name).float()
                else:
                    continue
                d = (p - exp).abs().max().item()
                n += 1
                if d > max_abs:
                    max_abs, worst = d, name
                if d > 1e-3:
                    nmis += 1
        return {"n_params_checked": n, "max_abs_diff": max_abs,
                "worst_param": worst, "n_mismatch_gt_1e-3": nmis}
