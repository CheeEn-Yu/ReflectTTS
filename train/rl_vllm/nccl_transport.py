"""Standalone NCCL broadcast group shared by the trainer (step env) and the vLLM
sidecar worker (vllm env). Both have torch 2.11+cu130 (ABI-compatible), so a
plain torch `ProcessGroupNCCL` built on a shared `TCPStore` works across the two
processes/envs. We do NOT call init_process_group (which would clash with vLLM's
own global group) — we construct an independent PG object bound to its own store.

Used to push merged LoRA weights GPU->GPU (NVLink, sub-second) instead of the v0
disk round-trip. Trainer = rank 0 (producer), vLLM worker = rank 1 (consumer).
"""
from __future__ import annotations

import argparse
from datetime import timedelta

import torch
import torch.distributed as dist

try:  # BroadcastOptions location varies; both work in torch 2.11
    from torch._C._distributed_c10d import BroadcastOptions
except Exception:  # pragma: no cover
    from torch.distributed import BroadcastOptions  # type: ignore


class NcclGroup:
    def __init__(self, host: str, port: int, rank: int, world_size: int,
                 device, timeout_s: int = 1800):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.store = dist.TCPStore(host, int(port), world_size, rank == 0,
                                   timedelta(seconds=timeout_s))
        opts = dist.ProcessGroupNCCL.Options()
        self.pg = dist.ProcessGroupNCCL(self.store, rank, world_size, opts)
        # force the NCCL comm to build now (so later broadcasts can't deadlock mid-train)
        warm = torch.zeros(1, device=self.device)
        self._bcast(warm, src=0)
        torch.cuda.synchronize(self.device)

    def _bcast(self, tensor: torch.Tensor, src: int = 0):
        opt = BroadcastOptions()
        opt.rootRank = src
        opt.rootTensor = 0
        self.pg.broadcast([tensor], opt).wait()

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        self._bcast(tensor.contiguous(), src=src)


def _selftest():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world_size", type=int, default=2)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29555)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    g = NcclGroup(args.host, args.port, args.rank, args.world_size, args.device)
    print(f"[nccl rank{args.rank}] group up on {args.device}", flush=True)
    # rank0 sends a known pattern; rank1 verifies
    n = 1024
    if args.rank == 0:
        t = torch.arange(n, dtype=torch.float32, device=args.device) * 0.5
    else:
        t = torch.zeros(n, dtype=torch.float32, device=args.device)
    g.broadcast(t, src=0)
    torch.cuda.synchronize(g.device)
    expected = torch.arange(n, dtype=torch.float32, device=args.device) * 0.5
    ok = bool(torch.equal(t, expected))
    print(f"[nccl rank{args.rank}] broadcast match={ok} sum={float(t.sum()):.1f}", flush=True)
    print(f"[nccl rank{args.rank}] SELFTEST {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
