"""Lightweight, env-gated phase profiler for the RL training step.

Enable with `RL_PROFILE=1` (any value other than 0/""/false). When disabled the
`section(...)` context manager is a near-zero-overhead no-op so this can stay
wired into the hot path permanently.

Why a custom timer instead of torch.profiler: we want a coarse per-step
wall-clock breakdown of the *pipeline phases* (generation vs vocode/ASR/CLSP
scoring vs ref-logprob forward vs policy-loss forward vs backward) — not a
kernel trace. Each timed block does a `torch.cuda.synchronize()` on entry and
exit so async CUDA work is attributed to the phase that launched it; otherwise
generation (which kicks off a long async stream) would look free and the next
sync point would absorb its cost.

Usage:
    from .profiling import PROF
    with PROF.section("rollout"):
        ...
    # at the end of a micro-step:
    PROF.report_step(global_step)   # prints "[prof] step N ..." and resets
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import torch

ENABLED = os.environ.get("RL_PROFILE", "0").lower() not in ("0", "", "false", "no")

# Key used to stash the full micro-step wall time; everything else is a phase.
_TOTAL_KEY = "_step_total"


def _sync_all():
    """Synchronize every visible CUDA device. The policy runs on cuda:0 (and
    possibly DataParallel cuda:1) while the scorer lives on cuda:1, so syncing
    only the current device would misattribute cross-device async work."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


class _Profiler:
    def __init__(self):
        self._acc: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._micro = 0  # micro-steps seen since last report

    @contextmanager
    def section(self, name: str, sync: bool = True):
        if not ENABLED:
            yield
            return
        if sync:
            _sync_all()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                _sync_all()
            dt = time.perf_counter() - t0
            self._acc[name] = self._acc.get(name, 0.0) + dt
            self._calls[name] = self._calls.get(name, 0) + 1

    def add(self, name: str, dt: float):
        if not ENABLED:
            return
        self._acc[name] = self._acc.get(name, 0.0) + dt
        self._calls[name] = self._calls.get(name, 0) + 1

    def mark_micro(self):
        """Count one completed micro-step (forward+backward) since last report."""
        if ENABLED:
            self._micro += 1

    def report_step(self, global_step: int):
        """Print the accumulated breakdown and reset. Call once per optimizer step.

        Phases are listed largest-first with their share of the measured total.
        `rest` = step_total - sum(phases) captures backward/optimizer/overhead
        not wrapped in an explicit section.
        """
        if not ENABLED or not self._acc:
            return
        phases = {k: v for k, v in self._acc.items() if k != _TOTAL_KEY}
        total = self._acc.get(_TOTAL_KEY, sum(phases.values()))
        rest = max(0.0, total - sum(phases.values()))
        items = sorted(phases.items(), key=lambda kv: kv[1], reverse=True)
        if rest > 1e-6:
            items.append(("backward/optim/rest", rest))
        micro = max(1, self._micro)
        parts = " | ".join(
            f"{k} {v:.2f}s ({100 * v / total:.0f}%, {v / micro * 1000:.0f}ms/micro)"
            for k, v in items
        )
        print(
            f"[prof] opt-step {global_step} | {micro} micro-steps | "
            f"total {total:.2f}s ({total / micro:.2f}s/micro) :: {parts}",
            flush=True,
        )
        self._acc.clear()
        self._calls.clear()
        self._micro = 0


PROF = _Profiler()
