"""Process-level runtime defaults that must be applied before torch is imported."""

from __future__ import annotations

import math
import os
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def cpu_budget() -> int:
    """CPUs this process may actually use: cgroup quota first, then affinity.

    ``os.cpu_count()`` and ``nproc`` report the host's cores inside a container,
    which makes torch spawn more OpenMP threads than the CPU quota allows.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass

    quota = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota is not None and quota > 0 and period:
        return max(1, math.ceil(quota / period))

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # Windows
        return max(1, os.cpu_count() or 1)


def configure_threads() -> int:
    """Pin OpenMP/MKL/OpenBLAS to the container budget unless already set.

    Call this before torch is imported: OpenMP reads the environment when the
    library loads, so a later assignment has no effect.
    """
    budget = str(cpu_budget())
    for key in _THREAD_VARS:
        os.environ.setdefault(key, budget)
    try:
        return max(1, int(os.environ["OMP_NUM_THREADS"]))
    except ValueError:
        return cpu_budget()
