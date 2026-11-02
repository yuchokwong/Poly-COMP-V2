from __future__ import annotations

import os

from tsr.runtime import configure_threads, cpu_budget

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def test_cpu_budget_is_positive() -> None:
    assert cpu_budget() >= 1


def test_configure_threads_respects_explicit_value(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert configure_threads() == 3


def test_configure_threads_fills_defaults(monkeypatch) -> None:
    for key in _THREAD_VARS:
        monkeypatch.delenv(key, raising=False)

    value = configure_threads()

    assert value >= 1
    for key in _THREAD_VARS:
        assert os.environ[key] == str(value)
