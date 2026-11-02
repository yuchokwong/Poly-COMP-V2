from __future__ import annotations

import threading
import time

from tsr.nlp.rerank import CrossEncoderReranker, _length_batches


def test_length_batches_group_similar_lengths() -> None:
    assert _length_batches([50, 3, 20, 4], batch_size=2) == [[1, 3], [2, 0]]


def test_load_is_serialised_across_concurrent_callers(monkeypatch) -> None:
    """Preload and the first reranked query must share one model copy."""
    reranker = CrossEncoderReranker("fake")
    calls = []
    lock = threading.Lock()

    def fake_load() -> None:
        with lock:
            calls.append(1)
        time.sleep(0.05)
        reranker._model = object()

    monkeypatch.setattr(reranker, "_load", fake_load)
    barrier = threading.Barrier(4)

    def worker() -> None:
        barrier.wait()
        reranker.load()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert len(calls) == 1
