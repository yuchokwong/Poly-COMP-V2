"""Score normalisation and fusion shared by strategies B and C."""

from __future__ import annotations

import math
from collections.abc import Mapping


def minmax(scores: Mapping[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {key: 1.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def zscore(scores: Mapping[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-12:
        return {key: 0.0 for key in scores}
    return {key: (value - mean) / std for key, value in scores.items()}


def normalize(
    scores: Mapping[int, float],
    method: str = "minmax",
    *,
    candidates: list[int] | None = None,
) -> dict[int, float]:
    """Normalise one signal; missing candidates become 0."""
    if method == "minmax":
        out = minmax(scores)
    elif method == "zscore":
        out = zscore(scores)
    elif method == "none":
        return {key: float(value) for key, value in scores.items()}
    else:
        raise ValueError(f"unknown fusion method {method!r}")
    if candidates is not None:
        return {candidate: out.get(candidate, 0.0) for candidate in candidates}
    return out


def fuse(
    signals: Mapping[str, Mapping[int, float]],
    weights: Mapping[str, float],
    *,
    method: str = "minmax",
) -> dict[int, float]:
    """Weighted sum of normalised signals over their union.

    Missing modalities get 0. Weights of disabled/empty signals are dropped and
    the remaining weights renormalised. ``minmax`` and ``zscore`` return a final
    score in [0, 1]; ``rrf`` fuses rank positions instead of score magnitudes.
    """
    active_signals = {
        name: scores
        for name, scores in signals.items()
        if weights.get(name, 0.0) > 0 and scores
    }
    if not active_signals:
        return {}
    if method == "rrf":
        ranked = {
            name: [pk for pk, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
            for name, scores in active_signals.items()
        }
        return minmax(rrf(ranked, weights=weights))
    if method not in {"minmax", "zscore"}:
        raise ValueError(f"unknown fusion method {method!r}")
    candidates: set[int] = set()
    for scores in active_signals.values():
        candidates.update(scores)
    order = sorted(candidates)
    normalized = {
        name: normalize(scores, method, candidates=order)
        for name, scores in active_signals.items()
    }
    active = {name: float(weights[name]) for name in normalized}
    total = sum(active.values())
    if total <= 0:
        return {candidate: 0.0 for candidate in order}
    fused: dict[int, float] = {}
    for candidate in order:
        fused[candidate] = sum(
            (weight / total) * normalized[name].get(candidate, 0.0)
            for name, weight in active.items()
        )
    if method == "zscore":
        lo, hi = min(fused.values()), max(fused.values())
        if hi - lo < 1e-12:
            return {candidate: 0.5 for candidate in fused}
        return {candidate: (value - lo) / (hi - lo) for candidate, value in fused.items()}
    return fused


def rrf(
    ranked_lists: Mapping[str, list[int]],
    weights: Mapping[str, float] | None = None,
    k: int = 60,
) -> dict[int, float]:
    weights = weights or {}
    out: dict[int, float] = {}
    for name, ranked in ranked_lists.items():
        weight = float(weights.get(name, 1.0))
        for rank, image_pk in enumerate(ranked, start=1):
            out[image_pk] = out.get(image_pk, 0.0) + weight / (k + rank)
    return out
