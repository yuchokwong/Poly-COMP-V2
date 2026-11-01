import pytest

from tsr.nlp.fusion import fuse, minmax, normalize, rrf


def test_minmax_handles_empty_and_constant():
    assert minmax({}) == {}
    assert minmax({1: 0.4, 2: 0.4}) == {1: 1.0, 2: 1.0}
    assert minmax({1: 0.0, 2: 5.0, 3: 10.0}) == {1: 0.0, 2: 0.5, 3: 1.0}


def test_normalize_fills_missing_candidates_with_zero():
    scores = {1: 1.0, 2: 3.0}
    out = normalize(scores, "minmax", candidates=[2, 3])
    assert out == {2: 1.0, 3: 0.0}
    assert normalize({1: 0.2}, "none") == {1: 0.2}
    with pytest.raises(ValueError):
        normalize({1: 0.2}, "bogus")


def test_fuse_renormalises_weights_and_fills_missing_modalities():
    signals = {"clip": {1: 0.9, 2: 0.8}, "doc": {2: 10.0, 3: 5.0}}
    fused = fuse(signals, {"clip": 0.5, "doc": 0.5})
    # candidates 1,2,3; clip normalised {1:1,2:0,3:0}; doc {2:1,3:0}
    assert fused == {1: 0.5, 2: 0.5, 3: 0.0}

    only_doc = fuse(signals, {"clip": 0.0, "doc": 0.7})
    assert only_doc == {2: 1.0, 3: 0.0}

    no_signals = fuse(signals, {"clip": 0.0, "doc": 0.0})
    assert no_signals == {}
    assert fuse({}, {"clip": 1.0}) == {}


def test_fuse_keeps_scores_in_unit_interval():
    signals = {
        "clip": {1: 30.0, 2: 10.0},
        "doc": {1: 0.1, 2: 0.9},
        "bm25": {1: -4.0, 2: -2.0},
    }
    bounded = fuse(signals, {"clip": 0.5, "doc": 0.3, "bm25": 0.2})
    assert all(0.0 <= value <= 1.0 for value in bounded.values())


def test_rrf_merges_ranked_lists():
    scores = rrf({"a": [1, 2], "b": [2, 3]}, weights={"a": 1.0, "b": 1.0}, k=60)
    assert scores[2] > scores[1]
    assert scores[2] > scores[3]
    assert scores[1] == pytest.approx(1 / 61)


def test_zscore_normalisation_and_fusion_stay_bounded():
    assert normalize({1: 0.0, 2: 1.0}, "zscore") == {1: -1.0, 2: 1.0}
    assert normalize({1: 3.0, 2: 3.0}, "zscore") == {1: 0.0, 2: 0.0}

    fused = fuse(
        {"clip": {1: 0.9, 2: 0.1}, "doc": {1: 0.1, 2: 0.9}},
        {"clip": 0.5, "doc": 0.5},
        method="zscore",
    )
    assert fused == {1: 0.5, 2: 0.5}
    assert all(0.0 <= value <= 1.0 for value in fused.values())


def test_fuse_rrf_uses_ranks_and_normalises_output():
    signals = {"a": {1: 0.9, 2: 0.8, 3: 0.1}, "b": {1: 0.2, 2: 0.9, 3: 0.8}}
    fused = fuse(signals, {"a": 1.0, "b": 1.0}, method="rrf")

    assert set(fused) == {1, 2, 3}
    assert fused[2] == pytest.approx(1.0)  # rank 1 + rank 1 across both lists
    assert all(0.0 <= value <= 1.0 for value in fused.values())
    with pytest.raises(ValueError):
        fuse(signals, {"a": 1.0}, method="bogus")
