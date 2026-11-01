from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tsr.compare import compare_strategies, evaluate_strategy, failure_cases, write_report
from tsr.evaluate import EvalQuery
from tsr.strategies.base import Hit, SearchOutput


class FakeStore:
    def __init__(self) -> None:
        self.records = {
            1: SimpleNamespace(id=1, image_id="memes_a", path="memecap/a.jpg"),
            2: SimpleNamespace(id=2, image_id="memes_b", path="memecap/b.jpg"),
        }

    def get(self, pk: int):
        return self.records.get(pk)

    def iter_all(self):
        return list(self.records.values())


class FakeStrategy:
    key = "clip"
    title = "Fake CLIP"
    ready = True
    load_error = None

    def __init__(self) -> None:
        self.store = FakeStore()
        self.vector = np.zeros((1, 4), dtype=np.float32)

    def ensure_loaded(self) -> None:
        pass

    def search(self, query: str, k: int) -> SearchOutput:
        return SearchOutput(
            query=query,
            strategy=self.key,
            hits=[Hit(image_pk=1, score=0.9, components={"clip": 0.9})],
            latency_ms=1.0,
        )


def _query() -> EvalQuery:
    return EvalQuery(
        qid="q1",
        query="a dog in a burning room",
        relevant=["memes_a"],
        category="visual",
        grades={"memes_a": 2},
    )


def test_evaluate_strategy_maps_hit_pks_to_qrel_image_ids() -> None:
    result = evaluate_strategy(
        "clip", FakeStrategy(), [_query()], top_k=(1, 5), n_boot=50
    )

    assert result["metrics"]["mAP"] == 1.0
    assert result["metrics"]["MRR"] == 1.0
    assert result["per_query"][0]["retrieved"] == ["memes_a"]


def test_failure_cases_enrich_relevant_and_retrieved_with_pks() -> None:
    strategy = FakeStrategy()
    result = evaluate_strategy("clip", strategy, [_query()], top_k=(1, 5), n_boot=50)
    failures = failure_cases({"strategies": {"clip": result}}, [_query()], strategy.store)

    row = failures["clip"][0]
    assert row["relevant"][0]["image_id"] == "memes_a"
    assert row["relevant"][0]["pk"] == 1
    assert row["retrieved"][0]["pk"] == 1
    assert row["retrieved"][0]["path"] == "memecap/a.jpg"


def test_write_report_writes_failure_gallery(tmp_path) -> None:
    strategy = FakeStrategy()
    result = evaluate_strategy("clip", strategy, [_query()], top_k=(1, 5), n_boot=50)
    report = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "eval_set": "test",
        "num_queries": 1,
        "n_boot": 50,
        "seed": 1,
        "strategies": {"clip": result},
        "pairwise": [],
        "ablations": {},
        "probe": {"num_images": 2, "leakage_passed": 1, "leakage_flagged": 0},
        "failures": failure_cases({"strategies": {"clip": result}}, [_query()], strategy.store),
    }

    json_path, html_path, failures_path = write_report(report, tmp_path)

    assert json_path.exists()
    assert html_path.exists()
    assert failures_path.exists()
    assert "Failure analysis" in failures_path.read_text(encoding="utf-8")


def test_diagnostics_read_separately_from_the_main_table(tmp_path) -> None:
    import json

    from tsr.compare import load_diagnostics, render_html
    from tsr.config import AttrDict

    cfg = AttrDict(
        {"paths": AttrDict({"data_dir": str(tmp_path), "reports_dir": "reports"})}
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "eval_clip_nlp_raw_captions.json").write_text(
        json.dumps(
            {
                "ready": True,
                "eval_set": "raw_captions",
                "metrics": {"mAP": 0.83, "MRR": 0.93, "ndcg@10": 0.88, "num_queries": 240},
            }
        ),
        encoding="utf-8",
    )
    diagnostics = load_diagnostics(cfg, ["clip", "clip_nlp"])
    assert set(diagnostics) == {"clip_nlp"}
    assert diagnostics["clip_nlp"]["mAP"] == pytest.approx(0.83)

    report = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "eval_set": "reviewed",
        "num_queries": 240,
        "strategies": {"clip_nlp": {"ready": True, "metrics": {"mAP": 0.67}}},
        "pairwise": [],
        "ablations": {},
        "diagnostics": diagnostics,
        "probe": {},
    }
    page = render_html(report)
    assert "raw captions as queries" in page
    assert "+0.1600" in page


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_paired_bootstrap_handles_identical_scores(value: float) -> None:
    from tsr.evaluate import paired_bootstrap_test

    delta, low, high, p = paired_bootstrap_test(
        [value] * 8, [value] * 8, n_boot=100, seed=1
    )
    assert (delta, low, high, p) == (0.0, 0.0, 0.0, 1.0)


def test_pairwise_holm_corrects_across_all_pairs() -> None:
    """Three strategies produce three pairs; they share one Holm family."""
    from tsr.compare import _pairwise

    def result(values):
        return {"ready": True, "per_query_values": values}

    n = 8
    perfect = [1.0] * n
    decent = [0.5] * n
    poor = [0.0] * n
    results = {
        "clip": result({"mAP": poor}),
        "clip_nlp": result({"mAP": perfect}),
        "vlm": result({"mAP": decent}),
    }
    pairs = _pairwise(results, n_boot=200, seed=0)
    assert len(pairs) == 3

    rows = [row for pair in pairs for row in pair["comparisons"]]
    assert len(rows) == 3
    # The smallest raw p-value must be multiplied by the family size (3).
    best = min(rows, key=lambda row: row["p_raw"])
    assert best["p_holm"] >= best["p_raw"]
    assert all(row["p_holm"] >= row["p_raw"] - 1e-9 for row in rows)
    # Every row records its pair so the HTML can render it.
    assert all(row["a"] and row["b"] for row in rows)


def test_compare_refuses_to_publish_a_diagnostic_query_set(cfg):
    """Raw-caption queries are leaky by design and must never become the main table."""
    diagnostic = EvalQuery(
        "q1",
        "a cat at dinner",
        ["memes_a"],
        category="visual",
        grades={"memes_a": 2},
        metadata={
            "query_origin": "raw_caption_diagnostic",
            "leakage": {"passed": False},
        },
    )
    with pytest.raises(ValueError, match="diagnostic"):
        compare_strategies(cfg, {}, [diagnostic], write=False)
