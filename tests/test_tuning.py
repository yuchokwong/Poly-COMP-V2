"""Parameter-sweep definitions, aggregation and isolation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tsr.tuning import (
    ALL_RUNS,
    BASELINE_RUNS,
    GROUPS,
    SUMMARY_SCHEMA_VERSION,
    build_summary,
    compare_baseline,
    diff_snapshots,
    env_overrides_for,
    extract_run,
    flatten_overrides,
    snapshot_paths,
    write_summary,
)

SPECS = {spec.id: spec for spec in ALL_RUNS}


def fake_report(mAP: float, **overrides):
    report = {
        "ready": True,
        "eval_set": "memecap_eval_v1",
        "metrics": {
            "mAP": mAP,
            "MRR": 0.5,
            "ndcg@10": 0.5,
            "precision@1": 0.4,
            "recall@10": 0.8,
            "median_rank": 3,
            "misses": 0,
            "num_queries": 240,
        },
        "ci": {"mAP": {"mean": mAP, "low": mAP - 0.01, "high": mAP + 0.01}},
        "per_category": {"visual": {"num_queries": 10, "mAP": mAP, "MRR": 0.5}},
        "latency": {"p50_ms": 12.5, "p95_ms": 30.0},
        "num_errors": 0,
        "per_query_values": {"mAP": [0.1, 0.2]},
    }
    report.update(overrides)
    return report


def run_payload(run_id: str, mAP: float, **overrides):
    spec = SPECS[run_id]
    return extract_run(
        spec,
        fake_report(mAP, **overrides),
        report_path=Path("tuning") / "runs" / run_id / f"eval_{spec.strategy}.json",
    )


def sample_runs() -> list[dict]:
    return [
        run_payload("baseline_clip", 0.44),
        run_payload("baseline_clip_nlp", 0.76),
        run_payload("baseline_vlm", 0.45),
        run_payload("b_weights_clip_0.2", 0.77),
        run_payload("a_prompt_ensemble", 0.46),
    ]


def test_run_and_group_definitions_are_consistent():
    ids = [spec.id for spec in ALL_RUNS]
    assert len(ids) == len(set(ids))
    assert {spec.strategy for spec in ALL_RUNS} == {"clip", "clip_nlp", "vlm"}
    assert {spec.id for spec in BASELINE_RUNS} == {
        "baseline_clip",
        "baseline_clip_nlp",
        "baseline_vlm",
    }

    group_ids = {spec.group for spec in ALL_RUNS if not spec.is_baseline}
    assert {group.id for group in GROUPS} == group_ids
    for group in GROUPS:
        assert group.baseline_run_id in SPECS
        assert SPECS[group.baseline_run_id].strategy == group.strategy
        assert SPECS[group.baseline_run_id].is_baseline


def test_clip_weight_sweep_keeps_the_weights_normalised():
    for spec in ALL_RUNS:
        if spec.group != "b_weights_clip":
            continue
        weights = spec.overrides["strategies"]["clip_nlp"]["weights"]
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["clip"] == pytest.approx(spec.value)
        # doc:bm25 stays at 60:40 while the CLIP share moves.
        if weights["doc"] + weights["bm25"] > 0:
            assert weights["doc"] / (weights["doc"] + weights["bm25"]) == pytest.approx(0.6)


def test_flatten_overrides_builds_tsr_environment_names():
    flat = flatten_overrides(
        {
            "strategies": {"clip_nlp": {"weights": {"clip": 0.2}}},
            "app": {"preload_models": True},
        }
    )
    assert flat == {
        "TSR_STRATEGIES__CLIP_NLP__WEIGHTS__CLIP": "0.2",
        "TSR_APP__PRELOAD_MODELS": "true",
    }

    env = env_overrides_for(SPECS["baseline_clip"])
    assert env == {"TSR_PATHS__REPORTS_DIR": "tuning/runs/baseline_clip"}


def test_extract_run_drops_per_query_payloads():
    run = run_payload("b_weights_clip_0.2", 0.77)
    assert run["ready"] is True
    assert "per_query_values" not in run
    assert run["metrics"]["num_queries"] == 240
    assert run["eval_set"] == "memecap_eval_v1"
    assert run["env_overrides"]["TSR_PATHS__REPORTS_DIR"] == (
        "tuning/runs/b_weights_clip_0.2"
    )


def test_build_summary_computes_deltas_groups_and_failures():
    runs = sample_runs()
    failed = run_payload("c_clip_weight_0.35", 0.0, ready=False, error="boom")
    runs.append(failed)

    summary = build_summary(
        runs,
        eval_set="memecap_eval_v1",
        bootstrap=1000,
        seed=42,
    )
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["num_queries"] == 240
    assert set(summary["baselines"]) == {"clip", "clip_nlp", "vlm"}
    assert summary["failures"] == [{"id": "c_clip_weight_0.35", "error": "boom"}]

    by_id = summary["by_run"]
    delta = by_id["b_weights_clip_0.2"]["delta_vs_baseline"]
    assert delta["mAP"] == pytest.approx(0.01)
    assert by_id["baseline_clip_nlp"]["delta_vs_baseline"] is None

    groups = {group["id"]: group for group in summary["groups"]}
    xs = [point["x"] for point in groups["b_weights_clip"]["points"]]
    assert xs == [0.2, 0.5]  # numeric order, default injected at its value
    assert groups["b_weights_clip"]["points"][1]["is_baseline"] is True

    boolean_points = groups["a_prompt_ensemble"]["points"]
    assert [point["x"] for point in boolean_points] == [False, True]
    assert boolean_points[0]["run_id"] == "baseline_clip"


def test_write_summary_strips_the_lookup_table(tmp_path):
    summary = build_summary(
        sample_runs(), eval_set="memecap_eval_v1", bootstrap=1000, seed=42
    )
    path = write_summary(tmp_path / "data" / "tuning" / "summary.json", summary)
    payload = path.read_text(encoding="utf-8")
    assert '"by_run"' not in payload
    assert '"schema_version": 1' in payload
    assert not (tmp_path / "data" / "tuning" / "summary.json.tmp").exists()


def test_compare_baseline_matches_and_reports_drift():
    reference = fake_report(0.76)
    match = compare_baseline(fake_report(0.76), reference)
    assert match["status"] == "match"
    assert match["max_abs_delta"] == 0

    drift = compare_baseline(fake_report(0.77), reference)
    assert drift["status"] == "mismatch"
    assert any(item["path"] == "metrics.mAP" for item in drift["mismatches"])

    assert compare_baseline(None, reference)["status"] == "missing"


def test_snapshot_diff_detects_changes_and_ignores_wal_sidecars(tmp_path):
    database = tmp_path / "gallery.db"
    database.write_text("one", encoding="utf-8")
    sidecar = tmp_path / "gallery.db-shm"
    sidecar.write_text("one", encoding="utf-8")

    before = snapshot_paths([database, sidecar])
    sidecar.write_text("two", encoding="utf-8")
    after_sidecar = snapshot_paths([database, sidecar])
    diff = diff_snapshots(before, after_sidecar)
    assert diff["unchanged"] is True
    assert diff["changes"] == []
    assert list(diff["volatile"]) == [str(sidecar)]

    database.write_text("two", encoding="utf-8")
    diff = diff_snapshots(before, snapshot_paths([database, sidecar]))
    assert diff["unchanged"] is False
    assert [item["reason"] for item in diff["changes"]] == ["changed"]
