"""Parameter-sweep definitions and aggregation for the read-only tuning page.

The module is deliberately light: it imports neither torch nor faiss. The sweep
itself runs through the existing ``python -m tsr eval`` entry point with
``TSR_*`` environment overrides, so retrieval code is not changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUMMARY_SCHEMA_VERSION = 1

STRATEGIES = ("clip", "clip_nlp", "vlm")

EXPLORATORY_CAVEAT = (
    "Exploratory parameter tuning on the frozen 240-query test set. Choosing "
    "parameters from these numbers and reporting them as an unbiased result is "
    "test-set tuning. Do not mix these rows into the /evaluation table. The main "
    "evaluation report remains the protocol of record; confirm any chosen setting "
    "on a held-out set."
)

METRICS_META: dict[str, dict[str, Any]] = {
    "mAP": {"label": "mAP", "higher_is_better": True, "digits": 4},
    "MRR": {"label": "MRR", "higher_is_better": True, "digits": 4},
    "ndcg@10": {"label": "nDCG@10", "higher_is_better": True, "digits": 4},
    "precision@1": {"label": "P@1", "higher_is_better": True, "digits": 4},
    "recall@10": {"label": "R@10", "higher_is_better": True, "digits": 4},
    "median_rank": {"label": "median rank", "higher_is_better": False, "digits": 0},
    "misses": {"label": "misses", "higher_is_better": False, "digits": 0},
}

DELTA_METRICS = tuple(METRICS_META)
BASELINE_METRICS = (
    "mAP",
    "MRR",
    "ndcg@10",
    "precision@1",
    "recall@10",
    "median_rank",
    "misses",
)


@dataclass(frozen=True)
class RunSpec:
    id: str
    strategy: str
    group: str
    family: str
    parameter: str | None
    value: float | bool | None
    value_label: str
    x_kind: str
    overrides: dict[str, Any] = field(default_factory=dict)
    is_baseline: bool = False


@dataclass(frozen=True)
class GroupSpec:
    id: str
    strategy: str
    family: str
    parameter: str | None
    x_label: str
    x_kind: str
    kind: str
    baseline_run_id: str
    baseline_x: float | bool


def _b_weights(clip: float) -> dict[str, Any]:
    """Keep the doc:bm25 split at 60:40 while varying the CLIP share."""
    return {
        "strategies": {
            "clip_nlp": {
                "weights": {
                    "clip": round(clip, 4),
                    "doc": round((1.0 - clip) * 0.6, 4),
                    "bm25": round((1.0 - clip) * 0.4, 4),
                }
            }
        }
    }


def _b_ratio(doc: float, bm25: float) -> dict[str, Any]:
    return {
        "strategies": {
            "clip_nlp": {"weights": {"clip": 0.5, "doc": doc, "bm25": bm25}}
        }
    }


BASELINE_RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "baseline_clip", "clip", "baseline", "Baseline", None, None, "default",
        "reference", is_baseline=True,
    ),
    RunSpec(
        "baseline_clip_nlp", "clip_nlp", "baseline", "Baseline", None, None,
        "default", "reference", is_baseline=True,
    ),
    RunSpec(
        "baseline_vlm", "vlm", "baseline", "Baseline", None, None, "default",
        "reference", is_baseline=True,
    ),
)


RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "a_prompt_ensemble", "clip", "a_prompt_ensemble", "A · Prompt ensemble",
        "strategies.clip.prompt_ensemble", True, "ensemble on", "boolean",
        {"strategies": {"clip": {"prompt_ensemble": True}}},
    ),
    RunSpec(
        "b_weights_clip_0.0", "clip_nlp", "b_weights_clip", "B · CLIP weight",
        "strategies.clip_nlp.weights.clip", 0.0, "0.0", "numeric",
        _b_weights(0.0),
    ),
    RunSpec(
        "b_weights_clip_0.1", "clip_nlp", "b_weights_clip", "B · CLIP weight",
        "strategies.clip_nlp.weights.clip", 0.1, "0.1", "numeric",
        _b_weights(0.1),
    ),
    RunSpec(
        "b_weights_clip_0.2", "clip_nlp", "b_weights_clip", "B · CLIP weight",
        "strategies.clip_nlp.weights.clip", 0.2, "0.2", "numeric",
        _b_weights(0.2),
    ),
    RunSpec(
        "b_weights_clip_0.3", "clip_nlp", "b_weights_clip", "B · CLIP weight",
        "strategies.clip_nlp.weights.clip", 0.3, "0.3", "numeric",
        _b_weights(0.3),
    ),
    RunSpec(
        "b_ratio_doc_heavy", "clip_nlp", "b_doc_bm25_ratio", "B · doc / BM25 ratio",
        "doc share (doc / (doc + bm25))", 0.8, "doc 0.4 / bm25 0.1", "numeric",
        _b_ratio(0.4, 0.1),
    ),
    RunSpec(
        "b_ratio_bm25_heavy", "clip_nlp", "b_doc_bm25_ratio", "B · doc / BM25 ratio",
        "doc share (doc / (doc + bm25))", 0.4, "doc 0.2 / bm25 0.3", "numeric",
        _b_ratio(0.2, 0.3),
    ),
    RunSpec(
        "b_rerank_alpha_0.5", "clip_nlp", "b_rerank_alpha", "B · rerank α",
        "strategies.clip_nlp.rerank.alpha", 0.5, "0.5", "numeric",
        {"strategies": {"clip_nlp": {"rerank": {"alpha": 0.5}}}},
    ),
    RunSpec(
        "b_rerank_alpha_0.7", "clip_nlp", "b_rerank_alpha", "B · rerank α",
        "strategies.clip_nlp.rerank.alpha", 0.7, "0.7", "numeric",
        {"strategies": {"clip_nlp": {"rerank": {"alpha": 0.7}}}},
    ),
    RunSpec(
        "b_rerank_topn_20", "clip_nlp", "b_rerank_topn", "B · rerank top_n",
        "strategies.clip_nlp.rerank.top_n", 20.0, "20", "numeric",
        {"strategies": {"clip_nlp": {"rerank": {"top_n": 20}}}},
    ),
    RunSpec(
        "b_rerank_topn_100", "clip_nlp", "b_rerank_topn", "B · rerank top_n",
        "strategies.clip_nlp.rerank.top_n", 100.0, "100", "numeric",
        {"strategies": {"clip_nlp": {"rerank": {"top_n": 100}}}},
    ),
    RunSpec(
        "c_clip_weight_0.0", "vlm", "c_clip_weight", "C · CLIP weight",
        "strategies.vlm.clip_weight", 0.0, "0.0", "numeric",
        {"strategies": {"vlm": {"clip_weight": 0.0}}},
    ),
    RunSpec(
        "c_clip_weight_0.35", "vlm", "c_clip_weight", "C · CLIP weight",
        "strategies.vlm.clip_weight", 0.35, "0.35", "numeric",
        {"strategies": {"vlm": {"clip_weight": 0.35}}},
    ),
    RunSpec(
        "c_clip_weight_0.5", "vlm", "c_clip_weight", "C · CLIP weight",
        "strategies.vlm.clip_weight", 0.5, "0.5", "numeric",
        {"strategies": {"vlm": {"clip_weight": 0.5}}},
    ),
)


GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec(
        "a_prompt_ensemble", "clip", "A · Prompt ensemble",
        "strategies.clip.prompt_ensemble", "prompt ensemble", "boolean", "bar",
        "baseline_clip", False,
    ),
    GroupSpec(
        "b_weights_clip", "clip_nlp", "B · CLIP weight",
        "strategies.clip_nlp.weights.clip", "weights.clip", "numeric", "line",
        "baseline_clip_nlp", 0.5,
    ),
    GroupSpec(
        "b_doc_bm25_ratio", "clip_nlp", "B · doc / BM25 ratio",
        "doc share (doc / (doc + bm25))", "doc share", "numeric", "line",
        "baseline_clip_nlp", 0.6,
    ),
    GroupSpec(
        "b_rerank_alpha", "clip_nlp", "B · rerank α",
        "strategies.clip_nlp.rerank.alpha", "rerank α", "numeric", "line",
        "baseline_clip_nlp", 0.3,
    ),
    GroupSpec(
        "b_rerank_topn", "clip_nlp", "B · rerank top_n",
        "strategies.clip_nlp.rerank.top_n", "top_n", "numeric", "line",
        "baseline_clip_nlp", 60.0,
    ),
    GroupSpec(
        "c_clip_weight", "vlm", "C · CLIP weight",
        "strategies.vlm.clip_weight", "clip_weight", "numeric", "line",
        "baseline_vlm", 0.2,
    ),
)


ALL_RUNS: tuple[RunSpec, ...] = BASELINE_RUNS + RUNS


def flatten_overrides(overrides: dict[str, Any]) -> dict[str, str]:
    """Flatten a nested config patch into ``TSR_*`` environment variables."""
    out: dict[str, str] = {}

    def walk(node: dict[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}__{str(key).upper()}"
            if isinstance(value, dict):
                walk(value, path)
            elif isinstance(value, bool):
                out[path] = "true" if value else "false"
            else:
                out[path] = str(value)

    for key, value in overrides.items():
        path = f"TSR_{str(key).upper()}"
        if isinstance(value, dict):
            walk(value, path)
        elif isinstance(value, bool):
            out[path] = "true" if value else "false"
        else:
            out[path] = str(value)
    return out


def env_overrides_for(spec: RunSpec) -> dict[str, str]:
    return {
        "TSR_PATHS__REPORTS_DIR": f"tuning/runs/{spec.id}",
        **flatten_overrides(spec.overrides),
    }


def run_report_path(tuning_root: Path, spec: RunSpec) -> Path:
    return tuning_root / "runs" / spec.id / f"eval_{spec.strategy}.json"


def extract_run(
    spec: RunSpec,
    report: dict[str, Any] | None,
    *,
    report_path: Path,
    duration_s: float | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    report = report or {}
    metrics = dict(report.get("metrics") or {})
    ready = bool(report.get("ready"))
    return {
        "id": spec.id,
        "strategy": spec.strategy,
        "group": spec.group,
        "family": spec.family,
        "is_baseline": spec.is_baseline,
        "ready": ready,
        "error": report.get("error") or (None if ready else "report missing or not ready"),
        "title": report.get("title"),
        "parameter": spec.parameter,
        "value": spec.value,
        "value_label": spec.value_label,
        "x_kind": spec.x_kind,
        "overrides": spec.overrides,
        "env_overrides": env_overrides_for(spec),
        "report": str(report_path),
        "eval_set": report.get("eval_set"),
        "num_queries": metrics.get("num_queries"),
        "metrics": metrics,
        "ci": report.get("ci") or {},
        "per_category": report.get("per_category") or {},
        "per_split": report.get("per_split") or {},
        "latency": report.get("latency") or {},
        "num_errors": report.get("num_errors", 0),
        "duration_s": duration_s,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def delta_vs_baseline(
    run_metrics: dict[str, Any] | None,
    baseline_metrics: dict[str, Any] | None,
) -> dict[str, float] | None:
    if not run_metrics or not baseline_metrics:
        return None
    deltas: dict[str, float] = {}
    for name in DELTA_METRICS:
        run_value = run_metrics.get(name)
        base_value = baseline_metrics.get(name)
        if isinstance(run_value, (int, float)) and isinstance(base_value, (int, float)):
            deltas[name] = round(float(run_value) - float(base_value), 6)
    return deltas or None


def build_summary(
    runs: list[dict[str, Any]],
    *,
    eval_set: str,
    bootstrap: int,
    seed: int,
    baseline_check: dict[str, Any] | None = None,
    integrity: dict[str, Any] | None = None,
    generated_at: str | None = None,
    tool: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baselines: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run.get("is_baseline"):
            baselines[str(run["strategy"])] = run

    for run in runs:
        baseline = baselines.get(str(run["strategy"]))
        run["delta_vs_baseline"] = (
            None
            if run.get("is_baseline") or not run.get("ready") or baseline is None
            else delta_vs_baseline(run.get("metrics"), baseline.get("metrics"))
        )

    by_id = {str(run["id"]): run for run in runs}
    groups: list[dict[str, Any]] = []
    for group in GROUPS:
        points: list[dict[str, Any]] = []
        baseline = baselines.get(group.strategy)
        if baseline is not None and baseline.get("ready"):
            points.append(
                {
                    "x": group.baseline_x,
                    "label": f"{group.baseline_x} (baseline)",
                    "run_id": baseline["id"],
                    "is_baseline": True,
                }
            )
        for run in runs:
            if run.get("group") != group.id or run.get("is_baseline"):
                continue
            points.append(
                {
                    "x": run.get("value"),
                    "label": run.get("value_label"),
                    "run_id": run["id"],
                    "is_baseline": False,
                }
            )
        points.sort(key=lambda point: float(point["x"]))
        groups.append(
            {
                "id": group.id,
                "strategy": group.strategy,
                "family": group.family,
                "title": group.family,
                "parameter": group.parameter,
                "x_label": group.x_label,
                "x_kind": group.x_kind,
                "kind": group.kind,
                "baseline_run_id": group.baseline_run_id,
                "baseline_x": group.baseline_x,
                "points": points,
            }
        )

    num_queries = next(
        (run["num_queries"] for run in runs if run.get("num_queries")),
        0,
    )
    failures = [
        {"id": run["id"], "error": run.get("error") or "run failed"}
        for run in runs
        if not run.get("ready")
    ]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "eval_set": eval_set,
        "num_queries": num_queries,
        "bootstrap": bootstrap,
        "seed": seed,
        "tool": tool or {},
        "protocol": {
            "exploratory": True,
            "frozen_test_set": True,
            "mixed_eval_sets": False,
            "caveat": EXPLORATORY_CAVEAT,
        },
        "metrics_meta": METRICS_META,
        "runs": runs,
        "baselines": baselines,
        "groups": groups,
        "baseline_check": baseline_check or {},
        "integrity": integrity or {},
        "failures": failures,
        "by_run": by_id,
    }


def compare_baseline(
    sweep_report: dict[str, Any] | None,
    reference_report: dict[str, Any] | None,
    *,
    reference_path: Path | None = None,
    tol: float = 1e-6,
) -> dict[str, Any]:
    reference = str(reference_path) if reference_path is not None else None
    if sweep_report is None or reference_report is None:
        return {
            "status": "missing",
            "reference": reference,
            "max_abs_delta": None,
            "mismatches": [],
        }
    if not sweep_report.get("ready") or not reference_report.get("ready"):
        return {
            "status": "error",
            "reference": reference,
            "max_abs_delta": None,
            "mismatches": [],
        }

    mismatches: list[dict[str, Any]] = []
    max_abs = 0.0

    def check(path: str, sweep_value: Any, reference_value: Any) -> None:
        nonlocal max_abs
        if isinstance(sweep_value, (int, float)) and isinstance(reference_value, (int, float)):
            delta = abs(float(sweep_value) - float(reference_value))
            max_abs = max(max_abs, delta)
            if delta > tol:
                mismatches.append(
                    {
                        "path": path,
                        "sweep": sweep_value,
                        "reference": reference_value,
                        "delta": round(delta, 9),
                    }
                )

    for key in BASELINE_METRICS:
        check(
            f"metrics.{key}",
            (sweep_report.get("metrics") or {}).get(key),
            (reference_report.get("metrics") or {}).get(key),
        )
    for metric, interval in (sweep_report.get("ci") or {}).items():
        reference_interval = (reference_report.get("ci") or {}).get(metric) or {}
        for bound in ("mean", "low", "high"):
            check(
                f"ci.{metric}.{bound}",
                (interval or {}).get(bound),
                reference_interval.get(bound),
            )
    for category, values in (sweep_report.get("per_category") or {}).items():
        reference_values = (reference_report.get("per_category") or {}).get(category) or {}
        for metric in ("mAP", "MRR"):
            check(
                f"per_category.{category}.{metric}",
                (values or {}).get(metric),
                reference_values.get(metric),
            )
    return {
        "status": "mismatch" if mismatches else "match",
        "reference": reference,
        "max_abs_delta": round(max_abs, 9),
        "mismatches": mismatches[:50],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_paths(paths: list[Path] | tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for path in paths:
        item = Path(path)
        if not item.exists():
            continue
        files = [item] if item.is_file() else sorted(p for p in item.rglob("*") if p.is_file())
        for file_path in files:
            stat = file_path.stat()
            snapshot[str(file_path)] = {
                "sha256": _sha256(file_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return snapshot


def diff_snapshots(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    *,
    volatile_suffixes: tuple[str, ...] = ("-shm",),
) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []
    volatile: dict[str, dict[str, Any]] = {}
    for path in sorted(set(before) | set(after)):
        previous = before.get(path)
        current = after.get(path)
        if previous == current:
            continue
        reason = "added" if previous is None else "removed" if current is None else "changed"
        item = {"path": path, "reason": reason, "before": previous, "after": current}
        if any(path.endswith(suffix) for suffix in volatile_suffixes):
            volatile[path] = item
        else:
            changes.append(item)
    return {"unchanged": not changes, "changes": changes, "volatile": volatile}


def compact_diff(diff: dict[str, Any]) -> dict[str, Any]:
    return {
        "unchanged": bool(diff.get("unchanged")),
        "changes": [
            {"path": item["path"], "reason": item["reason"]}
            for item in diff.get("changes", [])
        ],
        "volatile": [
            {"path": item["path"], "reason": item["reason"]}
            for item in (diff.get("volatile") or {}).values()
        ],
    }


def write_summary(path: Path, summary: dict[str, Any]) -> Path:
    payload = {key: value for key, value in summary.items() if key != "by_run"}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path
