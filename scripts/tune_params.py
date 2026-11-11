#!/usr/bin/env python
"""Run the read-only parameter sweep and write data/tuning/summary.json.

The sweep reuses `python -m tsr eval`; every run gets its own
``TSR_PATHS__REPORTS_DIR`` under ``data/tuning/runs/`` so the main
``data/reports/`` tree is never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsr.config import (  # noqa: E402
    db_path,
    eval_dir,
    index_root,
    load_config,
    reports_dir,
    settings_path,
    tuning_dir,
)
from tsr.tuning import (  # noqa: E402
    ALL_RUNS,
    BASELINE_RUNS,
    RunSpec,
    build_summary,
    compact_diff,
    compare_baseline,
    diff_snapshots,
    env_overrides_for,
    extract_run,
    run_report_path,
    snapshot_paths,
    write_summary,
)

INDEX_FILES = {
    "clip": "clip/image.faiss",
    "clip_nlp": "clip_nlp/doc.faiss",
    "vlm": "vlm/doc.faiss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list run ids and exit")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated run ids or group ids (baselines are added automatically)",
    )
    parser.add_argument("--force", action="store_true", help="delete and rerun selected runs")
    parser.add_argument("--no-resume", action="store_true", help="rerun finished runs")
    parser.add_argument(
        "--summary-only", action="store_true", help="aggregate existing reports only"
    )
    parser.add_argument(
        "--strict-baseline",
        action="store_true",
        help="stop before the sweep when a baseline does not reproduce data/reports",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="per-run timeout (s)")
    parser.add_argument("--no-integrity", action="store_true", help="skip hash snapshots")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spec_hash(spec: RunSpec) -> str:
    payload = json.dumps(
        {"strategy": spec.strategy, "overrides": spec.overrides}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_env(spec: RunSpec) -> dict[str, str]:
    env = os.environ.copy()
    source = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([source, existing]) if existing else source
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(env_overrides_for(spec))
    return env


def _command(spec: RunSpec) -> list[str]:
    return [sys.executable, "-m", "tsr", "eval", "--strategy", spec.strategy]


def _select_specs(only: str | None) -> list[RunSpec]:
    if not only:
        return list(ALL_RUNS)
    wanted = {token.strip() for token in only.split(",") if token.strip()}
    selected = [spec for spec in ALL_RUNS if spec.id in wanted or spec.group in wanted]
    strategies = {spec.strategy for spec in selected if not spec.is_baseline}
    chosen = {spec.id for spec in selected}
    for spec in BASELINE_RUNS:
        if spec.strategy in strategies:
            chosen.add(spec.id)
    return [spec for spec in ALL_RUNS if spec.id in chosen]


def _check_inputs(cfg, specs: list[RunSpec]) -> None:
    if os.environ.get("TSR_PATHS__REPORTS_DIR"):
        raise SystemExit(
            "Unset TSR_PATHS__REPORTS_DIR before running the sweep; the runner sets it "
            "per run so the main reports tree stays untouched."
        )
    database = db_path(cfg)
    if not database.exists():
        raise SystemExit(f"Gallery database not found: {database}")
    eval_file = eval_dir(cfg) / f"{cfg.eval.get('set_name', 'eval')}.json"
    if not eval_file.exists():
        raise SystemExit(f"Evaluation set not found: {eval_file}")

    needed = {spec.strategy for spec in specs}
    if "clip_nlp" in needed:
        needed.add("clip")
    for strategy in sorted(needed):
        index_file = index_root(cfg) / INDEX_FILES[strategy]
        if not index_file.exists():
            raise SystemExit(
                f"Index for strategy {strategy!r} not found: {index_file}. "
                "Build the indexes before running the sweep."
            )

    tuning_root = tuning_dir(cfg).resolve()
    reports = reports_dir(cfg).resolve()
    if tuning_root == reports or tuning_root in reports.parents:
        raise SystemExit(
            "The tuning tree resolves inside the main reports tree; refusing to write."
        )

    first = specs[0]
    code = "from tsr.config import load_config, reports_dir; print(reports_dir(load_config()))"
    probe = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_build_env(first),
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(f"Preflight failed:\n{probe.stderr or probe.stdout}")
    child_reports = Path(probe.stdout.strip()).resolve()
    if not child_reports.is_relative_to(tuning_root):
        raise SystemExit(
            f"Preflight resolved reports to {child_reports}, outside {tuning_root}."
        )
    if child_reports == reports:
        raise SystemExit("Preflight resolved reports to the main reports tree; aborting.")


def _is_complete(spec: RunSpec, tuning_root: Path) -> bool:
    run_dir = tuning_root / "runs" / spec.id
    meta_path = run_dir / "run.json"
    report_path = run_report_path(tuning_root, spec)
    meta = _read_json(meta_path)
    report = _read_json(report_path)
    return bool(
        meta
        and report
        and meta.get("status") == "ok"
        and meta.get("overrides_sha256") == _spec_hash(spec)
        and report.get("ready")
    )


def _run_one(
    spec: RunSpec,
    args: argparse.Namespace,
    tuning_root: Path,
    *,
    resume: bool,
) -> None:
    run_dir = tuning_root / "runs" / spec.id
    report_path = run_report_path(tuning_root, spec)
    meta_path = run_dir / "run.json"

    if resume and not args.force and _is_complete(spec, tuning_root):
        print(f"[skip] {spec.id} (already complete)", flush=True)
        return
    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = tuning_root / "logs" / f"{spec.id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    start = time.monotonic()
    returncode: int | None = None
    error: str | None = None

    print(f"[run ] {spec.id} ({spec.family}, {spec.strategy}) ...", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                _command(spec),
                cwd=ROOT,
                env=_build_env(spec),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            error = f"timeout after {args.timeout}s"
        except OSError as exc:  # pragma: no cover - environment failure
            error = str(exc)

    duration_s = round(time.monotonic() - start, 2)
    report = _read_json(report_path)
    ready = bool(report and report.get("ready"))
    status = "ok" if returncode == 0 and ready and error is None else "failed"
    if not ready and error is None:
        error = (report or {}).get("error") or "report missing or not ready"

    meta = {
        "id": spec.id,
        "status": status,
        "returncode": returncode,
        "error": error,
        "overrides": spec.overrides,
        "overrides_sha256": _spec_hash(spec),
        "env_overrides": env_overrides_for(spec),
        "command": _command(spec),
        "report": str(report_path),
        "report_sha256": _sha256(report_path) if report_path.exists() else None,
        "eval_set": (report or {}).get("eval_set"),
        "num_queries": ((report or {}).get("metrics") or {}).get("num_queries"),
        "started_at": started_at,
        "finished_at": _now(),
        "duration_s": duration_s,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = (report or {}).get("metrics") or {}
    if status == "ok":
        print(
            f"[done] {spec.id} in {duration_s}s  mAP={metrics.get('mAP', 0):.4f}",
            flush=True,
        )
    else:
        print(f"[fail] {spec.id}: {error}", flush=True)


def _check_baselines(cfg, specs: list[RunSpec], tuning_root: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for spec in specs:
        if not spec.is_baseline:
            continue
        reference_path = reports_dir(cfg) / f"eval_{spec.strategy}.json"
        checks[spec.strategy] = compare_baseline(
            _read_json(run_report_path(tuning_root, spec)),
            _read_json(reference_path),
            reference_path=reference_path,
        )
    return checks


def _load_run(
    spec: RunSpec,
    tuning_root: Path,
    eval_set: str,
) -> dict[str, Any]:
    meta = _read_json(tuning_root / "runs" / spec.id / "run.json") or {}
    report = _read_json(run_report_path(tuning_root, spec))
    if report and report.get("eval_set") not in (None, eval_set):
        report = {
            "ready": False,
            "error": f"eval_set {report.get('eval_set')!r} != {eval_set!r}",
        }
    return extract_run(
        spec,
        report,
        report_path=run_report_path(tuning_root, spec),
        duration_s=meta.get("duration_s"),
        started_at=meta.get("started_at"),
        finished_at=meta.get("finished_at"),
    )


def _protected_paths(cfg) -> list[Path]:
    database = db_path(cfg)
    return [
        reports_dir(cfg),
        settings_path(cfg),
        database,
        database.with_name(database.name + "-wal"),
        database.with_name(database.name + "-shm"),
        index_root(cfg),
        eval_dir(cfg),
        Path(cfg.root) / "configs" / "default.yaml",
    ]


def _tool_info() -> dict[str, Any]:
    commit = ""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = completed.stdout.strip()
    except OSError:  # pragma: no cover - git not on PATH
        pass
    return {
        "name": "tsr-tune",
        "git_commit": commit,
        "python": sys.version.split()[0],
    }


def main() -> int:
    args = parse_args()
    if args.list:
        for spec in ALL_RUNS:
            marker = "baseline" if spec.is_baseline else spec.group
            print(f"{spec.id:28s} {spec.strategy:8s} {marker}")
        return 0

    cfg = load_config()
    tuning_root = tuning_dir(cfg)
    summary_path = tuning_root / "summary.json"
    eval_set = str(cfg.eval.get("set_name", "eval"))

    if args.summary_only:
        specs = _select_specs(args.only)
    else:
        specs = _select_specs(args.only)
        if not specs:
            raise SystemExit("No matching run ids or groups.")
        _check_inputs(cfg, specs)
        if args.dry_run:
            for spec in specs:
                print(f"# {spec.id}")
                print(f"cd {ROOT}")
                for key, value in sorted(env_overrides_for(spec).items()):
                    print(f"export {key}={value}")
                print(" ".join(_command(spec)))
                print()
            return 0

    before: dict[str, Any] = {}
    if not args.summary_only and not args.no_integrity:
        before = snapshot_paths(_protected_paths(cfg))

    baseline_specs = [spec for spec in specs if spec.is_baseline]
    sweep_specs = [spec for spec in specs if not spec.is_baseline]
    resume = not args.no_resume and not args.force

    if not args.summary_only:
        for spec in baseline_specs:
            _run_one(spec, args, tuning_root, resume=resume)

        baseline_check = _check_baselines(cfg, specs, tuning_root)
        for strategy, check in baseline_check.items():
            print(
                f"[base] {strategy}: {check['status']} "
                f"(max |Δ|={check.get('max_abs_delta')})",
                flush=True,
            )
        mismatched = [
            strategy for strategy, check in baseline_check.items()
            if check.get("status") == "mismatch"
        ]
        if args.strict_baseline and mismatched:
            print(
                "Baseline mismatch for: " + ", ".join(mismatched)
                + ". Refusing to run the sweep; inspect config/code drift first.",
                flush=True,
            )
            return 1

        for spec in sweep_specs:
            _run_one(spec, args, tuning_root, resume=resume)
    else:
        baseline_check = _check_baselines(cfg, specs, tuning_root)

    after: dict[str, Any] = {}
    integrity: dict[str, Any] = {
        "checked_at": _now(),
        "unchanged": None,
        "protected_paths": [str(path) for path in _protected_paths(cfg)],
        "changes": [],
        "volatile": [],
    }
    if not args.summary_only and not args.no_integrity:
        after = snapshot_paths(_protected_paths(cfg))
        diff = diff_snapshots(before, after)
        compact = compact_diff(diff)
        integrity.update(compact)
        integrity_path = tuning_root / "integrity.json"
        integrity_path.write_text(
            json.dumps(
                {"before": before, "after": after, "diff": diff},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if compact["changes"]:
            print("[warn] protected files changed during the sweep:", flush=True)
            for item in compact["changes"]:
                print(f"       {item['reason']}: {item['path']}", flush=True)

    runs = [_load_run(spec, tuning_root, eval_set) for spec in specs]
    summary = build_summary(
        runs,
        eval_set=eval_set,
        bootstrap=int(cfg.eval.get("bootstrap", 1000)),
        seed=int(cfg.eval.get("seed", 42)),
        baseline_check=baseline_check,
        integrity=integrity,
        tool=_tool_info(),
    )
    write_summary(summary_path, summary)

    ready = sum(1 for run in runs if run.get("ready"))
    print(f"\nWrote {summary_path}")
    print(f"{ready}/{len(runs)} runs ready; {len(summary['failures'])} failed.")
    if summary["failures"]:
        for failure in summary["failures"]:
            print(f"  - {failure['id']}: {failure['error']}")
    if integrity.get("changes"):
        print("Integrity check FAILED; see data/tuning/integrity.json")
        return 1
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
