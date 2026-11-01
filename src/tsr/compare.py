"""Run the three strategies on one frozen eval set and build comparison reports."""

from __future__ import annotations

import copy
import html
import json
import statistics
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from tsr.artifacts import config_hash
from tsr.config import AttrDict, reports_dir, vlm_doc_filters
from tsr.evaluate import (
    EvalQuery,
    bootstrap_ci,
    evaluate_queries,
    holm_bonferroni,
    paired_bootstrap_test,
    per_query_values,
    win_tie_loss,
)
from tsr.logging_utils import get_logger
from tsr.strategies import get_strategy
from tsr.strategies.base import Strategy, StrategyContext

logger = get_logger(__name__)

COMPARE_METRICS = ("mAP", "MRR", "ndcg@10", "precision@1", "recall@10")

ABLATIONS: dict[str, dict[str, dict]] = {
    "clip": {
        "clip + prompt ensemble": {"strategies": {"clip": {"prompt_ensemble": True}}},
    },
    "clip_nlp": {
        "clip_nlp - BM25": {"strategies": {"clip_nlp": {"weights": {"bm25": 0.0}}}},
        "clip_nlp - dense doc": {"strategies": {"clip_nlp": {"weights": {"doc": 0.0}}}},
        "clip_nlp - CLIP": {"strategies": {"clip_nlp": {"weights": {"clip": 0.0}}}},
        "clip_nlp - reranker": {
            "strategies": {"clip_nlp": {"rerank": {"enabled": False}}}
        },
    },
    "vlm": {
        "vlm only (no CLIP)": {"strategies": {"vlm": {"clip_weight": 0.0}}},
        "vlm + CLIP 0.35": {"strategies": {"vlm": {"clip_weight": 0.35}}},
    },
}


# --------------------------------------------------------------------------- #
# Strategy runners
# --------------------------------------------------------------------------- #
class _CachedRetriever:
    """Memoise strategy searches so metrics and per-query values share one run."""

    def __init__(self, strategy: Strategy, key: str) -> None:
        self.strategy = strategy
        self.key = key
        self.cache: dict[tuple[str, int], list[str]] = {}
        self.latencies: list[float] = []
        self.errors = 0
        self._image_ids: dict[int, str] = {}

    def _image_id(self, pk: int) -> str:
        """Qrels are keyed by ``image_id``; strategy hits carry the SQLite pk."""
        cached = self._image_ids.get(pk)
        if cached is not None:
            return cached
        record = self.strategy.store.get(pk) if self.strategy.store is not None else None
        image_id = record.image_id if record is not None else str(pk)
        self._image_ids[pk] = image_id
        return image_id

    def __call__(self, query: str, k: int) -> list[str]:
        cached = self.cache.get((query, k))
        if cached is not None:
            return cached
        try:
            output = self.strategy.search(query, k)
        except Exception as exc:  # noqa: BLE001 - a failed strategy should not kill compare
            logger.warning("%s search failed for %r: %s", self.key, query, exc)
            self.errors += 1
            self.cache[(query, k)] = []
            return []
        self.latencies.append(float(output.latency_ms))
        ids = [self._image_id(hit.image_pk) for hit in output.hits]
        self.cache[(query, k)] = ids
        return ids


def _latency_summary(values: Sequence[float]) -> dict:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "num_queries": 0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean_ms": round(statistics.mean(ordered), 3),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "num_queries": len(ordered),
    }


def evaluate_strategy(
    key: str,
    strategy: Strategy,
    queries: Sequence[EvalQuery],
    top_k: Sequence[int] = (1, 5, 10),
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    strategy.ensure_loaded()
    if not strategy.ready:
        return {
            "strategy": key,
            "ready": False,
            "error": strategy.load_error,
            "metrics": {},
            "per_query": [],
            "per_category": {},
            "per_split": {},
            "latency": {},
            "num_errors": 0,
        }
    retriever = _CachedRetriever(strategy, key)
    report = evaluate_queries(retriever, queries, top_k=top_k)
    values = per_query_values(retriever, queries, top_k=top_k)
    intervals = bootstrap_ci(values, n_boot=n_boot, seed=seed)
    report["metrics"] = {
        name: (round(value, 6) if isinstance(value, float) else value)
        for name, value in report["metrics"].items()
    }
    report["ci"] = {
        name: {
            "mean": round(mean, 6),
            "low": round(low, 6),
            "high": round(high, 6),
        }
        for name, (mean, low, high) in intervals.items()
    }
    report.update(
        {
            "strategy": key,
            "title": strategy.title,
            "ready": True,
            "error": None,
            "latency": _latency_summary(retriever.latencies),
            "num_errors": retriever.errors,
            "per_query_values": {name: values_list for name, values_list in values.items()},
        }
    )
    return report


def _pairwise(results: dict[str, dict], n_boot: int, seed: int) -> list[dict]:
    """Paired bootstrap for every pair x metric, Holm-corrected as one family.

    Correcting separately inside each pair would treat the three pairwise
    comparisons as independent families; the report claims one family, so all
    tests share a single Holm adjustment.
    """
    ready = [key for key, result in results.items() if result.get("ready")]
    pairs: list[dict] = []
    pvalues: dict[str, float] = {}
    for i, a in enumerate(ready):
        for b in ready[i + 1 :]:
            va = results[a]["per_query_values"]
            vb = results[b]["per_query_values"]
            rows: list[dict] = []
            for metric in COMPARE_METRICS:
                if metric not in va or metric not in vb:
                    continue
                delta, low, high, p = paired_bootstrap_test(
                    va[metric], vb[metric], n_boot=n_boot, seed=seed
                )
                pvalues[f"{a}|{b}|{metric}"] = p
                rows.append(
                    {
                        "metric": metric,
                        "delta": round(delta, 6),
                        "ci_low": round(low, 6),
                        "ci_high": round(high, 6),
                        "p_raw": round(p, 6),
                        "wins": win_tie_loss(va[metric], vb[metric]),
                    }
                )
            pairs.append({"a": a, "b": b, "comparisons": rows})
    adjusted = holm_bonferroni(pvalues)
    for pair in pairs:
        for row in pair["comparisons"]:
            key = f"{pair['a']}|{pair['b']}|{row['metric']}"
            row["p_holm"] = round(adjusted.get(key, 1.0), 6)
            row["significant"] = row["p_holm"] < 0.05
            row["a"] = pair["a"]
            row["b"] = pair["b"]
    return pairs


def load_diagnostics(
    cfg: AttrDict,
    keys: Sequence[str],
    *,
    eval_set: str | None = None,
    num_queries: int | None = None,
) -> dict[str, dict]:
    """Read ``reports/eval_<key>_raw_captions.json`` produced by the leaky diagnostic.

    The main table is the reviewed, leakage-controlled set. Raw captions as
    queries are reported alongside, clearly labelled, because they overstate any
    strategy that indexes annotations. A file left over from a different query
    set is ignored rather than compared against the current main table.
    """
    out: dict[str, dict] = {}
    root = reports_dir(cfg)
    expected_set = f"{eval_set}.raw_captions" if eval_set else None
    for key in keys:
        path = root / f"eval_{key}_raw_captions.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not data.get("ready"):
            continue
        if expected_set is not None and data.get("eval_set") not in (None, expected_set):
            logger.warning(
                "Ignoring %s: it was produced for %r, not %r",
                path.name,
                data.get("eval_set"),
                expected_set,
            )
            continue
        if num_queries is not None:
            reported = int((data.get("metrics") or {}).get("num_queries", 0))
            if reported and reported != num_queries:
                logger.warning(
                    "Ignoring %s: %d queries, main table has %d",
                    path.name,
                    reported,
                    num_queries,
                )
                continue
        metrics = data.get("metrics") or {}
        out[key] = {
            "mAP": round(float(metrics.get("mAP", 0.0)), 4),
            "MRR": round(float(metrics.get("MRR", 0.0)), 4),
            "ndcg@10": round(float(metrics.get("ndcg@10", 0.0)), 4),
            "num_queries": int(metrics.get("num_queries", 0)),
            "eval_set": data.get("eval_set"),
        }
    return out


def _apply_overrides(cfg: AttrDict, overrides: dict) -> AttrDict:
    variant = copy.deepcopy(cfg)

    def walk(node, patch: dict) -> None:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(node.get(key), dict):
                walk(node[key], value)
            else:
                node[key] = value

    walk(variant, overrides)
    return variant


def run_ablations(
    cfg: AttrDict,
    results: dict[str, dict],
    queries: Sequence[EvalQuery],
    top_k: Sequence[int],
    n_boot: int,
    seed: int,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    base_store = None
    for result in results.values():
        if result.get("_strategy") is not None:
            base_store = result["_strategy"].store
            break
    if base_store is None:
        return out
    for key, variants in ABLATIONS.items():
        if not results.get(key, {}).get("ready"):
            continue
        for label, overrides in variants.items():
            try:
                variant_cfg = _apply_overrides(cfg, overrides)
                ctx = StrategyContext(variant_cfg, base_store)
                strategy = get_strategy(key, ctx)
                result = evaluate_strategy(
                    key, strategy, queries, top_k=top_k, n_boot=n_boot, seed=seed
                )
                result["label"] = label
                result["config_hash"] = config_hash(variant_cfg.strategies.get(key, {}))
                result.pop("_strategy", None)
                out[label] = result
                logger.info("Ablation %s: mAP=%.4f", label, result["metrics"].get("mAP", 0.0))
            except Exception as exc:  # noqa: BLE001 - ablation is best effort
                logger.warning("Ablation %s failed: %s", label, exc)
                out[label] = {"strategy": key, "label": label, "ready": False, "error": str(exc)}
    return out


def _failures(result: dict, queries: Sequence[EvalQuery], k: int = 5) -> list[dict]:
    by_qid = {query.qid: query for query in queries}
    worst = sorted(
        (row for row in result.get("per_query", [])),
        key=lambda row: (row["ap"], row["rank"] is not None, row["rank"] or 10**6),
    )[:10]
    out = []
    for row in worst:
        query = by_qid.get(row["qid"])
        # Show grade-2 (the source meme) before rule-derived grade-1 duplicates.
        relevant_ids = []
        if query:
            relevant_ids = sorted(
                query.relevant,
                key=lambda image_id: -int(query.grades.get(image_id, 1)),
            )[:5]
        out.append(
            {
                "qid": row["qid"],
                "query": row["query"],
                "category": row["category"],
                "ap": row["ap"],
                "rank": row["rank"],
                "relevant": [
                    {"image_id": image_id, "grade": int(query.grades.get(image_id, 1))}
                    for image_id in relevant_ids
                ],
                "retrieved": [
                    {"image_id": image_id}
                    for image_id in list(row.get("retrieved", []))[:k]
                ],
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def render_html(report: dict) -> str:
    rows = []
    for key, result in report["strategies"].items():
        if not result.get("ready"):
            rows.append(
                f"<tr><td>{html.escape(key)}</td><td colspan='7'>"
                f"unavailable — {html.escape(str(result.get('error')))}</td></tr>"
            )
            continue
        metrics = result["metrics"]
        ci = result.get("ci", {})
        latency = result.get("latency", {})

        def with_ci(name: str, metrics=metrics, ci=ci) -> str:
            value = _fmt(metrics.get(name))
            interval = ci.get(name)
            if not interval:
                return value
            return f"{value} <small>[{_fmt(interval['low'], 3)}, {_fmt(interval['high'], 3)}]</small>"

        rows.append(
            "<tr>"
            f"<td><b>{html.escape(result.get('title') or key)}</b><br><code>{html.escape(key)}</code></td>"
            f"<td>{with_ci('mAP')}</td>"
            f"<td>{with_ci('MRR')}</td>"
            f"<td>{with_ci('ndcg@10')}</td>"
            f"<td>{_fmt(metrics.get('precision@1'))}</td>"
            f"<td>{_fmt(metrics.get('recall@10'))}</td>"
            f"<td>{_fmt(metrics.get('median_rank'), 0)}</td>"
            f"<td>{_fmt(latency.get('p50_ms'), 1)} / {_fmt(latency.get('p95_ms'), 1)}</td>"
            "</tr>"
        )

    category_rows = []
    categories = sorted(
        {name for result in report["strategies"].values() for name in result.get("per_category", {})}
    )
    for category in categories:
        cells = []
        for key in report["strategies"]:
            group = report["strategies"][key].get("per_category", {}).get(category)
            cells.append(_fmt(group.get("mAP")) if group else "—")
        category_rows.append(
            f"<tr><td>{html.escape(category)}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        )

    split_rows = []
    splits = sorted(
        {name for result in report["strategies"].values() for name in result.get("per_split", {})}
    )
    for split in splits:
        cells = []
        for key in report["strategies"]:
            group = report["strategies"][key].get("per_split", {}).get(split)
            cells.append(
                f"{_fmt(group.get('mAP'))} <small>(n={group['num_queries']})</small>"
                if group
                else "—"
            )
        split_rows.append(
            f"<tr><td>{html.escape(split)}</td>"
            + "".join(f"<td>{cell}</td>" for cell in cells)
            + "</tr>"
        )

    pair_rows = []
    for pair in report.get("pairwise", []):
        for comp in pair["comparisons"]:
            wtl = comp["wins"]
            pair_rows.append(
                "<tr>"
                f"<td>{html.escape(pair['a'])} vs {html.escape(pair['b'])}</td>"
                f"<td>{html.escape(comp['metric'])}</td>"
                f"<td>{comp['delta']:+.4f}</td>"
                f"<td>[{comp['ci_low']:+.4f}, {comp['ci_high']:+.4f}]</td>"
                f"<td>{comp['p_raw']:.4f}</td>"
                f"<td>{comp['p_holm']:.4f}</td>"
                f"<td>{'yes' if comp['significant'] else 'no'}</td>"
                f"<td>{wtl['win']}/{wtl['tie']}/{wtl['loss']}</td>"
                "</tr>"
            )

    ablation_rows = []
    for label, result in report.get("ablations", {}).items():
        if not result.get("ready"):
            ablation_rows.append(
                f"<tr><td>{html.escape(label)}</td><td colspan='4'>"
                f"{html.escape(str(result.get('error')))}</td></tr>"
            )
            continue
        metrics = result["metrics"]
        ablation_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{_fmt(metrics.get('mAP'))}</td>"
            f"<td>{_fmt(metrics.get('MRR'))}</td>"
            f"<td>{_fmt(metrics.get('ndcg@10'))}</td>"
            f"<td>{_fmt(result.get('latency', {}).get('p50_ms'), 1)} ms</td>"
            "</tr>"
        )

    diagnostic_rows = []
    for key, data in report.get("diagnostics", {}).items():
        main_map = report["strategies"].get(key, {}).get("metrics", {}).get("mAP")
        delta = f"{data['mAP'] - main_map:+.4f}" if main_map is not None else "—"
        diagnostic_rows.append(
            "<tr>"
            f"<td><code>{html.escape(key)}</code></td>"
            f"<td>{_fmt(data['mAP'])}</td>"
            f"<td>{delta}</td>"
            f"<td>{_fmt(data.get('MRR'))}</td>"
            f"<td>{_fmt(data.get('ndcg@10'))}</td>"
            f"<td>{_fmt(data.get('num_queries'), 0)}</td>"
            "</tr>"
        )

    probe = report.get("probe", {})
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meme retrieval — strategy comparison</title>
<style>
:root {{ color-scheme: light dark; --line: #8884; --accent: #7c5cff; }}
body {{ font: 15px/1.55 system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem auto;
       max-width: 1120px; padding: 0 1rem; }}
h1, h2 {{ line-height: 1.2; }}
h1 {{ margin-bottom: .25rem; }}
.sub {{ opacity: .7; margin-top: 0; }}
table {{ border-collapse: collapse; width: 100%; margin: .6rem 0 1.6rem; }}
th, td {{ text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--line); }}
th {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; opacity: .75; }}
small {{ opacity: .65; }}
code {{ font-size: .85em; }}
.badge {{ display: inline-block; padding: .1rem .5rem; border-radius: 999px;
          background: #7c5cff22; color: var(--accent); font-size: .8rem; }}
.note {{ background: #7c5cff11; border-left: 3px solid var(--accent); padding: .6rem .9rem; }}
</style></head><body>
<h1>Meme retrieval — CLIP vs CLIP+NLP vs VLM</h1>
<p class="sub">Generated {html.escape(report['generated_at'])} · eval set
<code>{html.escape(report.get('eval_set', 'unknown'))}</code> ·
{report.get('num_queries', 0)} queries · bootstrap {report.get('n_boot', 0)} resamples ·
seed {report.get('seed')}</p>

<h2>Overall</h2>
<table><thead><tr><th>Strategy</th><th>mAP [95% CI]</th><th>MRR</th><th>nDCG@10</th>
<th>P@1</th><th>R@10</th><th>median rank</th><th>p50 / p95 latency</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>

<h2>mAP by query category</h2>
<table><thead><tr><th>Category</th>{''.join(f'<th>{html.escape(k)}</th>' for k in report['strategies'])}
</tr></thead><tbody>{''.join(category_rows)}</tbody></table>

<h2>mAP by split</h2>
<table><thead><tr><th>Split</th>{''.join(f'<th>{html.escape(k)}</th>' for k in report['strategies'])}
</tr></thead><tbody>{''.join(split_rows) or '<tr><td colspan="4">No split metadata.</td></tr>'}</tbody></table>

<h2>Pairwise significance (paired bootstrap, Holm-corrected)</h2>
<p class="note">Delta is <code>B − A</code> for the metric. Win/tie/loss counts are per query
on that metric. Holm correction is applied once across every pairwise × metric test in the
report, so the three strategy comparisons form a single family.</p>
<table><thead><tr><th>Pair</th><th>Metric</th><th>Δ</th><th>95% CI</th><th>p raw</th>
<th>p Holm</th><th>significant</th><th>W/T/L</th></tr></thead>
<tbody>{''.join(pair_rows) or '<tr><td colspan="8">No ready strategies to compare.</td></tr>'}</tbody></table>

<h2>Ablations</h2>
<table><thead><tr><th>Variant</th><th>mAP</th><th>MRR</th><th>nDCG@10</th><th>p50 latency</th>
</tr></thead><tbody>{''.join(ablation_rows) or '<tr><td colspan="5">Run with --ablations to populate.</td></tr>'}
</tbody></table>

<h2>Failure analysis</h2>
<p><a href="failures.html">Worst queries per strategy, with relevant and retrieved images →</a></p>

<h2>Diagnostic: raw captions as queries (leaky, not the main table)</h2>
<p class="note">The main table uses rewritten queries that pass the 8-gram leak gate.
Here the original human captions are used verbatim, which is exactly what strategy B
indexes. It is an upper-bound diagnostic showing how much annotation overlap can
inflate a score — never mix these numbers into the main comparison.</p>
<table><thead><tr><th>Strategy</th><th>mAP (diagnostic)</th><th>Δ vs main mAP</th>
<th>MRR</th><th>nDCG@10</th><th>queries</th></tr></thead>
<tbody>{''.join(diagnostic_rows) or
    '<tr><td colspan="6">Run <code>tsr eval --queries data/eval/memecap_eval_v1.raw_captions.json</code> first.</td></tr>'}
</tbody></table>

<h2>Probe</h2>
<table><tbody>
<tr><td>Images in gallery</td><td>{_fmt(probe.get('num_images'), 0)}</td></tr>
<tr><td>Queries</td><td>{_fmt(report.get('num_queries'), 0)}</td></tr>
<tr><td>Leakage-checked queries passed</td><td>{_fmt(probe.get('leakage_passed'), 0)}</td></tr>
<tr><td>Build cost of VLM captions</td><td>{html.escape(str(probe.get('vlm_captions', 'n/a')))}</td></tr>
</tbody></table>

<p><small>Metrics treat grade ≥ 1 as relevant; nDCG@10 uses graded gains
(2 = source meme, 1 = rule-derived near-duplicate).</small></p>
</body></html>"""


def _failure_strip(
    items: list[dict],
    relevant_ids: set[str],
    with_grade: bool = False,
) -> str:
    figures = []
    for entry in items:
        pk = entry.get("pk")
        image_id = str(entry.get("image_id") or "")
        grade = entry.get("grade")
        classes = "thumb"
        if image_id in relevant_ids:
            classes += " relevant"
        image = (
            f"<img loading='lazy' src='/api/image/{int(pk)}' "
            f"alt='{html.escape(image_id)}'>"
            if pk is not None
            else "<div class='missing'>missing</div>"
        )
        badge = (
            f"<span class='badge'>g{int(grade)}</span>"
            if with_grade and grade is not None
            else ""
        )
        figures.append(
            f"<figure class='{classes}'>{image}"
            f"<figcaption>{html.escape(image_id)}{badge}</figcaption></figure>"
        )
    return "<div class='strip'>" + ("".join(figures) or "—") + "</div>"


def render_failures_html(report: dict) -> str:
    failures = report.get("failures") or {}
    sections = []
    for key, rows in failures.items():
        title = report.get("strategies", {}).get(key, {}).get("title") or key
        cards = []
        for row in rows:
            relevant_ids = {item["image_id"] for item in row.get("relevant", [])}
            rank = row.get("rank")
            cards.append(
                "<tr>"
                f"<td class='q'><b>{html.escape(str(row['qid']))}</b> "
                f"<span class='badge'>{html.escape(str(row['category']))}</span><br>"
                f"{html.escape(str(row['query']))}<br>"
                f"<small>AP {_fmt(row.get('ap'), 3)} · "
                f"rank {_fmt(rank, 0) if rank is not None else 'miss'}</small></td>"
                f"<td>{_failure_strip(row.get('relevant', []), relevant_ids, True)}</td>"
                f"<td>{_failure_strip(row.get('retrieved', []), relevant_ids)}</td>"
                "</tr>"
            )
        sections.append(
            f"<h2>{html.escape(str(title))} <code>{html.escape(key)}</code></h2>"
            "<table><thead><tr><th>Worst queries</th><th>Relevant (grade)</th>"
            "<th>Top 5 retrieved</th></tr></thead><tbody>"
            + "".join(cards)
            + "</tbody></table>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meme retrieval — failure analysis</title>
<style>
:root {{ color-scheme: light dark; --line: #8884; --accent: #7c5cff; }}
body {{ font: 15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem auto;
       max-width: 1280px; padding: 0 1rem; }}
h1, h2 {{ line-height: 1.2; }}
.sub {{ opacity: .7; margin-top: 0; }}
table {{ border-collapse: collapse; width: 100%; margin: .6rem 0 2rem; }}
th, td {{ text-align: left; padding: .55rem; border-bottom: 1px solid var(--line);
         vertical-align: top; }}
th {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; opacity: .75; }}
.q {{ width: 30%; }}
.badge {{ display: inline-block; padding: .05rem .4rem; border-radius: 999px;
          background: #7c5cff22; color: var(--accent); font-size: .75rem; }}
.strip {{ display: flex; gap: .45rem; flex-wrap: wrap; }}
.thumb {{ margin: 0; width: 104px; }}
.thumb img, .thumb .missing {{ width: 104px; height: 104px; object-fit: cover;
          border-radius: 6px; background: #8882; display: block; }}
.thumb.relevant img {{ outline: 2px solid #3fb950; }}
.thumb figcaption {{ font-size: .65rem; opacity: .7; overflow-wrap: anywhere; }}
</style></head><body>
<h1>Failure analysis</h1>
<p class="sub">Generated {html.escape(str(report.get('generated_at', '')))} · eval set
<code>{html.escape(str(report.get('eval_set', 'unknown')))}</code> ·
{report.get('num_queries', 0)} queries · worst 10 per strategy by average precision.
Green outline = retrieved image is in the graded qrels.</p>
<p class="sub">Open this page through the running server
(<code>/api/eval/failures.html</code>) so the thumbnails resolve.</p>
{''.join(sections) or '<p>No failures recorded.</p>'}
</body></html>"""


def write_report(report: dict, out_dir: str | Path) -> tuple[Path, Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "compare.json"
    html_path = out_dir / "compare.html"
    failures_path = out_dir / "failures.html"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    html_path.write_text(render_html(report), encoding="utf-8")
    failures_path.write_text(render_failures_html(report), encoding="utf-8")
    return json_path, html_path, failures_path


def compare_strategies(
    cfg: AttrDict,
    strategies: dict[str, Strategy],
    queries: Sequence[EvalQuery],
    *,
    top_k: Sequence[int] | None = None,
    n_boot: int | None = None,
    seed: int | None = None,
    ablations: bool = False,
    eval_set: str | None = None,
    write: bool = True,
) -> dict:
    """Evaluate every ready strategy on the same queries and report differences."""
    flagged = [
        query.qid
        for query in queries
        if (query.metadata.get("leakage") or {}).get("passed") is False
    ]
    diagnostic = [
        query.qid
        for query in queries
        if (query.metadata or {}).get("query_origin") == "raw_caption_diagnostic"
    ]
    if flagged or diagnostic:
        raise ValueError(
            "refusing to publish the main comparison from a diagnostic query set "
            f"({len(flagged)} leakage-flagged, {len(diagnostic)} raw-caption queries; "
            f"first: {(flagged or diagnostic)[:3]}). Use `tsr eval --queries ...` for "
            "diagnostics."
        )
    top_k = list(top_k or cfg.eval.get("top_k") or [1, 5, 10])
    n_boot = int(n_boot or cfg.eval.get("bootstrap", 1000))
    seed = int(seed if seed is not None else cfg.eval.get("seed", 42))
    results: dict[str, dict] = {}
    for key, strategy in strategies.items():
        logger.info("Evaluating strategy %s on %d queries", key, len(queries))
        results[key] = evaluate_strategy(
            key, strategy, queries, top_k=top_k, n_boot=n_boot, seed=seed
        )
        results[key]["_strategy"] = strategy

    pairwise = _pairwise(results, n_boot=n_boot, seed=seed)
    ablation_results = (
        run_ablations(cfg, results, queries, top_k, n_boot, seed) if ablations else {}
    )

    leak_checks = [
        query.metadata.get("leakage", {}).get("passed")
        for query in queries
        if query.metadata.get("leakage")
    ]
    store = next(
        (r["_strategy"].store for r in results.values() if r.get("_strategy")),
        None,
    )
    report_eval_set = eval_set or str(cfg.eval.get("set_name", "eval"))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_set": report_eval_set,
        "num_queries": len(queries),
        "top_k": top_k,
        "n_boot": n_boot,
        "seed": seed,
        "strategies": {
            key: {k: v for k, v in result.items() if not k.startswith("_")}
            for key, result in results.items()
        },
        "pairwise": pairwise,
        "ablations": ablation_results,
        "diagnostics": load_diagnostics(
            cfg,
            list(results),
            eval_set=str(report_eval_set),
            num_queries=len(queries),
        ),
        "probe": {
            "num_images": store.count() if store is not None else 0,
            "leakage_passed": sum(1 for value in leak_checks if value),
            "leakage_flagged": sum(1 for value in leak_checks if value is False),
            "vlm_captions": (
                store.vlm_count(**vlm_doc_filters(cfg)) if store is not None else 0
            ),
        },
    }
    report["failures"] = failure_cases(report, queries, store)

    for result in results.values():
        result.pop("_strategy", None)

    if write:
        json_path, html_path, failures_path = write_report(report, reports_dir(cfg))
        logger.info("Wrote %s, %s and %s", json_path, html_path, failures_path)
    return report


def load_report(cfg: AttrDict) -> dict | None:
    path = reports_dir(cfg) / "compare.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def failure_cases(
    report: dict,
    queries: Sequence[EvalQuery],
    store=None,
) -> dict[str, list[dict]]:
    """Worst queries per strategy, enriched with pk/path for the HTML gallery."""
    records = {rec.image_id: rec for rec in store.iter_all()} if store is not None else {}
    out: dict[str, list[dict]] = {}
    for key, result in report.get("strategies", {}).items():
        if not result.get("ready"):
            continue
        rows = _failures(result, queries)
        for row in rows:
            for entry in row["relevant"] + row["retrieved"]:
                record = records.get(entry["image_id"])
                entry["pk"] = record.id if record is not None else None
                entry["path"] = record.path if record is not None else None
        out[key] = rows
    return out


__all__ = [
    "ABLATIONS",
    "compare_strategies",
    "evaluate_strategy",
    "failure_cases",
    "load_report",
    "render_html",
    "write_report",
]
