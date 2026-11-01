"""Command-line interface.

Examples
--------
    python -m tsr prepare --limit 300
    python -m tsr build --strategy gallery
    python -m tsr vlm-caption --limit 20
    python -m tsr build --strategy all
    python -m tsr make-eval --num 240
    python -m tsr eval --strategy all
    python -m tsr compare --ablations
    python -m tsr serve
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tsr.config import (
    cache_dir,
    data_dir,
    db_path,
    eval_dir,
    index_root,
    load_config,
    manifest_path,
    reports_dir,
    vlm_cache_path,
    vlm_doc_filters,
)
from tsr.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

ALIASES = {"a": "clip", "b": "clip_nlp", "c": "vlm"}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsr", description="Meme retrieval — CLIP vs CLIP+NLP vs VLM"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Download Meme-Cap and build the image manifest")
    _add_common(p_prepare)
    p_prepare.add_argument("--source", default="hf", choices=["hf", "gdrive"])
    p_prepare.add_argument("--limit", type=int, default=None, help="Process only N images")
    p_prepare.add_argument("--force", action="store_true", help="Redo preparation")
    p_prepare.add_argument("--skip-download", action="store_true")

    p_build = sub.add_parser("build", help="Build indexes for one or all strategies")
    _add_common(p_build)
    p_build.add_argument(
        "--strategy",
        default="all",
        help="clip | clip_nlp | vlm | all | gallery | a | b | c",
    )
    p_build.add_argument("--force", action="store_true")

    p_vlm = sub.add_parser("vlm-caption", help="Generate offline VLM descriptions")
    _add_common(p_vlm)
    p_vlm.add_argument("--limit", type=int, default=None)
    p_vlm.add_argument("--out", default=None, help="Write a sample HTML contact sheet")

    p_make = sub.add_parser("make-eval", help="Draft a leakage-checked evaluation set")
    _add_common(p_make)
    p_make.add_argument("--num", type=int, default=None)
    p_make.add_argument("--seed", type=int, default=None)
    p_make.add_argument("--force", action="store_true", help="Overwrite an existing set")
    p_make.add_argument(
        "--sources",
        default=None,
        help="Draft sources JSON written for human review (with --rewrites)",
    )
    p_make.add_argument(
        "--rewrites",
        default=None,
        help="Directory of reviewed query rewrites keyed by draft qid",
    )

    p_eval = sub.add_parser("eval", help="Evaluate one or all strategies")
    _add_common(p_eval)
    p_eval.add_argument("--strategy", default="all", help="clip | clip_nlp | vlm | all")
    p_eval.add_argument("--queries", default=None, help="Path to a queries JSON file")
    p_eval.add_argument("--top-k", type=int, nargs="*", default=None)

    p_compare = sub.add_parser("compare", help="Compare strategies with bootstrap tests")
    _add_common(p_compare)
    p_compare.add_argument("--queries", default=None)
    p_compare.add_argument("--boot", type=int, default=None)
    p_compare.add_argument("--seed", type=int, default=None)
    p_compare.add_argument("--ablations", action="store_true")

    p_serve = sub.add_parser("serve", help="Run the web server")
    _add_common(p_serve)
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true")

    p_stats = sub.add_parser("stats", help="Print dataset and index statistics")
    _add_common(p_stats)
    p_stats.add_argument("--json", action="store_true", help="Print raw JSON")

    p_thumbs = sub.add_parser("thumbnails", help="Pre-generate cached thumbnails")
    _add_common(p_thumbs)
    p_thumbs.add_argument("--size", type=int, default=480)

    p_clean = sub.add_parser("clean", help="Delete generated data")
    _add_common(p_clean)
    p_clean.add_argument("--all", action="store_true", help="Also delete indexes, DB and manifest")
    p_clean.add_argument("--raw", action="store_true", help="Also delete the raw download")
    p_clean.add_argument("--yes", action="store_true", help="Skip confirmation")

    return parser


def _load(args) -> object:
    cfg = load_config(args.config)
    setup_logging(args.log_level)
    return cfg


def _parse_strategies(value: str, available: list[str]) -> list[str]:
    if value.strip().lower() in {"all", ""}:
        return list(available)
    selected = []
    for token in value.replace(",", " ").split():
        key = ALIASES.get(token.lower(), token.lower())
        if key not in available:
            raise SystemExit(
                f"Unknown strategy {token!r}. Available: {', '.join(available)}"
            )
        if key not in selected:
            selected.append(key)
    return selected


def _load_eval_queries(cfg, path: str | None):
    from tsr.evaluate import load_queries

    file = Path(path) if path else eval_dir(cfg) / f"{cfg.eval.get('set_name', 'eval')}.json"
    queries = load_queries(file)
    if not queries:
        raise SystemExit(
            f"No eval queries at {file}. Run `python -m tsr make-eval` first or pass --queries."
        )
    return queries, file


def _gallery_records(cfg):
    from tsr.pipeline import open_store

    store = open_store(cfg)
    try:
        records = store.iter_all()
        if not records:
            raise SystemExit("Gallery is empty. Run `python -m tsr build --strategy gallery`.")
        store.enrich({rec.id: rec for rec in records})
        return records
    finally:
        store.close()


def _load_rewrites(path: str | Path) -> dict[str, str]:
    """Merge one JSON file or a directory of ``{qid: query}`` batches."""
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"No rewrite JSON files found under {path}")
    rewrites: dict[str, str] = {}
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"{file}: expected a JSON object of qid -> query")
        rewrites.update({str(k): str(v) for k, v in data.items()})
    return rewrites


def _make_eval(cfg, args) -> int:
    from tsr.evaluate import (
        EvalQuery,
        check_leakage,
        draft_eval_set,
        reviewed_eval_set,
        save_qrels,
        save_queries,
        scale_per_category,
    )

    set_name = str(cfg.eval.get("set_name", "eval"))
    out_path = eval_dir(cfg) / f"{set_name}.json"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite.")
    num = args.num or int(cfg.eval.get("num_queries", 240))
    seed = args.seed if args.seed is not None else int(cfg.eval.get("seed", 42))
    per_category = cfg.eval.get("per_category") or None
    ngram = int(cfg.eval.get("max_ngram_overlap", 8))
    metaphor_max_df = int(cfg.eval.get("metaphor_max_df", 10))
    min_copy_tokens = int(cfg.eval.get("min_copy_tokens", 5))
    records = _gallery_records(cfg)
    if args.rewrites:
        if not args.sources:
            raise SystemExit("--rewrites needs --sources (the draft written for review).")
        sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
        if isinstance(sources, dict):
            sources = sources.get("sources") or []
        queries, rejected = reviewed_eval_set(
            sources,
            _load_rewrites(args.rewrites),
            records,
            ngram=ngram,
            metaphor_max_df=metaphor_max_df,
            min_copy_tokens=min_copy_tokens,
        )
        if rejected:
            for item in rejected:
                logger.error("Rejected %s: %s", item["qid"], item["reason"])
            # A frozen benchmark cannot silently shrink or drop a query: the
            # numbers would be published as if the whole reviewed set ran.
            raise SystemExit(
                f"{len(rejected)}/{len(sources)} reviewed queries were rejected "
                "(missing rewrite, stale source, non-test split or leakage); fix the "
                "drafts and rerun. Nothing was written."
            )
    else:
        queries = draft_eval_set(
            records,
            num_queries=num,
            seed=seed,
            per_category=(
                scale_per_category(dict(per_category), num) if per_category else None
            ),
            ngram=ngram,
            metaphor_max_df=metaphor_max_df,
            min_copy_tokens=min_copy_tokens,
        )
    save_queries(queries, out_path)
    qrels_path = save_qrels(queries, eval_dir(cfg) / f"{set_name}.qrels.jsonl")

    # Diagnostic set: raw captions as queries. Leaky by construction, upper bound only.
    raw_queries = []
    by_image = {rec.image_id: rec for rec in records}
    for query in queries:
        source = by_image.get(str(query.metadata.get("image_id")))
        if source is None or not source.img_captions:
            continue
        raw_text = source.img_captions[0]
        raw_queries.append(
            EvalQuery(
                qid=query.qid,
                query=raw_text,
                relevant=query.relevant,
                category=query.category,
                grades=query.grades,
                metadata={
                    **query.metadata,
                    "query_origin": "raw_caption_diagnostic",
                    "leakage": check_leakage(raw_text, source.img_captions),
                },
            )
        )
    raw_path = save_queries(raw_queries, eval_dir(cfg) / f"{set_name}.raw_captions.json")

    flagged = sum(
        1 for query in queries if not query.metadata.get("leakage", {}).get("passed")
    )
    logger.info(
        "Wrote %d queries to %s (%d flagged) and qrels to %s; raw diagnostic set: %s",
        len(queries),
        out_path,
        flagged,
        qrels_path,
        raw_path,
    )
    for query in queries[:3]:
        logger.info("  %s [%s] %s", query.qid, query.category, query.query)
    if len(queries) < num:
        logger.warning(
            "Only %d/%d queries survived leakage checks; expand the candidate pool "
            "or write queries manually.",
            len(queries),
            num,
        )
    return 0


def _print_metrics(name: str, result: dict) -> None:
    metrics = result.get("metrics", {})
    logger.info(
        "%-10s mAP=%.4f MRR=%.4f nDCG@10=%.4f P@1=%.4f misses=%s p50=%.1fms",
        name,
        metrics.get("mAP", 0.0),
        metrics.get("MRR", 0.0),
        metrics.get("ndcg@10", 0.0),
        metrics.get("precision@1", 0.0),
        metrics.get("misses"),
        result.get("latency", {}).get("p50_ms", 0.0),
    )


def _eval_set_label(path: str | Path, cfg) -> str:
    """Suffix for report filenames so diagnostic sets never overwrite the main set."""
    import re

    stem = Path(path).stem
    main = str(cfg.eval.get("set_name", "eval"))
    if stem == main:
        return ""
    if stem.startswith(main):
        stem = stem[len(main) :].lstrip(".")
    return "_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", stem) if stem else ""


def _eval(cfg, args) -> int:
    from tsr.compare import evaluate_strategy
    from tsr.pipeline import write_json
    from tsr.service import StrategyManager

    queries, path = _load_eval_queries(cfg, args.queries)
    label = _eval_set_label(path, cfg)
    if label:
        logger.info("Evaluating diagnostic set %r (not the main table)", label.strip("_"))
    manager = StrategyManager(cfg)
    try:
        selected = _parse_strategies(args.strategy, manager.keys())
        top_k = args.top_k or list(cfg.eval.get("top_k") or [1, 5, 10])
        for key in selected:
            result = evaluate_strategy(
                key,
                manager.get(key),
                queries,
                top_k=top_k,
                n_boot=int(cfg.eval.get("bootstrap", 1000)),
                seed=int(cfg.eval.get("seed", 42)),
            )
            out = reports_dir(cfg) / f"eval_{key}{label}.json"
            result["eval_set"] = Path(path).stem
            write_json(out, result)
            if result.get("ready"):
                _print_metrics(key, result)
                logger.info("Wrote %s", out)
            else:
                logger.warning("%s unavailable: %s", key, result.get("error"))
        logger.info("Evaluated %d queries from %s", len(queries), path)
        return 0
    finally:
        manager.close()


def _compare(cfg, args) -> int:
    from tsr.compare import compare_strategies
    from tsr.service import StrategyManager

    queries, path = _load_eval_queries(cfg, args.queries)
    label = _eval_set_label(path, cfg)
    if label:
        raise SystemExit(
            f"{Path(path).name} is a diagnostic set; the main comparison table must use "
            "the leakage-controlled set. Run `tsr eval --queries ...` for diagnostics "
            "(it writes a separate report)."
        )
    manager = StrategyManager(cfg)
    try:
        strategies = {key: manager.get(key) for key in manager.keys()}
        report = compare_strategies(
            cfg,
            strategies,
            queries,
            n_boot=args.boot or int(cfg.eval.get("bootstrap", 1000)),
            seed=args.seed if args.seed is not None else int(cfg.eval.get("seed", 42)),
            ablations=args.ablations,
            eval_set=Path(path).stem,
        )
        for key, result in report["strategies"].items():
            if result.get("ready"):
                _print_metrics(key, result)
            else:
                logger.warning("%s unavailable: %s", key, result.get("error"))
        for pair in report.get("pairwise", []):
            for comp in pair["comparisons"]:
                if comp["metric"] in {"mAP", "ndcg@10"}:
                    logger.info(
                        "%s vs %s %s: delta=%+.4f p_holm=%.4f %s",
                        pair["a"],
                        pair["b"],
                        comp["metric"],
                        comp["delta"],
                        comp["p_holm"],
                        "significant" if comp["significant"] else "n.s.",
                    )
        logger.info("Report: %s", reports_dir(cfg) / "compare.html")
        return 0
    finally:
        manager.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _load(args)

    if args.command == "prepare":
        from tsr.pipeline import prepare_dataset

        limit = args.limit if args.limit is not None else cfg.dataset.get("limit")
        count = prepare_dataset(
            cfg,
            source=args.source,
            limit=limit,
            force=args.force,
            skip_download=args.skip_download,
        )
        logger.info("Prepared %d images in %s", count, data_dir(cfg) / "images")
        return 0

    if args.command == "build":
        from tsr.pipeline import build

        results = build(cfg, strategy=args.strategy, force=args.force)
        for key, size in results.items():
            logger.info("%s: %d vectors", key, size)
        return 0

    if args.command == "vlm-caption":
        from tsr.encoders.vlm import caption_gallery
        from tsr.pipeline import open_store

        store = open_store(cfg)
        try:
            result = caption_gallery(cfg, store, limit=args.limit, out_html=args.out)
            available = store.vlm_count(**vlm_doc_filters(cfg))
        finally:
            store.close()
        logger.info(
            "VLM run: %d ok, %d errors, %d cached; %d descriptions available",
            result["ok"],
            result["errors"],
            result["skipped"],
            available,
        )
        return 0

    if args.command == "make-eval":
        return _make_eval(cfg, args)

    if args.command == "eval":
        return _eval(cfg, args)

    if args.command == "compare":
        return _compare(cfg, args)

    if args.command == "serve":
        import uvicorn

        if args.config:
            os.environ["TSR_CONFIG"] = str(Path(args.config).resolve())
        host = args.host or cfg.app.host
        port = args.port or int(cfg.app.port)
        logger.info("Serving on http://%s:%s", host, port)
        uvicorn.run("tsr.api:app", host=host, port=int(port), reload=bool(args.reload))
        return 0

    if args.command == "stats":
        from tsr.service import StrategyManager

        manager = StrategyManager(cfg)
        try:
            data = manager.stats()
        finally:
            manager.close()
        if args.json:
            print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        else:
            dataset = data["dataset"]
            logger.info("data dir : %s", cfg.paths.data_dir)
            logger.info("images   : %d", dataset["num_images"])
            logger.info("manifest : %s", dataset["manifest"])
            logger.info("vlm docs : %d", dataset["vlm_docs"])
            logger.info("clip     : %s", data["models"]["clip"])
            logger.info("text     : %s", data["models"]["text"])
            for item in data["strategies"]:
                logger.info(
                    "strategy %-9s ready=%-5s docs=%s %s",
                    item.get("key") or item.get("strategy"),
                    item.get("ready"),
                    item.get("num_docs"),
                    item.get("error") or "",
                )
        return 0

    if args.command == "thumbnails":
        from tsr.pipeline import pregenerate_thumbnails

        pregenerate_thumbnails(cfg, size=args.size)
        return 0

    if args.command == "clean":
        import shutil

        targets = [cache_dir(cfg)]
        if args.all:
            targets += [
                data_dir(cfg) / "images",
                index_root(cfg),
                db_path(cfg),
                manifest_path(cfg),
                manifest_path(cfg).with_suffix(".meta.json"),
                eval_dir(cfg),
                reports_dir(cfg),
                vlm_cache_path(cfg).parent,
                data_dir(cfg) / "settings.json",
            ]
        if args.raw:
            targets += [data_dir(cfg) / "raw"]
        missing = [str(path) for path in targets if not Path(path).exists()]
        if args.all and not args.yes:
            reply = input(f"Delete {len(targets)} paths under {data_dir(cfg)}? [y/N] ")
            if reply.strip().lower() not in {"y", "yes"}:
                logger.info("Aborted.")
                return 1
        removed = 0
        for target in targets:
            target = Path(target)
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
                removed += 1
            elif target.exists():
                target.unlink()
                removed += 1
        logger.info("Removed %d path(s) (%d already absent).", removed, len(missing))
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
