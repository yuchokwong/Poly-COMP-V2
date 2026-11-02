"""FastAPI application: three retrieval strategies, comparison and evaluation UI."""

from __future__ import annotations

import hmac
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tsr.compare import load_report
from tsr.config import AttrDict, eval_dir, load_config, reports_dir, web_dir
from tsr.evaluate import load_queries
from tsr.logging_utils import get_logger, setup_logging
from tsr.runtime import configure_threads
from tsr.schemas import (
    CompareResponse,
    DetailResponse,
    EvalSummaryResponse,
    HealthResponse,
    SearchResponse,
    StatsResponse,
    StrategyStatus,
)
from tsr.service import StrategyManager, StrategyUnavailable, UnknownStrategy

logger = get_logger(__name__)

# A public demo is a query endpoint: cap the input so one visitor cannot hand
# the tokenizer and the encoders an arbitrarily large string.
MAX_QUERY_CHARS = 200

# Image ids are stable and their bytes never change, so they can be cached hard.
CACHE_HEADERS = {"Cache-Control": "public, max-age=86400"}

PAGES = {
    "/": "index.html",
    "/clip": "search.html",
    "/clip-nlp": "search.html",
    "/vlm": "search.html",
    "/compare": "compare.html",
    "/evaluation": "evaluation.html",
    "/settings": "settings.html",
}


def create_app(cfg: AttrDict | None = None) -> FastAPI:
    cfg = cfg or load_config()
    setup_logging()
    logger.info("Runtime thread budget: %d", configure_threads())

    from tsr.settings import apply_persisted

    apply_persisted(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.manager = StrategyManager(cfg)
        if bool(cfg.app.get("preload_models", False)):
            threading.Thread(
                target=app.state.manager.warmup,
                name="tsr-preload",
                daemon=True,
            ).start()
        yield
        app.state.manager.close()

    # /docs advertises the whole API surface; keep it off unless asked for.
    expose_docs = bool(cfg.app.get("expose_docs", False))
    app = FastAPI(
        title=str(cfg.app.title),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if expose_docs else None,
        redoc_url="/redoc" if expose_docs else None,
        openapi_url="/openapi.json" if expose_docs else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.app.get("cors_origins", ["*"])),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def manager() -> StrategyManager:
        return app.state.manager

    default_k = max(1, int(cfg.app.get("default_top_k", 12)))

    def clamp(k: int) -> int:
        return max(1, min(int(k), int(cfg.app.get("max_top_k", 100))))

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #
    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(**manager().health())

    @app.get("/api/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        return StatsResponse(**manager().stats())

    @app.get("/api/strategies", response_model=list[StrategyStatus])
    def strategies() -> list[StrategyStatus]:
        return [StrategyStatus(**item) for item in manager().status()]

    @app.get("/api/strategies/{key}/stats", response_model=StrategyStatus)
    def strategy_stats(key: str) -> StrategyStatus:
        try:
            strategy = manager().get(key)
        except UnknownStrategy as exc:
            raise HTTPException(status_code=404, detail=f"Unknown strategy {key!r}") from exc
        try:
            data = strategy.stats()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        data.setdefault("key", key)
        return StrategyStatus(**data)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    @app.get("/api/strategies/{key}/search", response_model=SearchResponse)
    def strategy_search(
        key: str,
        q: str = Query(
            ..., min_length=1, max_length=MAX_QUERY_CHARS,
            description="Natural-language query",
        ),
        k: int = Query(default=default_k, ge=1, le=100),
    ) -> SearchResponse:
        try:
            payload = manager().search_payload(key, q, clamp(k))
        except UnknownStrategy as exc:
            raise HTTPException(status_code=404, detail=f"Unknown strategy {key!r}") from exc
        except StrategyUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SearchResponse(**payload)

    @app.get("/api/search", response_model=SearchResponse)
    def legacy_search(
        q: str = Query(..., min_length=1, max_length=MAX_QUERY_CHARS),
        k: int = Query(default=default_k, ge=1, le=100),
        strategy: str = Query(default="clip"),
    ) -> SearchResponse:
        """Backward-compatible endpoint; defaults to the CLIP baseline."""
        return strategy_search(strategy, q, k)

    @app.get("/api/compare", response_model=CompareResponse)
    def compare(
        q: str = Query(..., min_length=1, max_length=MAX_QUERY_CHARS),
        k: int = Query(default=10, ge=1, le=50),
    ) -> CompareResponse:
        columns = manager().compare(q, clamp(k))
        errors = {
            key: str(column.get("error"))
            for key, column in columns.items()
            if column.get("error")
        }
        return CompareResponse(
            query=q,
            top_k=clamp(k),
            columns={
                key: SearchResponse(**{**column, "query": q})
                for key, column in columns.items()
            },
            errors=errors,
        )

    @app.get("/api/similar/{pk}", response_model=SearchResponse)
    def similar(
        pk: int,
        strategy: str = Query(default="clip"),
        k: int = Query(default=default_k, ge=1, le=100),
    ) -> SearchResponse:
        try:
            payload = manager().similar_payload(pk, strategy, clamp(k))
        except UnknownStrategy as exc:
            raise HTTPException(status_code=404, detail=f"Unknown strategy {strategy!r}") from exc
        except StrategyUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc
        except RuntimeError as exc:
            # e.g. an index that does not support reconstruction; not a user error
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SearchResponse(**payload)

    # ------------------------------------------------------------------ #
    # Images
    # ------------------------------------------------------------------ #
    @app.get("/api/image/{pk}")
    def image(pk: int):
        path = manager().image_file(pk)
        if path is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(path, headers=CACHE_HEADERS)

    @app.get("/api/thumbnail/{pk}")
    def thumbnail(pk: int, size: int = Query(default=256, ge=32, le=1024)):
        path = manager().thumbnail(pk, size)
        if path is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(path, media_type="image/jpeg", headers=CACHE_HEADERS)

    @app.get("/api/detail/{pk}", response_model=DetailResponse)
    def detail(pk: int) -> DetailResponse:
        data = manager().detail(pk)
        if data is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return DetailResponse(**data)

    @app.get("/api/random")
    def random_sample(n: int = Query(default=12, ge=1, le=60)) -> dict:
        return {"results": manager().random_sample(n)}

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    @app.get("/api/eval/summary", response_model=EvalSummaryResponse)
    def eval_summary() -> EvalSummaryResponse:
        report = load_report(cfg)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail="No comparison report. Run `python -m tsr compare` first.",
            )
        return EvalSummaryResponse(**report)

    @app.get("/api/eval/report.html")
    def eval_report_html():
        path = reports_dir(cfg) / "compare.html"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="No HTML report. Run `python -m tsr compare` first.",
            )
        return FileResponse(path, media_type="text/html")

    @app.get("/api/eval/failures.html")
    def eval_failures_html():
        path = reports_dir(cfg) / "failures.html"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="No failure report. Run `python -m tsr compare` first.",
            )
        return FileResponse(path, media_type="text/html")

    @app.get("/api/eval/queries")
    def eval_queries() -> dict:
        set_name = str(cfg.eval.get("set_name", "eval"))
        path = eval_dir(cfg) / f"{set_name}.json"
        queries = load_queries(path)
        return {
            "set_name": set_name,
            "path": str(path),
            "count": len(queries),
            "queries": [query.to_dict() for query in queries],
        }

    # ------------------------------------------------------------------ #
    # Settings
    # ------------------------------------------------------------------ #
    def admin_token() -> str:
        return str(cfg.app.get("admin_token") or "")

    @app.get("/api/settings")
    def settings_get() -> dict:
        from tsr.settings import get_settings

        return {"fields": get_settings(cfg), "read_only": bool(admin_token())}

    @app.put("/api/settings")
    def settings_put(updates: dict, request: Request) -> dict:
        from tsr.settings import apply_settings, get_settings, save_settings

        token = admin_token()
        if token and not hmac.compare_digest(request.headers.get("X-Admin-Token", ""), token):
            raise HTTPException(status_code=403, detail="Admin token required")
        try:
            apply_settings(cfg, updates)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        save_settings(cfg)
        return {"fields": get_settings(cfg), "read_only": bool(token)}

    # ------------------------------------------------------------------ #
    # Errors and pages
    # ------------------------------------------------------------------ #
    @app.exception_handler(UnknownStrategy)
    def _unknown(_request: Request, exc: UnknownStrategy):
        return JSONResponse(status_code=404, content={"detail": f"Unknown strategy {exc.args[0]!r}"})

    @app.exception_handler(StrategyUnavailable)
    def _unavailable(_request: Request, exc: StrategyUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    def _missing(_request: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    static_dir = web_dir(cfg)
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    else:  # pragma: no cover
        logger.warning("Web directory not found: %s", static_dir)

    def page(name: str):
        path = static_dir / name
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Page {name} not found")
        return FileResponse(path, media_type="text/html")

    def page_handler(filename: str):
        def handler() -> FileResponse:
            return page(filename)

        return handler

    for route, filename in PAGES.items():
        app.get(route)(page_handler(filename))

    return app


# Module-level app for `uvicorn tsr.api:app`
app = create_app()
