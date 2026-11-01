"""Pydantic response schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Metaphor(BaseModel):
    metaphor: str = ""
    meaning: str = ""


class VlmSummary(BaseModel):
    scene: str | None = None
    ocr_text: str | None = None
    meaning: str | None = None
    model: str | None = None


class SearchResult(BaseModel):
    rank: int
    id: int
    image_id: str
    score: float
    score_components: dict[str, float] = Field(default_factory=dict)
    url: str
    thumbnail: str | None = None
    title: str | None = None
    img_captions: list[str] = Field(default_factory=list)
    meme_captions: list[str] = Field(default_factory=list)
    metaphors: list[Metaphor | dict] = Field(default_factory=list)
    category: str | None = None
    split: str | None = None
    width: int | None = None
    height: int | None = None
    source_url: str | None = None
    vlm: VlmSummary | dict | None = None


class SearchResponse(BaseModel):
    query: str
    strategy: str | None = None
    ready: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    expanded_queries: list[str] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)
    count: int
    results: list[SearchResult]


class StrategyStatus(BaseModel):
    key: str
    title: str = ""
    description: str = ""
    ready: bool = False
    error: str | None = None
    num_images: int | None = None
    num_docs: int | None = None
    coverage: float | None = None
    dim: int | None = None
    model: str | None = None
    built_at: str | None = None
    meta: dict = Field(default_factory=dict)


class CompareResponse(BaseModel):
    query: str
    top_k: int
    columns: dict[str, SearchResponse] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)


class DetailResponse(BaseModel):
    id: int
    image_id: str
    title: str | None = None
    split: str | None = None
    category: str | None = None
    width: int | None = None
    height: int | None = None
    source_url: str | None = None
    post_id: str | None = None
    url: str
    thumbnail: str | None = None
    img_captions: list[str] = Field(default_factory=list)
    meme_captions: list[str] = Field(default_factory=list)
    metaphors: list[Metaphor | dict] = Field(default_factory=list)
    vlm: dict | None = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    store_error: str | None = None
    num_images: int = 0
    strategies: list[dict] = Field(default_factory=list)


class StatsResponse(BaseModel):
    dataset: dict = Field(default_factory=dict)
    models: dict = Field(default_factory=dict)
    index: dict = Field(default_factory=dict)
    strategies: list[dict] = Field(default_factory=list)
    store_error: str | None = None


class EvalSummaryResponse(BaseModel):
    generated_at: str | None = None
    eval_set: str | None = None
    num_queries: int = 0
    top_k: list[int] = Field(default_factory=list)
    n_boot: int = 0
    seed: int | None = None
    strategies: dict[str, dict] = Field(default_factory=dict)
    pairwise: list[dict] = Field(default_factory=list)
    ablations: dict[str, dict] = Field(default_factory=dict)
    diagnostics: dict[str, dict] = Field(default_factory=dict)
    probe: dict = Field(default_factory=dict)


class SettingsResponse(BaseModel):
    fields: list[dict]
