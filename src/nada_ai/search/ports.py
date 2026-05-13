"""Search backend port (OpenSearch, Qdrant, …) — normalized request/response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

SearchMode = Literal["keyword", "vector", "hybrid"]
VectorStrategy = Literal["mean", "max_pool"]


@dataclass(kw_only=True)
class SearchParams:
    query: str
    mode: SearchMode
    query_vector: list[float] | None
    filters: dict[str, Any] | None
    size: int
    from_: int
    knn_k: int
    collapse_field: str | None
    collapse_inner_hits: dict[str, Any] | None
    include_embedding: bool
    include_facets: bool
    facet_fields: list[str] | None
    use_idno_fast_path: bool
    vector_score_threshold: float | None = None


@dataclass(kw_only=True)
class RecommendParams:
    idno: str
    size: int
    filters: dict[str, Any] | None
    exclude_idno: bool
    vector_strategy: VectorStrategy
    knn_k: int
    include_facets: bool
    facet_fields: list[str] | None
    vector_score_threshold: float | None = None


@dataclass
class SearchOutcome:
    total: int | None
    hits: list[dict[str, Any]]
    facets: dict[str, list[dict[str, Any]]] | None = None
    debug_request: dict[str, Any] | None = None


@runtime_checkable
class SearchBackendPort(Protocol):
    async def health(self) -> dict[str, Any]:
        """Backend-specific health payload (cluster status, collection name, …)."""

    async def search(self, params: SearchParams) -> SearchOutcome:
        """Execute search; ``query_vector`` may be None for keyword-only or ML neural."""

    async def recommend_by_idno(self, params: RecommendParams) -> SearchOutcome:
        """Similar-catalog recommendations from stored embeddings for ``idno``."""

    async def explain_by_idno(self, idno: str, filters: dict[str, Any] | None) -> dict[str, Any]:
        """Structured match info for a catalog idno (no LLM)."""

