"""Pydantic models for admin/ingest/job endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateIndexRequest(BaseModel):
    recreate: bool = Field(default=False, description="Drop and recreate the index if it already exists.")


class IndexByIdsRequest(BaseModel):
    idnos: list[str] = Field(..., min_length=1)
    metadata_type: str = Field(default="indicator", description="indicator | document | microdata | geospatial")
    force: bool = Field(default=False, description="Bypass MetadataLoader cache.")
    recreate_index: bool = Field(default=False, description="Drop and recreate the index before bulk indexing.")
    show_progress_bar: bool = Field(default=False, description="tqdm bars in API are usually noise; default off.")
    buffer_size: int = Field(default=1000, ge=1, le=10000)


class IndexFromCatalogRequest(BaseModel):
    catalog_type: str = Field(default="timeseries", description="timeseries | indicator | document | microdata | survey | geospatial")
    ps: int = Field(default=100, ge=1, le=1000, description="Catalog page size.")
    limit: int | None = Field(default=None, ge=1, description="Stop after first N catalog rows; None means no limit.")
    force: bool = Field(default=False)
    recreate_index: bool = Field(default=False)
    show_progress_bar: bool = Field(default=False)
    buffer_size: int = Field(default=1000, ge=1, le=10000)


class JobResponse(BaseModel):
    id: str
    kind: str
    key: str
    params: dict[str, Any]
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    result: dict[str, Any] | None
    error: str | None
    progress: dict[str, Any]


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


class EncodeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)
    as_query: bool = Field(default=True, description="Use encode_query (asymmetric prompt) instead of encode_corpus.")


class EncodeResponse(BaseModel):
    model_id: str
    dimension: int
    as_query: bool
    vectors: list[list[float]]


class IndexStatsResponse(BaseModel):
    index: str
    docs: int | None
    size_bytes: int | None
    primaries: dict[str, Any] | None
    raw: dict[str, Any] | None = None


class DeleteDocsResponse(BaseModel):
    index: str
    deleted: int
    matched: int | None
    raw: dict[str, Any] | None = None
