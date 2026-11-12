"""The studies to index, read from NADA's metadata-extract API by their internal id (``sid``, ``surveys.id``).

Indexing never looks a study up by idno: NADA reuses an idno when a study is deleted and created again (a new sid),
and some idnos cannot travel in a URL (``/``, ``:``, spaces). A study is fetched by sid (``studies?sids=``) or paged
from the catalog in id order (``studies?offset=&limit=``), and handed to the pipeline as a :class:`StudySource`: the
record itself, so ai4data's loader neither fetches it nor caches it under its idno.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import ai4data.discovery.catalog.extract as catalog_extract

from nada_ai.settings import Settings

logger = logging.getLogger(__name__)

#: NADA dataset_type (``surveys.type``) -> nada_ai metadata_type. A type not listed here has no ingest path yet: its
#: studies are reported failed rather than guessed at.
DATASET_TYPE_TO_METADATA_TYPE: dict[str, str] = {
    "timeseries": "indicator",
    "timeseriesdb": "indicator-db",
    "timeseries-db": "indicator-db",
    "document": "document",
    "geospatial": "geospatial",
    "survey": "microdata",
    "table": "table",
    "script": "script",
    "image": "image",
    "video": "video",
}

#: Full records (metadata and resources) per extract request: they can be large.
RECORD_BATCH_SIZE = 20


class StudySourceError(ValueError):
    """A study's extract record cannot be indexed (no metadata, no idno, a dataset type with no ingest path)."""


@dataclass(frozen=True)
class StudySource:
    """One study to index: its ids, its ingest type and its catalog metadata (``study_to_catalog_metadata``)."""

    sid: int
    idno: str
    metadata_type: str
    metadata: dict[str, Any]


def metadata_type_of(dataset_type: str | None) -> str | None:
    """The nada_ai metadata_type of a NADA dataset type, or ``None`` when it has no ingest path."""
    return DATASET_TYPE_TO_METADATA_TYPE.get(dataset_type) if dataset_type else None


def source_from_record(record: dict[str, Any]) -> StudySource:
    """The :class:`StudySource` of a full extract record. Raises :class:`StudySourceError`."""
    core = record.get("core_fields") if isinstance(record.get("core_fields"), dict) else {}
    sid = int(core.get("catalog_id") or 0)
    idno = str(core.get("idno") or "").strip()
    if sid <= 0 or not idno:
        raise StudySourceError("the extract record has no core_fields.catalog_id or core_fields.idno")
    filters = record.get("filters") if isinstance(record.get("filters"), dict) else {}
    dataset_type = filters.get("dataset_type")
    metadata_type = metadata_type_of(dataset_type)
    if metadata_type is None:
        raise StudySourceError(f"no ingest path for dataset type {dataset_type!r}")
    try:
        metadata = catalog_extract.study_to_catalog_metadata(record)
    except catalog_extract.CatalogExtractError as e:
        raise StudySourceError(str(e)) from e
    return StudySource(sid=sid, idno=idno, metadata_type=metadata_type, metadata=metadata)


def _extract_base(settings: Settings) -> str:
    base = (settings.metadata_extract_base_url or catalog_extract.extract_base_url() or "").rstrip("/")
    if not base:
        raise StudySourceError("No metadata-extract base URL configured")
    return base


def _flags(include_metadata: bool, include_resources: bool) -> dict[str, str]:
    return {
        "include_metadata": "1" if include_metadata else "0",
        "include_admin_metadata": "1" if include_metadata else "0",
        "include_resources": "1" if include_resources else "0",
    }


def fetch_studies_by_sids(
    settings: Settings, sids: list[int], *, include_metadata: bool = False, include_resources: bool = False
) -> dict[str, Any]:
    """Extract records of these studies in one request: ``{studies, missing, errors}`` (see NADA's
    ``GET search-metadata-extract/studies?sids=``). ``include_metadata=False`` sends core fields and filters only;
    ``include_metadata`` with ``include_resources`` is the full record indexing reads."""
    from nada_ai.ingest.search_index_sync import _client

    if not sids:
        return {"studies": [], "missing": [], "errors": []}
    base = _extract_base(settings)
    with _client(settings) as client:
        resp = client.get(
            f"{base}/studies",
            params={"sids": ",".join(str(i) for i in sids), **_flags(include_metadata, include_resources)},
        )
        resp.raise_for_status()
    data = resp.json()
    return {
        "studies": list(data.get("studies") or []),
        "missing": [int(i) for i in data.get("missing") or []],
        "errors": list(data.get("errors") or []),
    }


def fetch_study_page(settings: Settings, *, offset: int, limit: int, nada_type: str | None) -> dict[str, Any]:
    """One page of the catalog's full records, in id order (``GET search-metadata-extract/studies?offset=&limit=``):
    ``{studies, total, has_more}``. ``nada_type`` narrows to one ``surveys.type``."""
    from nada_ai.ingest.search_index_sync import _client

    params: dict[str, Any] = {"offset": offset, "limit": limit, **_flags(True, True)}
    if nada_type:
        params["type"] = nada_type
    with _client(settings) as client:
        resp = client.get(f"{_extract_base(settings)}/studies", params=params)
        resp.raise_for_status()
    data = resp.json()
    return {
        "studies": list(data.get("studies") or []),
        "total": int(data.get("total") or 0),
        "has_more": bool(data.get("has_more")),
    }


def _load_error(sid: int, idno: str | None, stage: str, error: str) -> dict[str, Any]:
    return {"sid": sid, "idno": idno, "metadata_type": None, "stage": stage, "error": error}


def sources_from_records(
    records: Iterable[dict[str, Any]],
    load_errors: list[dict[str, Any]] | None = None,
    *,
    on_skip: Callable[[int, str], None] | None = None,
) -> Iterator[StudySource]:
    """The :class:`StudySource` of each full record; a record that cannot be indexed goes to ``load_errors``, and to
    ``on_skip(sid, error)``."""
    for record in records:
        try:
            yield source_from_record(record)
        except StudySourceError as e:
            core = record.get("core_fields") if isinstance(record.get("core_fields"), dict) else {}
            sid = int(core.get("catalog_id") or 0)
            logger.warning("Skip study %s: %s", sid, e)
            if load_errors is not None:
                load_errors.append(_load_error(sid, core.get("idno"), "extract", str(e)))
            if on_skip is not None:
                on_skip(sid, str(e))


def sources_by_sids(
    settings: Settings,
    sids: list[int],
    load_errors: list[dict[str, Any]] | None = None,
    *,
    batch_size: int = RECORD_BATCH_SIZE,
) -> Iterator[StudySource]:
    """The studies ``sids``, fetched in batches. A study NADA does not have, or cannot build, goes to ``load_errors``
    (``stage`` ``fetch``)."""
    for start in range(0, len(sids), batch_size):
        batch = fetch_studies_by_sids(
            settings, sids[start : start + batch_size], include_metadata=True, include_resources=True
        )
        for sid in batch["missing"]:
            logger.warning("Skip study %s: not in NADA's catalog", sid)
            if load_errors is not None:
                load_errors.append(_load_error(sid, None, "fetch", "not in NADA's catalog"))
        for err in batch["errors"]:
            sid = int(err.get("catalog_id") or 0)
            logger.warning("Skip study %s: NADA could not build its record: %s", sid, err.get("message"))
            if load_errors is not None:
                load_errors.append(_load_error(sid, None, "fetch", str(err.get("message") or err)))
        yield from sources_from_records(batch["studies"], load_errors)
