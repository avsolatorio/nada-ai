"""Live check for step 5: browse on the real study index matches an independent oracle.

Needs a real OpenSearch, e.g.:

  NADA_INTEGRATION_OPENSEARCH=1 NADA_OPENSEARCH_URL=http://localhost:9201 \
      uv run pytest tests/integration/test_studies_browse_live.py -m integration

Indexes a handful of synthetic studies through the real writer into throwaway indexes, then compares ``browse`` with a
plain-Python evaluation of the same filters and sorts. No NADA or embedding model is needed.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unicodedata
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from langchain_core.documents import Document

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("NADA_INTEGRATION_OPENSEARCH", "").lower() not in ("1", "true", "yes"),
        reason="Set NADA_INTEGRATION_OPENSEARCH=1 and start OpenSearch to run",
    ),
]

# sid: the fields the extract provides for one study
STUDIES: dict[int, dict[str, Any]] = {
    1: dict(
        idno="A",
        type="survey",
        title="Zebra survey",
        nation="Kenya",
        countries=[404],
        years=[2010, 2011],
        repos=["central"],
        formid=1,
        dc=1,
        tags=["health"],
        author=[7],
        created=100,
        views=5,
    ),
    2: dict(
        idno="B",
        type="survey",
        title="apple census",
        nation="Ethiopia",
        countries=[231, 404],
        years=[2015],
        repos=["central", "demo"],
        formid=2,
        dc=2,
        tags=["health", "poverty"],
        author=[8],
        created=200,
        views=50,
    ),
    3: dict(
        idno="C",
        type="document",
        title="Éclair report",
        nation="France",
        countries=[250],
        years=[2020],
        repos=["demo"],
        formid=1,
        dc=1,
        tags=[],
        author=[7, 8],
        created=300,
        views=50,
    ),
    4: dict(
        idno="D",
        type="timeseries",
        title="",
        nation="Ghana",
        countries=[288],
        years=[2000, 2001, 2002],
        repos=["central"],
        formid=3,
        dc=1,
        tags=["poverty"],
        author=[],
        created=400,
        views=0,
    ),
    5: dict(
        idno="E",
        type="table",
        title="Banana table",
        nation="Ghana",
        countries=[288, 404],
        years=[2010],
        repos=["central"],
        formid=1,
        dc=3,
        tags=["health"],
        author=[9],
        created=500,
        views=7,
    ),
    6: dict(
        idno="F",
        type="survey",
        title="banana survey",
        nation="",
        countries=[404],
        years=[2010],
        repos=["demo"],
        formid=2,
        dc=1,
        tags=[],
        author=[7],
        created=600,
        views=5,
    ),
}


def _fold(text: str) -> str:
    """What the index normalizer does: lowercase and strip accents."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def _raw(sid: int, s: dict[str, Any]) -> dict[str, Any]:
    core = {
        "survey_uid": sid,
        "idno": s["idno"],
        "title": s["title"],
        "nation": s["nation"],
        "year_start": min(s["years"]),
        "year_end": max(s["years"]),
        "created": s["created"],
        "changed": s["created"] + 1,
        "total_views": s["views"],
    }
    filters = {
        "dataset_type": s["type"],
        "published": 1,
        "formid": s["formid"],
        "data_class_id": s["dc"],
        "countries": s["countries"],
        "years": s["years"],
        "repositories": s["repos"],
        "tags": s["tags"],
        "fq_author": s["author"],
    }
    return {"_extract_core_fields": core, "_extract_filters": filters}


# ---- the oracle: the same filters and sorts, evaluated in plain Python -----------------------------------------


def _matches(s: dict[str, Any], sid: int, f: dict[str, Any]) -> bool:
    checks = [
        "countries" not in f or bool(set(f["countries"]) & set(s["countries"])),
        "year_from" not in f or any(y >= f["year_from"] for y in s["years"]),
        "year_to" not in f or any(y <= f["year_to"] for y in s["years"]),
        # year_from and year_to together must hold for ONE year, like a range query on a multi-valued field
        not ("year_from" in f and "year_to" in f) or any(f["year_from"] <= y <= f["year_to"] for y in s["years"]),
        "repository" not in f or f["repository"] in s["repos"],
        "collections" not in f or bool(set(f["collections"]) & set(s["repos"])),
        "form_ids" not in f or s["formid"] in f["form_ids"],
        "data_class_ids" not in f or s["dc"] in f["data_class_ids"],
        "tags" not in f or bool(set(f["tags"]) & set(s["tags"])),
        "facets" not in f or all(set(ids) & set(s[name]) for name, ids in f["facets"].items()),
        "sids" not in f or sid in f["sids"],
        "created_from" not in f or s["created"] >= f["created_from"],
        "created_to" not in f or s["created"] <= f["created_to"],
    ]
    return all(checks)


def _expected(f: dict[str, Any]) -> tuple[list[int], dict[str, int]]:
    """(sids matching all filters incl. types, per-type counts ignoring types)."""
    without_types = {sid: s for sid, s in STUDIES.items() if _matches(s, sid, f)}
    counts: dict[str, int] = {}
    for s in without_types.values():
        counts[s["type"]] = counts.get(s["type"], 0) + 1
    types = f.get("types")
    kept = [sid for sid, s in without_types.items() if not types or s["type"] in types]
    return kept, counts


def _sorted(sids: list[int], by: str, order: str) -> list[int]:
    field = {
        "title": "title",
        "nation": "nation",
        "year": "year",
        "popularity": "views",
        "created": "created",
        "changed": "created",
    }[by]

    def value(sid: int) -> Any:
        s = STUDIES[sid]
        if field == "year":
            return min(s["years"])
        raw = s[field]
        return _fold(raw) if isinstance(raw, str) else raw

    def missing(sid: int) -> bool:
        return field in ("title", "nation") and not STUDIES[sid][field]

    def title(sid: int) -> str:
        return _fold(STUDIES[sid]["title"])

    def tie_break(sid: int) -> tuple[Any, ...]:
        # year_start desc, title asc (missing last), sid asc
        return (-min(STUDIES[sid]["years"]), not STUDIES[sid]["title"], title(sid), sid)

    present = [sid for sid in sids if not missing(sid)]
    absent = [sid for sid in sids if missing(sid)]
    present.sort(key=tie_break)
    present.sort(key=value, reverse=order == "desc")  # stable: ties keep tie_break order
    absent.sort(key=tie_break)
    return present + absent


FILTER_CASES: list[dict[str, Any]] = [
    {},
    {"types": ["survey"]},
    {"types": ["survey", "table"]},
    {"types": ["nothing"]},
    {"countries": [404]},
    {"countries": [404, 250]},
    {"countries": [999]},
    {"year_from": 2010},
    {"year_to": 2010},
    {"year_from": 2010, "year_to": 2011},
    {"year_from": 2003, "year_to": 2009},  # a gap: no study has a year inside it
    {"repository": "demo"},
    {"collections": ["demo"]},
    {"repository": "central", "collections": ["demo"]},
    {"form_ids": [1]},
    {"form_ids": [1, 2], "data_class_ids": [1]},
    {"tags": ["poverty"]},
    {"facets": {"author": [7]}},
    {"facets": {"author": [7, 8]}},
    {"sids": [1, 3, 5, 99]},
    {"created_from": 200, "created_to": 500},
    {"created_from": 550},
    {"types": ["survey"], "countries": [404], "year_from": 2010, "form_ids": [2]},
]


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_browse_matches_the_oracle_on_the_real_index() -> None:
    from nada_ai.app.studies_schemas import SortField, SortOrder, StudyFilters
    from nada_ai.ingest import pipeline
    from nada_ai.ingest.opensearch_writer import OpenSearchIngestWriter
    from nada_ai.search.backend.opensearch.client import build_async_client, build_client
    from nada_ai.search.backend.opensearch.index_template import (
        composable_index_template_name,
        studies_index_template_name,
    )
    from nada_ai.search.backend.opensearch.studies_search import SearchJob, browse
    from nada_ai.search.backend.opensearch.studies_semantic import StudyPolicy
    from nada_ai.settings import Settings

    settings = Settings(
        search_backend="opensearch",
        index_name=f"nada-browse-{uuid.uuid4().hex[:8]}",
        dynamic_filter_facets_path=os.path.join(tempfile.mkdtemp(), "facets.json"),
    )
    by_idno = {s["idno"]: (sid, s) for sid, s in STUDIES.items()}

    def loader(idno: str, metadata_type: str, force: bool = False, include_resources: bool = True):
        sid, s = by_idno[idno]
        docs = [Document(page_content=f"{idno} text", metadata={"type": "microdata", "idno": idno, "qfield": "title"})]
        handler = SimpleNamespace(get_langdocs=lambda: docs)
        return SimpleNamespace(metadata=_raw(sid, s), get_metadata_handler=lambda: handler)

    class Embedding:
        def embedding_dimension(self) -> int:
            return 4

        def encode_corpus(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
            return np.array([[float(len(t)), 1.0, 0.0, 0.0] for t in texts])

    sync = build_client(settings)
    try:
        with patch.object(pipeline, "MetadataLoader", loader):
            _, errors = OpenSearchIngestWriter(settings).run_bulk(
                [(s["idno"], "microdata") for s in STUDIES.values()],
                embedding=Embedding(),  # type: ignore[arg-type]
                recreate_target=True,
                show_progress_bar=False,
            )
        assert errors is None

        async def scenario() -> None:
            client = build_async_client(settings)
            try:
                index = settings.studies_index

                async def run(
                    filters: dict[str, Any], by: str = "title", order: str = "asc", limit: int = 100, offset: int = 0
                ):
                    job = SearchJob(
                        client=client,
                        index=index,
                        chunk_index=settings.index_name,
                        policy=StudyPolicy.from_settings(settings),
                        query=None,
                        filters=StudyFilters.model_validate(filters),
                        sort_by=SortField(by),
                        sort_order=SortOrder(order),
                        limit=limit,
                        offset=offset,
                        result_cap=100,
                    )
                    return await browse(job)

                # every filter combination: found, ids and tab counts equal the oracle
                for filters in FILTER_CASES:
                    page = await run(filters)
                    kept, counts = _expected(filters)
                    assert sorted(h["sid"] for h in page.hits) == sorted(kept), filters
                    assert page.found == len(kept), filters
                    assert page.counts_by_type == counts, filters
                    assert all(h["idno"] == STUDIES[h["sid"]]["idno"] for h in page.hits)

                # every sort, both directions: same order as the oracle, ties and missing values included
                for by in ("title", "nation", "year", "popularity", "created", "changed"):
                    for order in ("asc", "desc"):
                        page = await run({}, by, order)
                        assert [h["sid"] for h in page.hits] == _sorted(list(STUDIES), by, order), (by, order)

                # paging: pages of two walk the whole set exactly once, and repeating a page returns the same ids
                for by, order in (("title", "asc"), ("popularity", "desc"), ("year", "asc")):
                    walked: list[int] = []
                    for offset in range(0, len(STUDIES) + 2, 2):
                        page = await run({}, by, order, limit=2, offset=offset)
                        again = await run({}, by, order, limit=2, offset=offset)
                        assert [h["sid"] for h in page.hits] == [h["sid"] for h in again.hits]
                        assert page.found == len(STUDIES)
                        walked += [h["sid"] for h in page.hits]
                    assert walked == _sorted(list(STUDIES), by, order), (by, order)

                # a window past the end is empty but still reports the total
                page = await run({}, limit=5, offset=50)
                assert (page.hits, page.found) == ([], len(STUDIES))
            finally:
                await client.close()

        _run(scenario())
    finally:
        for index in (settings.index_name, settings.studies_index):
            sync.indices.delete(index=index, ignore_unavailable=True)
        for name in (composable_index_template_name(settings), studies_index_template_name(settings)):
            try:
                sync.indices.delete_index_template(name=name)
            except Exception:
                pass
        sync.transport.close()
