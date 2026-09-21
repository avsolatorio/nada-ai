"""Live check for step 7: semantic and hybrid search, and re-indexing, on real OpenSearch.

Needs a real OpenSearch, e.g.:

  NADA_INTEGRATION_OPENSEARCH=1 NADA_OPENSEARCH_URL=http://localhost:9201 \
      uv run pytest tests/integration/test_studies_hybrid_live.py -m integration

The embedding is a small deterministic stand-in: words of one concept (wage, salary, earnings, pay, income) share a
direction, other words add a little noise, so which studies are "semantically" close to a query is exactly known.
Everything else (indexes, the vector search, collapse, filters, fusion) is the real code on a real cluster.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import uuid
import zlib
from dataclasses import replace
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

CONCEPTS = {
    "wage": 0, "wages": 0, "salary": 0, "salaries": 0, "earnings": 0, "pay": 0, "income": 0,
    "farm": 1, "farming": 1, "farmer": 1, "cattle": 1, "poultry": 1, "crops": 1,
    "school": 2, "students": 2, "education": 2,
    "hospital": 3, "health": 3,
}  # fmt: skip
NOISE_DIMS = 256  # many, so unrelated texts are nearly orthogonal, as real embeddings roughly are
DIM = 4 + NOISE_DIMS


def embed_text(text: str) -> np.ndarray:
    """Concept words point along their concept's axis; every other word adds a little noise; unit length."""
    vector = np.zeros(DIM)
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in CONCEPTS:
            vector[CONCEPTS[word]] += 1.0
        else:
            vector[4 + zlib.crc32(word.encode()) % NOISE_DIMS] += 0.02
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class Embedding:
    def embedding_dimension(self) -> int:
        return DIM

    def encode_corpus(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return np.array([embed_text(t) for t in texts])


async def embed_query(text: str) -> list[float]:
    return [float(x) for x in embed_text(text)]


# sid -> the study as NADA's extract describes it, and the chunks (qfield, text, doc_meta) it produces
STUDIES: dict[int, dict[str, Any]] = {
    1: dict(type="survey", title="Wages and salary survey", countries=[404], created=100,
            chunks=[("title", "Wages and salary survey", None), ("abstract", "Household survey of wages", None)]),
    2: dict(type="survey", title="Income and earnings report", countries=[404], created=200,
            chunks=[("title", "Income and earnings report", None)]),
    3: dict(type="survey", title="Poultry farming census", countries=[356], created=300,
            chunks=[("title", "Poultry farming census farmer", None)]),
    4: dict(type="document", title="Labour market document", countries=[250], created=400,
            chunks=[("title", "Labour market document", None),
                    ("passages", "average pay by sector", {"page": 2, "total_pages": 10}),
                    ("passages", "pay gaps widened", {"page": 5, "total_pages": 10})]),
    5: dict(type="table", title="Salary tables", countries=[250], created=500,
            chunks=[("title", "Salary tables", None)]),
    # the word "salary" is in the title (a keyword match) but the text is mostly about farming: far from the query
    6: dict(type="survey", title="Salary of civil servants", countries=[356], created=600,
            chunks=[("title", "Salary of civil servants and unrelated farm farm farm", None)]),
}  # fmt: skip


def raw_for(sid: int, s: dict[str, Any]) -> dict[str, Any]:
    return {
        "_extract_core_fields": {
            "survey_uid": sid,
            "idno": f"NADA_{sid}",
            "title": s["title"],
            "nation": "Somewhere",
            "year_start": 2010 + sid,
            "year_end": 2010 + sid,
            "created": s["created"],
            "changed": s["created"],
            "total_views": sid,
        },
        "_extract_filters": {
            "dataset_type": s["type"],
            "published": 1,
            "countries": s["countries"],
            "years": [2010 + sid],
            "repositories": ["central"],
        },
    }


def documents_for(sid: int, s: dict[str, Any]) -> list[Document]:
    docs = []
    for qfield, text, doc_meta in s["chunks"]:
        # the chunk idno is the record's schema idno, which differs from NADA's on purpose
        meta: dict[str, Any] = {"type": "microdata", "idno": f"schema_{sid}", "qfield": qfield}
        if doc_meta:
            meta["doc_meta"] = doc_meta
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def test_semantic_and_hybrid_search_on_the_real_index() -> None:
    from nada_ai.app.studies_schemas import SortField, SortOrder, StudyFilters
    from nada_ai.ingest import pipeline
    from nada_ai.ingest.opensearch_writer import OpenSearchIngestWriter
    from nada_ai.search.backend.opensearch.client import build_async_client, build_client
    from nada_ai.search.backend.opensearch.index_template import (
        composable_index_template_name,
        studies_index_template_name,
    )
    from nada_ai.search.backend.opensearch.studies_search import SearchJob, hybrid, lexical, semantic
    from nada_ai.search.backend.opensearch.studies_semantic import StudyPolicy
    from nada_ai.settings import Settings

    settings = Settings(
        search_backend="opensearch",
        index_name=f"nada-hyb-{uuid.uuid4().hex[:8]}",
        dynamic_filter_facets_path=os.path.join(tempfile.mkdtemp(), "facets.json"),
        # the synthetic vectors and expectations below are built for these thresholds, not for the tuned defaults
        studies_semantic_min_score=0.68,
        studies_semantic_relative_cutoff=0.9,
    )
    studies = dict(STUDIES)

    def loader(idno: str, metadata_type: str, force: bool = False, include_resources: bool = True):
        sid = int(idno.removeprefix("NADA_"))
        handler = SimpleNamespace(get_langdocs=lambda: documents_for(sid, studies[sid]))
        return SimpleNamespace(metadata=raw_for(sid, studies[sid]), get_metadata_handler=lambda: handler)

    def ingest(writer: Any, recreate: bool) -> None:
        with patch.object(pipeline, "MetadataLoader", loader):
            _, errors = writer.run_bulk(
                [(f"NADA_{sid}", "microdata") for sid in studies],
                embedding=Embedding(),
                recreate_target=recreate,
                show_progress_bar=False,
            )
        assert errors is None

    sync = build_client(settings)
    try:
        writer = OpenSearchIngestWriter(settings)
        ingest(writer, recreate=True)

        async def scenario() -> None:
            client = build_async_client(settings)
            try:
                policy = StudyPolicy.from_settings(settings)

                def job(query: str, **overrides: Any) -> SearchJob:
                    fields: dict[str, Any] = dict(
                        client=client, index=settings.studies_index, chunk_index=settings.index_name, query=query,
                        filters=StudyFilters(), sort_by=SortField.relevance, sort_order=SortOrder.desc, limit=100,
                        offset=0, policy=policy, embed=embed_query,
                    )  # fmt: skip
                    fields.update(overrides)
                    return SearchJob(**fields)

                def ids(page: Any) -> list[int]:
                    return [h["sid"] for h in page.hits]

                def by_sid(page: Any) -> dict[int, dict[str, Any]]:
                    return {h["sid"]: h for h in page.hits}

                # the legs on their own
                assert set(ids(await lexical(job("salary")))) == {1, 5, 6}
                sem = await semantic(job("salary"))
                assert set(ids(sem)) == {1, 2, 4, 5}  # 6 is mostly about farming: below the floor
                assert all(h["matched_by"] == ["semantic"] for h in sem.hits)
                assert by_sid(sem)[2]["idno"] == "NADA_2"  # NADA's idno, not the chunk's schema idno

                # hybrid: the semantic studies fused with the best keyword matches, then every other keyword match; each
                # study says what found it
                page = await hybrid(job("salary"))
                found = by_sid(page)
                assert set(found) == {1, 2, 4, 5, 6}
                assert found[1]["matched_by"] == found[5]["matched_by"] == ["lexical", "semantic"]
                assert found[6]["matched_by"] == ["lexical"]
                assert found[2]["matched_by"] == found[4]["matched_by"] == ["semantic"]
                order = ids(page)
                assert set(order[:2]) == {1, 5} and set(order[2:]) == {2, 4, 6}  # both legs first, then the rest
                assert {found[2]["idno"], found[4]["idno"]} == {"NADA_2", "NADA_4"}
                assert page.found == 5
                assert page.counts_by_type == {"survey": 3, "table": 1, "document": 1}

                # a document study returns the pages that matched, best first
                passages = found[4]["passages"]
                assert {p["page"] for p in passages} == {3, 6}
                assert [p["score"] for p in passages] == sorted((p["score"] for p in passages), reverse=True)
                assert all(p["total_pages"] == 10 and p["excerpt"] for p in passages)
                assert "passages" not in found[1]

                # a concept no keyword mentions is found only by the vector leg
                assert ids(await lexical(job("cattle"))) == []
                cattle = await hybrid(job("cattle"))
                # study 3 is all about farming; study 6's text is mostly farming too, so it follows
                assert ids(cattle) == [3, 6]
                assert all(h["matched_by"] == ["semantic"] for h in cattle.hits)

                # gibberish is far from everything: no keyword match, nothing above the floor
                nothing = await hybrid(job("xyzzy qwerty flurbo"))
                assert (nothing.found, nothing.hits, nothing.counts_by_type) == (0, [], {})

                # filters reach both legs: countries (a filter field) and created (stamped on chunks at ingest)
                assert set(ids(await hybrid(job("salary", filters=StudyFilters(countries=[250]))))) == {4, 5}
                assert set(ids(await hybrid(job("salary", filters=StudyFilters(created_from=300))))) == {4, 5, 6}
                assert set(ids(await semantic(job("salary", filters=StudyFilters(created_from=300))))) == {4, 5}

                # `types` narrows found but not the tab counts, which describe every keyword match plus the block
                tab = await hybrid(job("salary", filters=StudyFilters(types=["table"])))
                assert (ids(tab), tab.found) == ([5], 1)
                assert tab.counts_by_type == {"survey": 3, "table": 1, "document": 1}

                # nothing is cut, and the pages join up (the fused head spans the first page)
                pages = [await hybrid(job("salary", limit=2, offset=o)) for o in (0, 2, 4)]
                assert [i for p in pages for i in ids(p)] == order and {p.found for p in pages} == {5}
                assert ids(await hybrid(job("salary", sort_order=SortOrder.asc))) == order[::-1]

                # any other sort re-sorts the same set (title, case-insensitive)
                by_title = await hybrid(job("salary", sort_by=SortField.title, sort_order=SortOrder.asc))
                assert ids(by_title) == [2, 4, 6, 5, 1]
                assert by_sid(by_title)[2]["matched_by"] == ["semantic"]  # each hit keeps what found it

                # the policy is the knob: a floor of 1.0 leaves only the keyword matches
                strict = replace(policy, semantic_min_score=1.0)
                assert set(ids(await hybrid(job("salary", policy=strict)))) == {1, 5, 6}
                # a very tight relative cutoff keeps only the studies that tie for the best vector score
                tight = replace(policy, semantic_relative_cutoff=0.99999)
                assert set(ids(await semantic(job("salary", policy=tight)))) <= {1, 2, 4, 5}
            finally:
                await client.close()

        asyncio.run(scenario())

        # re-indexing a study whose text changed replaces its chunks: nothing stale is left behind
        def chunks_of(sid: int) -> int:
            sync.indices.refresh(index=settings.index_name)
            return sync.count(index=settings.index_name, body={"query": {"term": {"metadata.sid": sid}}})["count"]

        assert chunks_of(1) == 2
        studies[1] = {**STUDIES[1], "chunks": [("title", "Completely rewritten title about hospital health", None)]}
        ingest(writer, recreate=False)
        assert chunks_of(1) == 1
        old = sync.count(
            index=settings.index_name, body={"query": {"match_phrase": {"page_content": "Household survey of wages"}}}
        )["count"]
        assert old == 0
        assert chunks_of(2) == 1 and chunks_of(4) == 3  # the untouched studies keep theirs

        # a study that now produces no chunks loses all of them
        studies[2] = {**STUDIES[2], "chunks": []}
        ingest(writer, recreate=False)
        assert chunks_of(2) == 0
        assert (
            sync.count(index=settings.studies_index, body={"query": {"term": {"sid": 2}}})["count"] == 1
        )  # still a study
    finally:
        for index in (settings.index_name, settings.studies_index):
            sync.indices.delete(index=index, ignore_unavailable=True)
        for name in (composable_index_template_name(settings), studies_index_template_name(settings)):
            try:
                sync.indices.delete_index_template(name=name)
            except Exception:
                pass
        sync.transport.close()
