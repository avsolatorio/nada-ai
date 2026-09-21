"""Live check for step 3: one ingest job writes the study index and the chunk index with flat filter fields.

Needs a real OpenSearch, e.g.:

  NADA_INTEGRATION_OPENSEARCH=1 NADA_OPENSEARCH_URL=http://localhost:9201 \
      uv run pytest tests/integration/test_opensearch_indexes_live.py -m integration

Uses throwaway indexes and templates that are removed afterwards. No NADA or embedding model is needed.
"""

from __future__ import annotations

import os
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


class _Embedding:
    def embedding_dimension(self) -> int:
        return 4

    def encode_corpus(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        return np.array([[float(len(t)), 1.0, 0.0, 0.0] for t in texts])


def _chunk(idno: str, qfield: str) -> Document:
    # the chunk idno comes from the record's schema; for study "B" it differs from NADA's idno on purpose
    return Document(
        page_content=f"{idno} {qfield} text", metadata={"type": "microdata", "idno": idno, "qfield": qfield}
    )


def _raw(sid: int, idno: str, countries: list[int], years: list[int]) -> dict[str, Any]:
    return {
        "_extract_core_fields": {
            "survey_uid": sid,
            "idno": idno,
            "title": f"Study {idno}",
            "nation": "Somewhere",
            "abstract": "About things",
            "year_start": min(years),
            "year_end": max(years),
            "created": 1700000000 + sid,
            "changed": 1700000500 + sid,
            "total_views": sid,
            "varcount": 10,
        },
        "_extract_filters": {
            "dataset_type": "survey",
            "published": 1,
            "formid": 1,
            "countries": countries,
            "years": years,
            "repositories": ["central"],
            "tags": [],
            "fq_author": [7, 8],
        },
    }


STUDIES = {
    # nada idno -> (sid, chunks (idno stored on chunk, qfield), countries, years)
    "A": (1, [("A", "title"), ("A", "abstract")], [16], [2010, 2011]),
    "B": (2, [("B-schema", "title")], [16, 191], [2015]),
    "C": (3, [], [102], [2020]),  # no chunk documents at all
}


def test_one_ingest_job_writes_both_indexes_with_flat_filters() -> None:
    import tempfile

    from nada_ai.ingest import pipeline
    from nada_ai.ingest.opensearch_writer import OpenSearchIngestWriter
    from nada_ai.ingest.service import delete_by_sids_op
    from nada_ai.search.backend.opensearch.client import build_client
    from nada_ai.search.backend.opensearch.index_template import (
        composable_index_template_name,
        studies_index_template_name,
    )
    from nada_ai.settings import Settings

    settings = Settings(
        search_backend="opensearch",
        index_name=f"nada-live-{uuid.uuid4().hex[:8]}",
        dynamic_filter_facets_path=os.path.join(tempfile.mkdtemp(), "facets.json"),
    )
    chunks_index, studies_index = settings.index_name, settings.studies_index

    def loader(idno: str, metadata_type: str, force: bool = False, include_resources: bool = True):
        sid, chunks, countries, years = STUDIES[idno]
        docs = [_chunk(chunk_idno, qfield) for chunk_idno, qfield in chunks]
        handler = SimpleNamespace(get_langdocs=lambda: docs)
        return SimpleNamespace(metadata=_raw(sid, idno, countries, years), get_metadata_handler=lambda: handler)

    client = build_client(settings)
    try:
        writer = OpenSearchIngestWriter(settings)
        pairs = [(idno, "microdata") for idno in STUDIES]

        def ingest(recreate: bool) -> tuple[int, Any]:
            with patch.object(pipeline, "MetadataLoader", loader):
                return writer.run_bulk(
                    pairs, embedding=_Embedding(), recreate_target=recreate, show_progress_bar=False  # type: ignore[arg-type]
                )

        written, errors = ingest(recreate=True)
        assert errors is None
        assert written == 3  # chunk documents

        def count(index: str, query: dict | None = None) -> int:
            return client.count(index=index, body={"query": query or {"match_all": {}}})["count"]

        def lucene_docs(index: str) -> int:
            return client.indices.stats(index=index)["_all"]["primaries"]["docs"]["count"]

        # counts: one study document per study, and no hidden nested documents inflating either index
        assert count(studies_index) == 3
        assert count(chunks_index) == 3
        assert lucene_docs(studies_index) == 3
        assert lucene_docs(chunks_index) == 3

        # the study document: _id is the sid, idno is NADA's, filters are flat and typed
        study = client.get(index=studies_index, id="2")["_source"]
        assert (study["sid"], study["idno"]) == (2, "B")
        assert study["filter_facets"]["countries"] == ["16", "191"]
        study_props = client.indices.get_mapping(index=studies_index)[studies_index]["mappings"]["properties"]
        assert study_props["filter_facets"]["properties"]["fq_author"]["type"] == "integer"  # dynamic template
        assert study_props["filter_facets"]["properties"]["countries"]["type"] == "integer"
        assert "embedding" not in study_props  # the chunk template did not leak onto the study index

        # typed flat filters work on both indexes
        assert count(studies_index, {"terms": {"filter_facets.countries": [16]}}) == 2
        assert count(studies_index, {"range": {"filter_facets.years": {"gte": 2012}}}) == 2
        assert count(studies_index, {"term": {"filter_facets.fq_author": 7}}) == 3
        assert count(chunks_index, {"terms": {"metadata.filter_facets.countries": [16]}}) == 3
        assert count(chunks_index, {"term": {"metadata.sid": 2}}) == 1

        # both indexes carry a generation; the chunk index also records the embedding it was built for
        chunk_meta = client.indices.get_mapping(index=chunks_index)[chunks_index]["mappings"]["_meta"]
        study_meta = client.indices.get_mapping(index=studies_index)[studies_index]["mappings"]["_meta"]
        assert chunk_meta["generation"] and study_meta["generation"]
        assert chunk_meta["embedding_dim"] == 4

        # a second run replaces study documents and adds nothing
        written, errors = ingest(recreate=False)
        assert errors is None
        assert (count(studies_index), count(chunks_index)) == (3, 3)

        # recreate is a new generation
        ingest(recreate=True)
        new_meta = client.indices.get_mapping(index=studies_index)[studies_index]["mappings"]["_meta"]
        assert new_meta["generation"] != study_meta["generation"]

        # deleting by sid empties the study from both indexes and leaves the others
        result = delete_by_sids_op(settings, [1])
        assert result["deleted"] == 2 and result["studies_deleted"] == 1
        assert (count(studies_index), count(chunks_index)) == (2, 1)
        assert count(studies_index, {"term": {"sid": 1}}) == 0
    finally:
        for index in (chunks_index, studies_index):
            client.indices.delete(index=index, ignore_unavailable=True)
        for name in (composable_index_template_name(settings), studies_index_template_name(settings)):
            try:
                client.indices.delete_index_template(name=name)
            except Exception:
                pass
        client.transport.close()
