"""Live check for step 2: index two studies, delete one by ``sid``, the other is untouched.

Needs a real OpenSearch, e.g.:

  NADA_INTEGRATION_OPENSEARCH=1 NADA_OPENSEARCH_URL=http://localhost:9201 \
      uv run pytest tests/integration/test_sid_live.py -m integration

Uses a throwaway index that is removed afterwards.
"""

from __future__ import annotations

import os
import uuid

import pytest
from langchain_core.documents import Document

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("NADA_INTEGRATION_OPENSEARCH", "").lower() not in ("1", "true", "yes"),
        reason="Set NADA_INTEGRATION_OPENSEARCH=1 and start OpenSearch to run",
    ),
]


def _doc(idno: str, qfield: str) -> Document:
    return Document(page_content=f"{idno} {qfield}", metadata={"type": "microdata", "idno": idno, "qfield": qfield})


def test_delete_by_sid_leaves_other_studies_untouched() -> None:
    from nada_ai.ingest.service import delete_by_sid_op
    from nada_ai.search.backend.opensearch.client import build_client
    from nada_ai.search.backend.opensearch.mapping import index_body
    from nada_ai.search.documents import langdoc_to_source
    from nada_ai.settings import Settings

    settings = Settings(search_backend="opensearch", index_name=f"nada-sid-test-{uuid.uuid4().hex[:8]}")
    client = build_client(settings)
    try:
        client.indices.create(index=settings.index_name, body=index_body(4))
        studies = [(1, "A", ["title", "abstract"]), (2, "B", ["title"]), (3, "C", ["title"])]
        n = 0
        for sid, idno, qfields in studies:
            for qfield in qfields:
                n += 1
                client.index(
                    index=settings.index_name,
                    id=f"doc-{n}",
                    body=langdoc_to_source(_doc(idno, qfield), [float(n), 1.0, 0.0, 0.0], sid=sid),
                )
        client.indices.refresh(index=settings.index_name)

        def count(query: dict) -> int:
            return client.count(index=settings.index_name, body={"query": query})["count"]

        assert count({"term": {"metadata.sid": 1}}) == 2
        assert count({"term": {"metadata.sid": 2}}) == 1
        assert count({"match_all": {}}) == 4

        result = delete_by_sid_op(settings, 1)

        assert result["deleted"] == 2
        assert count({"term": {"metadata.sid": 1}}) == 0
        assert count({"term": {"metadata.sid": 2}}) == 1  # other studies untouched
        assert count({"term": {"metadata.sid": 3}}) == 1
    finally:
        client.indices.delete(index=settings.index_name, ignore_unavailable=True)
        client.transport.close()
