"""Live check for step 6: keyword search on the real study index.

Needs a real OpenSearch, e.g.:

  NADA_INTEGRATION_OPENSEARCH=1 NADA_OPENSEARCH_URL=http://localhost:9201 \
      uv run pytest tests/integration/test_studies_lexical_live.py -m integration

Indexes a few synthetic studies through the real writer into throwaway indexes. No NADA or embedding model is needed.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
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

STUDIES: dict[int, dict[str, Any]] = {
    1: dict(
        idno="EGY_2014_DHS_v01_M",
        type="survey",
        title="Demographic and Health Survey 2014",
        nation="Egypt",
        authoring="CAPMAS",
        keywords="fertility mortality",
        abstract="Household survey on population health",
        countries=[818],
    ),
    2: dict(
        idno="KEN_2018_LFS",
        type="survey",
        title="Labour Force Survey 2018",
        nation="Kenya",
        authoring="KNBS",
        keywords="employment",
        abstract="Employment and unemployment",
        countries=[404],
    ),
    3: dict(
        idno="consumer-price-index",
        type="timeseries",
        title="Consumer Price Index",
        nation="World",
        authoring="IMF",
        keywords="inflation",
        abstract="Monthly price data for goods",
        countries=[900],
    ),
    4: dict(
        idno="RPT_2020_EDU",
        type="document",
        title="Éclair education report",
        nation="France",
        authoring="Ministry",
        keywords="school",
        abstract="This report discusses consumer prices and school",
        countries=[250],
    ),
    5: dict(
        idno="GEO_2015_MAP",
        type="geospatial",
        title="Poverty map",
        nation="Ghana",
        authoring="WB",
        keywords="map",
        abstract="Poverty estimates from a survey and satellite images",
        countries=[288],
    ),
    6: dict(
        idno="TBL_2011",
        type="table",
        title="Census tables 2011",
        nation="India",
        authoring="ORGI",
        keywords="census",
        abstract="A survey of census counts",
        countries=[356],
    ),
    7: dict(
        idno="OTHER_1",
        type="survey",
        title="Household expenditure",
        nation="Peru",
        authoring="INEI",
        keywords="expenditure",
        abstract="Price survey of households",
        countries=[604],
    ),
}


def _raw(sid: int, s: dict[str, Any]) -> dict[str, Any]:
    return {
        "_extract_core_fields": {
            "survey_uid": sid,
            "idno": s["idno"],
            "title": s["title"],
            "nation": s["nation"],
            "authoring_entity": s["authoring"],
            "keywords": s["keywords"],
            "abstract": s["abstract"],
            "year_start": 2010 + sid,
            "year_end": 2010 + sid,
            "created": 1000 + sid,
            "changed": 1000 + sid,
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


def test_keyword_search_on_the_real_index() -> None:
    from nada_ai.app.studies_schemas import SortField, SortOrder, StudyFilters
    from nada_ai.ingest import pipeline
    from nada_ai.ingest.opensearch_writer import OpenSearchIngestWriter
    from nada_ai.search.backend.opensearch.client import build_async_client, build_client
    from nada_ai.search.backend.opensearch.index_template import (
        composable_index_template_name,
        studies_index_template_name,
    )
    from nada_ai.search.backend.opensearch.studies_search import SearchJob, lexical
    from nada_ai.search.backend.opensearch.studies_semantic import StudyPolicy
    from nada_ai.settings import Settings

    settings = Settings(
        search_backend="opensearch",
        index_name=f"nada-lex-{uuid.uuid4().hex[:8]}",
        dynamic_filter_facets_path=os.path.join(tempfile.mkdtemp(), "facets.json"),
        studies_lexical_relative_cutoff=0.0,  # these checks are about which studies match, not about the weak tail
    )
    by_idno = {s["idno"]: (sid, s) for sid, s in STUDIES.items()}

    def loader(idno: str, metadata_type: str, force: bool = False, include_resources: bool = True):
        sid, s = by_idno[idno]
        docs = [Document(page_content=f"{idno}", metadata={"type": "microdata", "idno": idno, "qfield": "title"})]
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

                async def search(
                    query: str,
                    *,
                    filters: dict[str, Any] | None = None,
                    by: str = "relevance",
                    order: str = "desc",
                    limit: int = 100,
                    offset: int = 0,
                ):
                    return await lexical(
                        SearchJob(
                            client=client,
                            index=settings.studies_index,
                            chunk_index=settings.index_name,
                            policy=StudyPolicy.from_settings(settings),
                            query=query,
                            filters=StudyFilters.model_validate(filters or {}),
                            sort_by=SortField(by),
                            sort_order=SortOrder(order),
                            limit=limit,
                            offset=offset,
                        )
                    )

                def ids(page: Any) -> list[int]:
                    return [h["sid"] for h in page.hits]

                # nothing matches gibberish
                page = await search("xyzzy qwerty flurbo")
                assert (page.found, page.hits, page.counts_by_type) == (0, [], {})

                # a distinctive title word finds its study, whatever the case and accents
                assert ids(await search("poverty")) == [5]
                assert ids(await search("POVERTY")) == [5]
                assert 4 in ids(await search("eclair"))
                assert ids(await search("ECLAIR education")) == [4]

                # typos: fuzziness AUTO:5,9 with the first two letters kept
                assert ids(await search("cencus tables")) == [6]

                # idno is searchable, case-insensitively
                assert ids(await search("ken_2018_lfs"))[0] == 2

                # a title match (boost 40) outranks an abstract-only match
                order = ids(await search("price"))
                assert order[0] == 3 and set(order) == {3, 4, 7}

                # minimum_should_match: two terms must both be present in a field
                assert ids(await search("consumer poverty")) == []
                assert ids(await search("consumer price"))[:1] == [3]

                # scores descend, and ties (if any) break by sid ascending
                page = await search("survey")
                scores = [h["score"] for h in page.hits]
                assert scores == sorted(scores, reverse=True) and all(s > 0 for s in scores)
                assert all(h["matched_by"] == ["lexical"] for h in page.hits)
                assert (await search("survey")).hits == page.hits  # deterministic

                # filters apply to the matches; `types` narrows found and the hits but not the tab counts
                assert ids(await search("survey", filters={"countries": [818]})) == [1]
                everything = await search("survey")
                assert everything.found == 5 and everything.counts_by_type == {"survey": 3, "geospatial": 1, "table": 1}
                surveys = await search("survey", filters={"types": ["survey"]})
                assert surveys.found == 3
                assert surveys.counts_by_type == everything.counts_by_type  # the tabs do not change

                # nothing is cut: every match is returned, and the total counts them all however small the page
                assert (await search("survey", limit=2)).found == 5
                assert sum((await search("survey", limit=1)).counts_by_type.values()) == 5

                # another sort orders ALL the matches (by title here); a title sort has no relevance score
                by_title = await search("survey", by="title", order="asc")
                titles = {sid: STUDIES[sid]["title"].lower() for sid in ids(everything)}
                assert ids(by_title) == sorted(ids(everything), key=lambda sid: (titles[sid], sid))
                assert by_title.found == 5 and all(h["score"] is None for h in by_title.hits)

                # paging is done by OpenSearch, and the pages join up
                first = await search("survey", limit=2, offset=0)
                second = await search("survey", limit=2, offset=2)
                third = await search("survey", limit=2, offset=4)
                assert ids(first) + ids(second) + ids(third) == ids(everything)
                assert (await search("survey", limit=2, offset=50)).hits == []
                assert (await search("survey", limit=2, offset=50)).found == 5
            finally:
                await client.close()

        asyncio.run(scenario())
    finally:
        for index in (settings.index_name, settings.studies_index):
            sync.indices.delete(index=index, ignore_unavailable=True)
        for name in (composable_index_template_name(settings), studies_index_template_name(settings)):
            try:
                sync.indices.delete_index_template(name=name)
            except Exception:
                pass
        sync.transport.close()
