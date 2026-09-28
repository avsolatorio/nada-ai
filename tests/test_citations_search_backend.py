"""``citations_search.py``: the lexical query builder and executor for ``POST /citations/search``."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from opensearchpy.exceptions import NotFoundError

from nada_ai.search.backend.opensearch.citations_search import (
    DOI_BOOST,
    LEXICAL_FIELDS,
    CitationFilters,
    CitationSearchJob,
    IndexNotReady,
    filter_clauses,
    search_body,
    search_citations,
)
from nada_ai.search.backend.opensearch.mapping import CITATION_TEXT_FIELDS


def test_lexical_fields_name_every_citation_text_field_exactly_once() -> None:
    named = [f.split("^")[0] for f in LEXICAL_FIELDS]
    assert sorted(named) == sorted(CITATION_TEXT_FIELDS)
    assert len(named) == len(set(named))


def test_filter_clauses_always_require_published() -> None:
    assert filter_clauses(CitationFilters()) == [{"term": {"published": 1}}]


def test_filter_clauses_add_types_and_the_year_range() -> None:
    clauses = filter_clauses(CitationFilters(ctypes=("book",), year_from=2000, year_to=2010))
    assert {"terms": {"ctype": ["book"]}} in clauses
    assert {"range": {"pub_year": {"gte": 2000, "lte": 2010}}} in clauses


def test_an_open_ended_year_range_has_one_bound() -> None:
    assert {"range": {"pub_year": {"gte": 2015}}} in filter_clauses(CitationFilters(year_from=2015))
    assert {"range": {"pub_year": {"lte": 1999}}} in filter_clauses(CitationFilters(year_to=1999))


def _job(**overrides: object) -> CitationSearchJob:
    base: dict[str, object] = dict(
        client=AsyncMock(),
        index="idx-citations",
        query="poverty",
        filters=CitationFilters(),
        limit=15,
        offset=30,
        sort_by="relevance",
        order="asc",
    )
    base.update(overrides)
    return CitationSearchJob(**base)  # type: ignore[arg-type]


def test_the_query_matches_the_text_fields_or_the_doi_exactly() -> None:
    query = search_body(_job(query="10.1/ABC"))["query"]["bool"]["must"][0]["bool"]
    assert query["minimum_should_match"] == 1
    assert query["should"][0]["multi_match"]["fields"] == list(LEXICAL_FIELDS)
    assert query["should"][1] == {"term": {"doi": {"value": "10.1/ABC", "boost": DOI_BOOST}}}


def test_paging_and_the_returned_fields() -> None:
    body = search_body(_job())
    assert body["size"] == 15 and body["from"] == 30 and body["track_total_hits"] is True
    assert "sort" not in body  # relevance
    assert set(body["_source"]) == {"citation_id", "uuid", "title", "authors", "ctype", "pub_year", "doi"}


@pytest.mark.parametrize(
    ("sort_by", "order", "field"),
    [("title", "asc", "title_sort"), ("year", "desc", "pub_year")],
)
def test_sorting_by_title_or_year_puts_missing_values_last_and_ties_by_score(sort_by, order, field) -> None:
    body = search_body(_job(sort_by=sort_by, order=order))
    assert body["sort"] == [{field: {"order": order, "missing": "_last"}}, "_score"]


def test_the_executor_returns_hits_in_order_with_the_total() -> None:
    client = AsyncMock()
    client.search.return_value = {
        "took": 4,
        "hits": {
            "total": {"value": 2},
            "hits": [
                {"_score": 3.0, "_source": {"citation_id": 5, "title": "A", "pub_year": 2001}},
                {"_score": 1.0, "_source": {"citation_id": 2, "title": "B"}},
            ],
        },
    }
    page = asyncio.run(search_citations(_job(client=client)))
    assert page.found == 2 and page.took_ms == 4
    assert [h.citation_id for h in page.hits] == [5, 2]
    assert page.hits[0].pub_year == 2001 and page.hits[1].pub_year is None


def test_a_missing_index_is_reported_as_not_ready() -> None:
    client = AsyncMock()
    client.search.side_effect = NotFoundError(404, "index_not_found_exception", {})
    with pytest.raises(IndexNotReady):
        asyncio.run(search_citations(_job(client=client)))


def _no_hits(client: AsyncMock, indexed: int) -> None:
    client.search.return_value = {"took": 1, "hits": {"total": {"value": 0}, "hits": []}}
    client.count.return_value = {"count": indexed}


def test_an_empty_index_is_reported_as_not_ready() -> None:
    client = AsyncMock()
    _no_hits(client, indexed=0)
    with pytest.raises(IndexNotReady):
        asyncio.run(search_citations(_job(client=client)))


def test_no_matches_in_a_populated_index_is_an_empty_page() -> None:
    client = AsyncMock()
    _no_hits(client, indexed=3)
    page = asyncio.run(search_citations(_job(client=client)))
    assert page.found == 0 and page.hits == []


def test_index_deleted_between_search_and_count_is_not_ready() -> None:
    client = AsyncMock()
    client.search.return_value = {"took": 1, "hits": {"total": {"value": 0}, "hits": []}}
    client.count.side_effect = NotFoundError(404, "index_not_found_exception", {})
    with pytest.raises(IndexNotReady):
        asyncio.run(search_citations(_job(client=client)))
