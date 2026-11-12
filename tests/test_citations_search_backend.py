"""``citations_search.py``: the lexical query builder and executor for ``POST /citations/search``."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from opensearchpy.exceptions import NotFoundError

from nada_ai.search.backend.opensearch.citations_search import (
    DOI_BOOST,
    LEXICAL_FIELDS,
    SORT_VALUE_KINDS,
    SORTS,
    CitationFilters,
    CitationSearchJob,
    IndexNotReady,
    filter_clauses,
    search_body,
    search_citations,
)
from nada_ai.search.backend.opensearch.mapping import CITATION_TEXT_FIELDS
from nada_ai.search.cursor import Cursor, Direction, decode


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
    assert body["sort"] == [{"_score": {"order": "desc"}}, {"citation_id": {"order": "asc"}}]  # relevance
    assert body["track_scores"] is True
    assert set(body["_source"]) == {"citation_id", "uuid"}  # ids only: the caller reads the rest from its database


@pytest.mark.parametrize(
    ("sort_by", "order", "expected"),
    [
        ("title", "asc", [{"title_sort": {"order": "asc", "missing": "_last"}}, {"citation_id": {"order": "asc"}}]),
        (
            "year",
            "desc",
            [
                {"pub_year": {"order": "desc", "missing": "_last"}},
                {"title_sort": {"order": "desc", "missing": "_last"}},
                {"citation_id": {"order": "desc"}},
            ],
        ),
    ],
)
def test_sorting_by_title_or_year_puts_missing_values_last_and_ties_by_citation_id(sort_by, order, expected) -> None:
    assert search_body(_job(sort_by=sort_by, order=order))["sort"] == expected


def test_every_sort_ends_on_the_unique_citation_id_and_has_cursor_kinds() -> None:
    for name, fields in SORTS.items():
        assert fields[-1][0] == "citation_id"
        assert len(SORT_VALUE_KINDS[name]) == len(fields)


def test_a_cursor_page_uses_search_after_one_extra_row_and_no_count() -> None:
    cursor = Cursor(Direction.after, (2.5, 40), 10_000, 14_200, False)
    body = search_body(_job(cursor=cursor))
    assert body["search_after"] == [2.5, 40] and body["size"] == 16 and body["track_total_hits"] is False
    assert "from" not in body


def test_a_before_cursor_reads_the_sort_reversed() -> None:
    cursor = Cursor(Direction.before, ("poverty", 40), 10_000, 14_200, False)
    body = search_body(_job(sort_by="title", order="asc", cursor=cursor))
    assert body["sort"] == [{"title_sort": {"order": "desc", "missing": "_first"}}, {"citation_id": {"order": "desc"}}]


def _rows(ids: list[int]) -> list[dict[str, object]]:
    return [{"_score": 1.0, "_source": {"citation_id": i}, "sort": [1.0, i]} for i in ids]


def test_the_first_page_issues_a_next_cursor_carrying_the_total() -> None:
    client = AsyncMock()
    client.search.return_value = {"took": 1, "hits": {"total": {"value": 40}, "hits": _rows([1, 2, 3])}}
    page = asyncio.run(search_citations(_job(client=client, limit=3, offset=0)))
    assert page.offset == 0 and page.prev_cursor is None
    cursor = decode(page.next_cursor, SORT_VALUE_KINDS["relevance"])
    assert cursor.direction is Direction.after and cursor.sort_values == (1.0, 3)
    assert cursor.position == 3 and cursor.total == 40


def test_an_after_cursor_page_takes_its_position_and_total_from_the_cursor() -> None:
    client = AsyncMock()
    client.search.return_value = {"took": 1, "hits": {"hits": _rows([4, 5, 6, 7])}}  # one more than the page
    cursor = Cursor(Direction.after, (1.0, 3), 10_000, 14_200, False)
    page = asyncio.run(search_citations(_job(client=client, limit=3, cursor=cursor)))
    assert [h.citation_id for h in page.hits] == [4, 5, 6]
    assert page.found == 14_200 and page.offset == 10_000
    assert page.next_cursor is not None and page.prev_cursor is not None
    client.count.assert_not_called()


def test_the_last_after_page_has_no_next_cursor() -> None:
    client = AsyncMock()
    client.search.return_value = {"took": 1, "hits": {"hits": _rows([8, 9])}}
    cursor = Cursor(Direction.after, (1.0, 7), 14_198, 14_200, False)
    page = asyncio.run(search_citations(_job(client=client, limit=3, cursor=cursor)))
    assert page.next_cursor is None and page.prev_cursor is not None


def test_a_before_cursor_page_is_turned_around_and_ends_before_the_cursor() -> None:
    client = AsyncMock()
    client.search.return_value = {"took": 1, "hits": {"hits": _rows([6, 5, 4, 3])}}  # reversed, one extra
    cursor = Cursor(Direction.before, (1.0, 7), 10_003, 14_200, False)
    page = asyncio.run(search_citations(_job(client=client, limit=3, cursor=cursor)))
    assert [h.citation_id for h in page.hits] == [4, 5, 6]
    assert page.offset == 10_000 and page.prev_cursor is not None and page.next_cursor is not None


def test_the_executor_returns_hits_in_order_with_the_total() -> None:
    client = AsyncMock()
    client.search.return_value = {
        "took": 4,
        "hits": {
            "total": {"value": 2},
            "hits": [
                {"_score": 3.0, "_source": {"citation_id": 5, "uuid": "u-5"}, "sort": [3.0, 5]},
                {"_score": 1.0, "_source": {"citation_id": 2}, "sort": [1.0, 2]},
            ],
        },
    }
    page = asyncio.run(search_citations(_job(client=client)))
    assert page.found == 2 and page.took_ms == 4
    assert [h.citation_id for h in page.hits] == [5, 2]
    assert page.hits[0].uuid == "u-5" and page.hits[1].uuid is None


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


def test_the_empty_index_check_is_bounded_by_the_search_timeout() -> None:
    client = AsyncMock()
    _no_hits(client, indexed=3)
    job = _job(client=client)
    asyncio.run(search_citations(job))
    assert client.count.call_args.kwargs["request_timeout"] == job.timeout_seconds


def test_index_deleted_between_search_and_count_is_not_ready() -> None:
    client = AsyncMock()
    client.search.return_value = {"took": 1, "hits": {"total": {"value": 0}, "hits": []}}
    client.count.side_effect = NotFoundError(404, "index_not_found_exception", {})
    with pytest.raises(IndexNotReady):
        asyncio.run(search_citations(_job(client=client)))
