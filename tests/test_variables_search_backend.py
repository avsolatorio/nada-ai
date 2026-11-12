"""``variables_search.py``: the lexical query builder and executor for ``POST /variables/search``."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from opensearchpy.exceptions import NotFoundError

from nada_ai.app.studies_schemas import StudyFilters
from nada_ai.search.backend.opensearch.mapping import VARIABLE_TEXT_FIELDS
from nada_ai.search.backend.opensearch.variables_search import (
    LEXICAL_FIELDS,
    SORT_VALUE_KINDS,
    SORTS,
    IndexNotReady,
    TooManyStudies,
    VariableSearchJob,
    preference,
    search_body,
    search_variables,
    sort_clauses,
    study_clauses,
)
from nada_ai.search.cursor import Cursor, Direction, decode

STUDIES = "nada-ai-studies"
VARIABLES = "nada-ai-variables"


def test_lexical_fields_name_every_variable_text_field_exactly_once() -> None:
    named = [f.split("^")[0] for f in LEXICAL_FIELDS]
    assert sorted(named) == sorted(VARIABLE_TEXT_FIELDS)
    assert len(named) == len(set(named))


def test_the_allowed_studies_are_always_published_ones() -> None:
    assert study_clauses(StudyFilters()) == [{"term": {"filter_facets.published": 1}}]


def test_study_level_filters_are_the_study_searchs_own() -> None:
    clauses = study_clauses(
        StudyFilters(
            sids=[1, 2],
            types=["survey"],
            countries=[4, 7],
            year_from=2000,
            year_to=2010,
            repository="central",
            collections=["wb"],
            form_ids=[3],
        )
    )
    assert {"terms": {"sid": [1, 2]}} in clauses
    assert {"terms": {"filter_facets.dataset_type": ["survey"]}} in clauses
    assert {"terms": {"filter_facets.countries": [4, 7]}} in clauses
    assert {"range": {"filter_facets.years": {"gte": 2000, "lte": 2010}}} in clauses
    assert {"term": {"filter_facets.repositories": "central"}} in clauses
    assert {"terms": {"filter_facets.repositories": ["wb"]}} in clauses
    assert {"terms": {"filter_facets.formid": [3]}} in clauses


def _job(**overrides: object) -> VariableSearchJob:
    defaults: dict[str, object] = dict(
        client=None,
        index=VARIABLES,
        studies_index=STUDIES,
        query="diarrhea",
        filters=StudyFilters(),
        limit=15,
        offset=0,
        sort_by="relevance",
    )
    defaults.update(overrides)
    return VariableSearchJob(**defaults)  # type: ignore[arg-type]


def test_search_body_keeps_the_allowed_studies_variables() -> None:
    body = search_body(_job(), [3, 5])
    assert body["size"] == 15
    assert body["from"] == 0
    assert body["track_total_hits"] == 20_000  # a query's matches are counted up to the cap
    assert body["query"]["bool"]["must"] == [
        {"multi_match": {"query": "diarrhea", "fields": list(LEXICAL_FIELDS), "type": "best_fields"}}
    ]
    assert body["query"]["bool"]["filter"] == [{"terms": {"sid": [3, 5]}}]
    assert body["sort"] == [{"_score": {"order": "desc"}}, {"uid": {"order": "asc"}}]


def test_a_browse_is_a_filter_only_query_counted_exactly() -> None:
    body = search_body(_job(query=None, sort_by="title"), [3])
    assert body["query"] == {"bool": {"filter": [{"terms": {"sid": [3]}}]}}
    assert body["track_total_hits"] is True


def test_every_sort_ends_on_the_uid_tiebreaker_and_has_a_kind_per_field() -> None:
    for name, fields in SORTS.items():
        assert fields[-1][0] == "uid", name
        assert len(SORT_VALUE_KINDS[name]) == len(fields), name


def test_name_and_title_sorts_follow_the_order_and_keep_missing_values_last() -> None:
    assert sort_clauses("name", "desc") == [
        {"name.sort": {"order": "desc", "missing": "_last"}},
        {"uid": {"order": "desc"}},
    ]
    assert sort_clauses("title", "asc") == [
        {"title": {"order": "asc", "missing": "_last"}},
        {"name.sort": {"order": "asc", "missing": "_last"}},
        {"uid": {"order": "asc"}},
    ]


def test_a_reversed_sort_flips_every_field_and_moves_missing_values_first() -> None:
    """Exactly the reverse order, or a page read backwards would not be the previous page."""
    assert sort_clauses("name", "asc", reverse=True) == [
        {"name.sort": {"order": "desc", "missing": "_first"}},
        {"uid": {"order": "desc"}},
    ]
    assert sort_clauses("relevance", "asc", reverse=True) == [{"_score": {"order": "asc"}}, {"uid": {"order": "desc"}}]


def test_every_sort_field_is_a_normalized_keyword_in_the_variables_mapping() -> None:
    """What the body sorts on must be sortable in the mapping — the check the body test alone could not make."""
    from nada_ai.search.backend.opensearch.mapping import variables_index_body

    properties = variables_index_body()["mappings"]["properties"]
    for fields in SORTS.values():
        for path, _ in fields:
            if path in ("_score", "uid"):
                continue
            field, _, sub = path.partition(".")
            mapping = properties[field]["fields"][sub] if sub else properties[field]
            assert mapping["type"] == "keyword", path
            assert mapping["normalizer"] == "nada_sort", path
    assert properties["uid"]["type"] == "integer"


def _cursor(direction: Direction, values: tuple[object, ...], position: int, total: int = 50) -> Cursor:
    return Cursor(direction, values, position, total, False)


def test_a_cursor_page_uses_search_after_one_extra_row_and_no_count() -> None:
    body = search_body(_job(cursor=_cursor(Direction.after, (3.5, 9), 15)), [1])
    assert body["search_after"] == [3.5, 9]
    assert body["size"] == 16
    assert body["track_total_hits"] is False
    assert "from" not in body


def test_a_before_cursor_reads_the_sort_reversed() -> None:
    body = search_body(_job(sort_by="name", cursor=_cursor(Direction.before, ("hhid", 9), 15)), [1])
    assert body["sort"][0] == {"name.sort": {"order": "desc", "missing": "_first"}}


def _row(uid: int, score: float) -> dict[str, object]:
    return {"_score": score, "_source": {"uid": uid, "sid": 1}, "sort": [score, uid]}


def _client(
    rows: list[dict[str, object]] | None = None,
    total: int = 0,
    relation: str = "eq",
    *,
    sids: list[int] | None = None,
    other_studies: int = 0,
    published_studies: int = 10,
    indexed_variables: int = 100,
) -> AsyncMock:
    """The study index answers the allowed study ids (``sids``), the variable index the rows."""
    client = AsyncMock()

    async def search(index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        if index == STUDIES:
            buckets = [{"key": sid, "doc_count": 1} for sid in ([1] if sids is None else sids)]
            return {"aggregations": {"sids": {"buckets": buckets, "sum_other_doc_count": other_studies}}}
        return {"took": 2, "hits": {"total": {"value": total, "relation": relation}, "hits": rows or []}}

    async def count(index: str, **_: Any) -> dict[str, Any]:
        return {"count": published_studies if index == STUDIES else indexed_variables}

    client.search.side_effect = search
    client.count.side_effect = count
    return client


def _variable_calls(client: AsyncMock) -> list[Any]:
    return [c for c in client.search.call_args_list if c.kwargs["index"] == VARIABLES]


def _decoded(token: str | None) -> Cursor:
    assert token is not None
    return decode(token, SORT_VALUE_KINDS["relevance"])


def test_search_variables_parses_hits_and_total() -> None:
    client = _client([_row(1, 5.1)], total=1)
    page = asyncio.run(search_variables(_job(client=client)))
    assert (page.found, page.found_is_lower_bound, page.offset, page.took_ms) == (1, False, 0, 2)
    hit = page.hits[0]
    assert (hit.uid, hit.sid, hit.score) == (1, 1, 5.1)
    assert page.prev_cursor is None and page.next_cursor is None  # one page holds everything


def test_the_variables_of_the_allowed_studies_are_searched() -> None:
    client = _client([_row(1, 5.1)], total=1, sids=[4, 9])
    asyncio.run(search_variables(_job(client=client, filters=StudyFilters(collections=["wb"]))))
    studies_query = client.search.call_args_list[0].kwargs["body"]["query"]["bool"]["filter"]
    assert {"terms": {"filter_facets.repositories": ["wb"]}} in studies_query
    assert _variable_calls(client)[0].kwargs["body"]["query"]["bool"]["filter"] == [{"terms": {"sid": [4, 9]}}]


def test_no_matching_study_is_an_empty_page_without_asking_the_variable_index() -> None:
    client = _client(sids=[])
    page = asyncio.run(search_variables(_job(client=client, filters=StudyFilters(collections=["none"]))))
    assert (page.found, page.hits, page.next_cursor) == (0, [], None)
    assert _variable_calls(client) == []


def test_an_empty_study_index_is_not_ready_rather_than_no_results() -> None:
    """The allow-list fails closed: while the study index is rebuilt, no variable is shown — and it says why."""
    client = _client(sids=[], published_studies=0)
    with pytest.raises(IndexNotReady, match=STUDIES):
        asyncio.run(search_variables(_job(client=client)))


def test_more_studies_than_one_search_can_allow_is_refused() -> None:
    with pytest.raises(TooManyStudies):
        asyncio.run(search_variables(_job(client=_client(sids=[1], other_studies=5))))


def test_the_search_is_routed_by_preference_and_bounded_by_the_timeout() -> None:
    client = _client([_row(1, 5.1)], total=1)
    asyncio.run(search_variables(_job(client=client, timeout_seconds=3.0)))
    kwargs = _variable_calls(client)[0].kwargs
    assert kwargs["request_timeout"] == 3.0
    assert kwargs["preference"] == preference(_job())  # the same for every page of one search
    assert client.search.call_args_list[0].kwargs["request_timeout"] == 3.0  # the study index read too


def test_the_empty_index_checks_of_a_zero_result_search_are_bounded_by_the_timeout() -> None:
    no_studies = _client(sids=[])  # no allowed study: is the study index empty?
    asyncio.run(search_variables(_job(client=no_studies, timeout_seconds=3.0)))
    no_rows = _client([], total=0)  # no variable matches: is the variable index empty?
    asyncio.run(search_variables(_job(client=no_rows, timeout_seconds=3.0)))
    for client in (no_studies, no_rows):
        client.count.assert_called_once()
        assert client.count.call_args.kwargs["request_timeout"] == 3.0


def test_a_count_past_the_cap_is_a_lower_bound_carried_by_the_cursors() -> None:
    rows = [_row(i, 10.0 - i) for i in range(1, 16)]
    page = asyncio.run(search_variables(_job(client=_client(rows, total=20_000, relation="gte"), offset=15)))
    assert page.found_is_lower_bound is True
    nxt = _decoded(page.next_cursor)
    assert (nxt.direction, nxt.sort_values, nxt.position) == (Direction.after, (-5.0, 15), 30)
    assert (nxt.total, nxt.total_is_lower_bound) == (20_000, True)
    prev = _decoded(page.prev_cursor)
    assert (prev.direction, prev.sort_values, prev.position) == (Direction.before, (9.0, 1), 15)


def test_an_after_cursor_page_takes_its_position_and_total_from_the_cursor() -> None:
    rows = [_row(i, 50.0 - i) for i in range(20, 36)]  # 16 rows: one more than a page, so another follows
    client = _client(rows)
    page = asyncio.run(search_variables(_job(client=client, cursor=_cursor(Direction.after, (31.0, 19), 1005, 4000))))
    assert [h.uid for h in page.hits] == list(range(20, 35))
    assert (page.offset, page.found) == (1005, 4000)
    assert _decoded(page.next_cursor).position == 1020
    assert _decoded(page.prev_cursor).position == 1005
    client.count.assert_not_awaited()  # nothing is counted on a cursor page


def test_the_last_after_page_has_no_next_cursor() -> None:
    rows = [_row(i, 1.0) for i in (1, 2, 3)]
    page = asyncio.run(search_variables(_job(client=_client(rows), cursor=_cursor(Direction.after, (2.0, 9), 45))))
    assert page.next_cursor is None and page.prev_cursor is not None


def test_a_before_page_is_flipped_back_into_order() -> None:
    reversed_rows = [_row(i, float(i)) for i in range(30, 14, -1)]  # 16 rows, the reversed sort's order
    page = asyncio.run(
        search_variables(_job(client=_client(reversed_rows), cursor=_cursor(Direction.before, (31.0, 31), 1020)))
    )
    assert [h.uid for h in page.hits] == list(range(16, 31))
    assert page.offset == 1005
    assert page.next_cursor is not None and page.prev_cursor is not None


def test_a_before_page_that_reaches_the_start_is_the_first_page() -> None:
    reversed_rows = [_row(i, float(i)) for i in (3, 2, 1)]  # fewer than a page: nothing more before
    page = asyncio.run(
        search_variables(_job(client=_client(reversed_rows), cursor=_cursor(Direction.before, (4.0, 4), 40)))
    )
    assert [h.uid for h in page.hits] == [1, 2, 3]
    assert page.offset == 0  # whatever the cursor's position said
    assert page.prev_cursor is None


def test_a_missing_variable_index_is_not_ready() -> None:
    client = _client()
    studies = client.search.side_effect

    async def search(index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
        if index == VARIABLES:
            raise NotFoundError(404, "index_not_found_exception", {})
        return await studies(index, body, **kw)

    client.search.side_effect = search
    with pytest.raises(IndexNotReady, match=VARIABLES):
        asyncio.run(search_variables(_job(client=client)))


def test_a_missing_study_index_is_not_ready() -> None:
    client = AsyncMock()
    client.search.side_effect = NotFoundError(404, "index_not_found_exception", {})
    with pytest.raises(IndexNotReady, match=STUDIES):
        asyncio.run(search_variables(_job(client=client)))


def test_an_empty_variable_index_is_not_ready() -> None:
    with pytest.raises(IndexNotReady, match=VARIABLES):
        asyncio.run(search_variables(_job(client=_client([], total=0, indexed_variables=0))))


def test_nothing_matching_in_a_populated_index_is_an_empty_page() -> None:
    page = asyncio.run(search_variables(_job(client=_client([], total=0, indexed_variables=3))))
    assert page.found == 0 and page.hits == []
