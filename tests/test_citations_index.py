"""``citations_index``: sync of one citation (write or remove) and the catalog-wide backfill."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from nada_ai.ingest import citations_index as ci
from nada_ai.ingest.extract_access import ExtractError
from nada_ai.ingest.progress import CancelToken
from nada_ai.settings import Settings


def _citation(citation_id: int) -> dict:
    return {
        "metadata": {"title": f"Title {citation_id}"},
        "core_fields": {"citation_id": citation_id},
        "filters": {"published": 1},
    }


def _settings() -> Settings:
    return Settings(search_backend="opensearch", index_name="idx", metadata_extract_base_url="http://nada/extract")


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://nada/extract/citations/9")
    return httpx.HTTPStatusError("boom", request=request, response=httpx.Response(status, request=request))


class _Run:
    def __init__(self) -> None:
        self.client = MagicMock()
        self.client.indices.exists.return_value = True
        self.client.delete_by_query.return_value = {"deleted": 1}
        self.written: list[list[int]] = []

    def bulk(self, client, actions, **_):
        self.written.append([int(a["_id"]) for a in actions])
        return len(actions), []

    def __enter__(self):
        self._patches = [
            patch.object(ci, "build_client", return_value=self.client),
            patch.object(ci, "bulk", side_effect=self.bulk),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()


def test_sync_writes_the_citation_and_refreshes_the_index() -> None:
    with _Run() as run, patch.object(ci.catalog_extract, "fetch_extract_citation", return_value=_citation(7)) as fetch:
        result = ci.sync_citation_op(_settings(), 7)

    assert result == {"citation_id": 7, "indexed": 1, "deleted": 0, "errors": []}
    assert fetch.call_args.args == (7,)
    assert run.written == [[7]]
    run.client.indices.refresh.assert_called_once_with(index="idx-citations")


def test_sync_removes_the_citation_when_nada_no_longer_has_it() -> None:
    with (
        _Run() as run,
        patch.object(ci.catalog_extract, "fetch_extract_citation", side_effect=_http_error(404)),
    ):
        result = ci.sync_citation_op(_settings(), 9)

    assert result == {"citation_id": 9, "indexed": 0, "deleted": 1, "errors": []}
    body = run.client.delete_by_query.call_args.kwargs["body"]
    assert body == {"query": {"ids": {"values": ["9"]}}}
    assert run.written == []


def test_sync_keeps_the_existing_document_when_nada_cannot_be_read() -> None:
    with _Run() as run, patch.object(ci.catalog_extract, "fetch_extract_citation", side_effect=_http_error(500)):
        with pytest.raises(ExtractError):
            ci.sync_citation_op(_settings(), 9)
    run.client.delete_by_query.assert_not_called()


def test_sync_reports_a_failure_that_is_not_an_http_status_as_an_extract_error() -> None:
    with _Run() as run, patch.object(ci.catalog_extract, "fetch_extract_citation", side_effect=RuntimeError("down")):
        with pytest.raises(ExtractError, match="down"):
            ci.sync_citation_op(_settings(), 9)
    run.client.delete_by_query.assert_not_called()


def test_delete_tolerates_a_missing_index() -> None:
    with _Run() as run:
        ci.delete_citation_op(_settings(), 4)
    assert run.client.delete_by_query.call_args.kwargs["ignore_unavailable"] is True


def test_backfill_writes_pages_reports_progress_and_the_first_pages_total() -> None:
    progress: list[dict] = []

    def iterator(**kw):
        kw["on_page"]({"total": 3})
        yield from (_citation(i) for i in (1, 2, 3))

    with _Run() as run, patch.object(ci.catalog_extract, "iter_extract_citations", side_effect=iterator):
        result = ci.backfill_citations_op(
            _settings(), batch_size=2, show_progress_bar=False, progress_cb=progress.append
        )

    assert run.written == [[1, 2], [3]]
    assert result == {"seen": 3, "indexed": 3, "errors": [], "total": 3, "cancelled": False}
    assert [p["processed"] for p in progress] == [2, 3]
    assert progress[-1]["percent"] == 100.0 and progress[-1]["total"] == 3


def test_backfill_stops_between_pages_when_cancelled() -> None:
    token = CancelToken()

    def iterator(**kw):
        yield from (_citation(i) for i in (1, 2, 3, 4))

    def cancel_after_first(_client, actions, **_):
        token.set()
        return len(actions), []

    with (
        _Run() as run,
        patch.object(ci, "bulk", side_effect=cancel_after_first),
        patch.object(ci.catalog_extract, "iter_extract_citations", side_effect=iterator),
    ):
        result = ci.backfill_citations_op(_settings(), batch_size=2, show_progress_bar=False, cancel_token=token)

    assert result["cancelled"] is True and result["seen"] == 2
    del run


def test_backfill_can_recreate_the_index_first() -> None:
    with (
        _Run() as run,
        patch.object(ci.catalog_extract, "iter_extract_citations", return_value=iter(())),
    ):
        ci.backfill_citations_op(_settings(), show_progress_bar=False, recreate_index=True)
    run.client.indices.delete.assert_called_once_with(index="idx-citations")
