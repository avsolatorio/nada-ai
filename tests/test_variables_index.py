"""``variables_index``: paged sync of one study's variables, and the catalog-wide backfill."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nada_ai.ingest import variables_index as vi
from nada_ai.ingest.progress import CancelToken
from nada_ai.settings import Settings


def _variable(uid: int, sid: int = 5, idno: str = "S-1") -> dict:
    return {
        "core_fields": {"uid": uid, "sid": sid, "idno": idno, "name": f"v{uid}", "label": f"label {uid}"},
        "filters": {"published": 1},
    }


def _settings() -> Settings:
    return Settings(search_backend="opensearch", index_name="idx", metadata_extract_base_url="http://nada/extract")


class _Run:
    """Patches everything a sync/backfill touches outside itself and records the order things happen in."""

    def __init__(self, pages: list[list[dict]]):
        self.events: list[str] = []
        self.client = MagicMock()
        self.client.indices.exists.return_value = True
        self.client.delete_by_query.side_effect = lambda **kw: self.events.append("delete")
        self.written: list[list[int]] = []
        self.pages = pages

    def bulk(self, client, actions, **_):
        self.events.append("bulk")
        self.written.append([int(a["_id"]) for a in actions])
        return len(actions), []

    def iterator(self, *_a, **kw):
        page_size = kw.get("page_size", 1000)
        for page in self.pages:
            self.events.append("fetch")
            yield from page
        assert page_size  # the caller passes the page size through

    def __enter__(self):
        self._patches = [
            patch.object(vi, "build_client", return_value=self.client),
            patch.object(vi, "bulk", side_effect=self.bulk),
            patch.object(vi.catalog_extract, "iter_extract_survey_variables", side_effect=self.iterator),
            patch.object(vi.catalog_extract, "iter_extract_variables", side_effect=self.iterator),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()


def test_sync_writes_every_page_of_the_study() -> None:
    with _Run([[_variable(1), _variable(2)], [_variable(3)]]) as run:
        result = vi.sync_survey_variables_op(_settings(), "S-1", page_size=2)
    assert result == {"idno": "S-1", "indexed": 3, "errors": [], "cancelled": False}
    assert run.written == [[1, 2], [3]]  # one bulk write per page, never the whole study at once


def test_sync_reads_the_first_page_before_deleting_anything() -> None:
    """An unreadable study must leave its existing variables alone instead of emptying them."""
    with _Run([[_variable(1)]]) as run:
        vi.sync_survey_variables_op(_settings(), "S-1", page_size=10)
    assert run.events.index("fetch") < run.events.index("delete") < run.events.index("bulk")


def test_sync_deletes_nothing_when_the_first_page_cannot_be_read() -> None:
    run = _Run([])
    with run, patch.object(vi.catalog_extract, "iter_extract_survey_variables", side_effect=RuntimeError("nope")):
        with pytest.raises(vi.ExtractError, match="nope"):
            vi.sync_survey_variables_op(_settings(), "S-1")
    run.client.delete_by_query.assert_not_called()


def test_sync_of_a_study_with_no_variables_still_clears_the_old_ones() -> None:
    with _Run([[]]) as run:
        result = vi.sync_survey_variables_op(_settings(), "S-1")
    assert result["indexed"] == 0
    assert "delete" in run.events and "bulk" not in run.events


def test_sync_stops_between_pages_when_cancelled() -> None:
    token = CancelToken()
    with _Run([[_variable(1)], [_variable(2)]]) as run:
        original = run.bulk

        def bulk_then_cancel(client, actions, **kw):
            token.set()
            return original(client, actions, **kw)

        with patch.object(vi, "bulk", side_effect=bulk_then_cancel):
            result = vi.sync_survey_variables_op(_settings(), "S-1", page_size=1, cancel_token=token)
    assert result["cancelled"] is True
    assert run.written == [[1]]


def test_sync_does_not_delete_existing_variables_when_already_cancelled() -> None:
    token = CancelToken()
    token.set()
    with _Run([[_variable(1)]]) as run:
        result = vi.sync_survey_variables_op(_settings(), "S-1", cancel_token=token)

    assert result == {"idno": "S-1", "indexed": 0, "errors": [], "cancelled": True}
    run.client.delete_by_query.assert_not_called()


def test_backfill_reports_progress_with_the_first_pages_total() -> None:
    progress: list[dict] = []

    def iterator(*_a, **kw):
        kw["on_page"]({"total": 3})
        yield from [_variable(1), _variable(2), _variable(3)]

    with _Run([]) as run:
        with patch.object(vi.catalog_extract, "iter_extract_variables", side_effect=iterator):
            result = vi.backfill_variables_op(
                _settings(), batch_size=2, show_progress_bar=False, progress_cb=progress.append
            )
    assert run.written == [[1, 2], [3]]
    assert result == {"seen": 3, "indexed": 3, "errors": [], "total": 3, "cancelled": False}
    assert progress == [
        {"processed": 2, "total": 3, "failed": 0, "percent": 66.7},
        {"processed": 3, "total": 3, "failed": 0, "percent": 100.0},
    ]


def test_backfill_stops_when_cancelled() -> None:
    token = CancelToken()
    token.set()
    with _Run([[_variable(1)]]) as run:
        result = vi.backfill_variables_op(_settings(), show_progress_bar=False, cancel_token=token)
    assert result["cancelled"] is True and result["seen"] == 0
    assert run.written == []


def test_backfill_does_not_recreate_index_when_already_cancelled() -> None:
    token = CancelToken()
    token.set()
    with _Run([]) as run:
        result = vi.backfill_variables_op(
            _settings(), show_progress_bar=False, recreate_index=True, cancel_token=token
        )

    assert result == {"seen": 0, "indexed": 0, "errors": [], "total": None, "cancelled": True}
    run.client.indices.delete.assert_not_called()
