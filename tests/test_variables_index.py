"""``variables_index``: paged sync of one study's variables, and the catalog-wide backfill."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nada_ai.ingest import variables_index as vi
from nada_ai.ingest.progress import CancelToken
from nada_ai.settings import Settings


def _variable(uid: int, sid: int = 5, idno: str = "S-1") -> dict:
    return {
        "core_fields": {"uid": uid, "catalog_id": sid, "idno": idno, "name": f"v{uid}", "label": f"label {uid}"},
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
        for page in self.pages:
            self.events.append("fetch")
            yield from page

    def __enter__(self):
        self._patches = [
            patch.object(vi, "build_client", return_value=self.client),
            patch.object(vi, "bulk", side_effect=self.bulk),
            patch.object(vi, "_iter_study_variables", side_effect=self.iterator),
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
        result = vi.sync_survey_variables_op(_settings(), 5, page_size=2)
    assert result == {"sid": 5, "indexed": 3, "errors": [], "cancelled": False}
    assert run.written == [[1, 2], [3]]  # one bulk write per page, never the whole study at once


def test_sync_replaces_the_variables_of_the_study_by_its_id() -> None:
    with _Run([[_variable(1)]]) as run:
        vi.sync_survey_variables_op(_settings(), 42)
    assert run.client.delete_by_query.call_args.kwargs["body"] == {"query": {"term": {"sid": 42}}}


def test_variables_are_fetched_by_study_id_page_by_page() -> None:
    pages = [
        {"variables": [_variable(1), _variable(2)], "has_more": True, "next_after_uid": 2},
        {"variables": [_variable(3)], "has_more": False, "next_after_uid": None},
    ]
    with patch.object(vi.catalog_extract, "fetch_extract_survey_variables", side_effect=pages) as fetch:
        got = list(vi._iter_study_variables(_settings(), 42, 2))
    assert [v["core_fields"]["uid"] for v in got] == [1, 2, 3]
    first, second = (c for c in fetch.call_args_list)
    assert first.args == ("42",) and first.kwargs["params"] == {"limit": 2, "id_format": "id"}
    assert second.kwargs["params"] == {"limit": 2, "id_format": "id", "after_uid": 2}


def test_sync_reads_the_first_page_before_deleting_anything() -> None:
    """An unreadable study must leave its existing variables alone instead of emptying them."""
    with _Run([[_variable(1)]]) as run:
        vi.sync_survey_variables_op(_settings(), 5, page_size=10)
    assert run.events.index("fetch") < run.events.index("delete") < run.events.index("bulk")


def test_sync_deletes_nothing_when_the_first_page_cannot_be_read() -> None:
    run = _Run([])
    with run, patch.object(vi, "_iter_study_variables", side_effect=RuntimeError("nope")):
        with pytest.raises(vi.ExtractError, match="nope"):
            vi.sync_survey_variables_op(_settings(), 5)
    run.client.delete_by_query.assert_not_called()


def test_sync_deletes_nothing_when_the_first_page_cannot_be_built() -> None:
    """A page the extract sent without ``catalog_id`` must not empty the study before the build fails."""
    broken = _variable(1)
    del broken["core_fields"]["catalog_id"]
    with _Run([[broken]]) as run:
        with pytest.raises(vi.ExtractError, match="catalog_id"):
            vi.sync_survey_variables_op(_settings(), 5)
    run.client.delete_by_query.assert_not_called()


def test_sync_of_a_study_with_no_variables_still_clears_the_old_ones() -> None:
    with _Run([[]]) as run:
        result = vi.sync_survey_variables_op(_settings(), 5)
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
            result = vi.sync_survey_variables_op(_settings(), 5, page_size=1, cancel_token=token)
    assert result["cancelled"] is True
    assert run.written == [[1]]


def test_sync_does_not_delete_existing_variables_when_already_cancelled() -> None:
    token = CancelToken()
    token.set()
    with _Run([[_variable(1)]]) as run:
        result = vi.sync_survey_variables_op(_settings(), 5, cancel_token=token)

    assert result == {"sid": 5, "indexed": 0, "errors": [], "cancelled": True}
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
        result = vi.backfill_variables_op(_settings(), show_progress_bar=False, recreate_index=True, cancel_token=token)

    assert result == {"seen": 0, "indexed": 0, "errors": [], "total": None, "cancelled": True}
    run.client.indices.delete.assert_not_called()


def _indices(*, delete_error: Exception | None = None, create_errors: list[Exception] | None = None) -> MagicMock:
    client = MagicMock()
    if delete_error is not None:
        client.indices.delete.side_effect = delete_error
    if create_errors:
        client.indices.create.side_effect = [*create_errors, None]
    return client


def test_recreate_drops_the_variable_index_and_creates_it_with_the_current_mapping() -> None:
    from nada_ai.ingest.variables_index import recreate_variables_index
    from nada_ai.search.backend.opensearch.mapping import variables_index_body

    client = _indices()
    generation = recreate_variables_index(client, _settings())
    name = _settings().variables_index
    client.indices.delete.assert_called_once_with(index=name)
    body = client.indices.create.call_args.kwargs["body"]
    assert body["mappings"]["properties"] == variables_index_body()["mappings"]["properties"]
    assert body["mappings"]["_meta"]["generation"] == generation


def test_recreate_installs_the_current_index_template_first() -> None:
    """An old template would merge the old mapping's fields back into the new index."""
    from nada_ai.ingest.variables_index import recreate_variables_index
    from nada_ai.search.backend.opensearch.mapping import variables_index_body

    client = _indices()
    calls: list[str] = []
    client.indices.put_index_template.side_effect = lambda **kw: calls.append("template")
    client.indices.create.side_effect = lambda **kw: calls.append("create")
    recreate_variables_index(client, _settings())
    assert calls == ["template", "create"]
    template = client.indices.put_index_template.call_args.kwargs["body"]["template"]
    assert template["mappings"]["properties"] == variables_index_body()["mappings"]["properties"]


def test_creating_a_missing_variable_index_installs_its_current_template_first() -> None:
    """The first sync (or one after Delete and rebuild) creates the index: an old template must not shape it."""
    from nada_ai.ingest.variables_index import ensure_variables_index

    client = _indices()
    client.indices.exists.return_value = False
    calls: list[str] = []
    client.indices.put_index_template.side_effect = lambda **kw: calls.append("template")
    client.indices.create.side_effect = lambda **kw: calls.append("create")
    ensure_variables_index(client, _settings())
    assert calls == ["template", "create"]


def test_no_template_is_installed_when_templates_are_switched_off() -> None:
    from nada_ai.ingest.variables_index import recreate_variables_index

    client = _indices()
    recreate_variables_index(client, _settings().model_copy(update={"opensearch_put_composable_index_template": False}))
    client.indices.put_index_template.assert_not_called()
    client.indices.create.assert_called_once()


def test_recreate_creates_a_missing_variable_index() -> None:
    from opensearchpy.exceptions import NotFoundError

    from nada_ai.ingest.variables_index import recreate_variables_index

    client = _indices(delete_error=NotFoundError(404, "index_not_found_exception", {}))
    recreate_variables_index(client, _settings())
    client.indices.create.assert_called_once()


def test_recreate_deletes_again_when_a_write_created_the_index_in_between() -> None:
    from opensearchpy.exceptions import RequestError

    from nada_ai.ingest.variables_index import recreate_variables_index

    client = _indices(create_errors=[RequestError(400, "resource_already_exists_exception", {})])
    recreate_variables_index(client, _settings())
    assert client.indices.delete.call_count == 2 and client.indices.create.call_count == 2


def test_recreate_fails_at_once_on_any_other_error() -> None:
    from opensearchpy.exceptions import RequestError

    from nada_ai.ingest.variables_index import recreate_variables_index

    client = _indices(create_errors=[RequestError(400, "mapper_parsing_exception", {})])
    with pytest.raises(RequestError):
        recreate_variables_index(client, _settings())
    assert client.indices.create.call_count == 1


def test_recreate_variables_op_recreates_then_indexes_every_variable() -> None:
    import nada_ai.ingest.service as service

    calls: list[str] = []
    with (
        patch("nada_ai.search.backend.opensearch.client.build_client", return_value=MagicMock()),
        patch(
            "nada_ai.ingest.variables_index.recreate_variables_index",
            side_effect=lambda client, settings: calls.append("recreate") or "gen-1",
        ),
        patch.object(
            service,
            "sync_variables_op",
            side_effect=lambda settings, idnos, **kw: calls.append(f"sync:{idnos}") or {"scope": "all", "indexed": 7},
        ),
    ):
        result = service.recreate_variables_op(_settings())
    assert calls == ["recreate", "sync:None"]
    assert (result["generation"], result["indexed"]) == ("gen-1", 7)
