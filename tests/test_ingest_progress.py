"""Tests for checkpointing/resume and live progress reporting (ingest/progress.py),
and their wiring into iter_langdoc_records (ingest/pipeline.py) for cancellation
and per-idno load-failure capture.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from nada_ai.ingest.progress import (
    CancelToken,
    IngestCheckpoint,
    IngestProgressTracker,
    checkpoint_path,
    clear_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from nada_ai.settings import Settings


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(ingest_checkpoint_dir=str(tmp_path / "checkpoints"), **overrides)


# ---------------------------------------------------------------------------
# checkpoint file round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    settings = _settings(tmp_path)
    cp = IngestCheckpoint(catalog_type="survey", total=10, completed_idnos={"A", "B"}, failed={"C": "boom"})
    save_checkpoint(settings, cp)

    loaded = load_checkpoint(settings, "survey")
    assert loaded is not None
    assert loaded.total == 10
    assert loaded.completed_idnos == {"A", "B"}
    assert loaded.failed == {"C": "boom"}


def test_load_missing_checkpoint_returns_none(tmp_path):
    settings = _settings(tmp_path)
    assert load_checkpoint(settings, "survey") is None


def test_load_corrupt_checkpoint_returns_none_instead_of_raising(tmp_path):
    settings = _settings(tmp_path)
    path = checkpoint_path(settings, "survey")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")
    assert load_checkpoint(settings, "survey") is None


def test_clear_checkpoint_removes_file(tmp_path):
    settings = _settings(tmp_path)
    save_checkpoint(settings, IngestCheckpoint(catalog_type="survey", total=1))
    assert checkpoint_path(settings, "survey").exists()
    clear_checkpoint(settings, "survey")
    assert not checkpoint_path(settings, "survey").exists()


def test_clear_checkpoint_missing_file_is_a_noop(tmp_path):
    settings = _settings(tmp_path)
    clear_checkpoint(settings, "survey")  # must not raise


def test_save_checkpoint_is_atomic_no_leftover_tmp_file(tmp_path):
    settings = _settings(tmp_path)
    save_checkpoint(settings, IngestCheckpoint(catalog_type="survey", total=1))
    leftovers = list((tmp_path / "checkpoints").glob("*.tmp"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# IngestProgressTracker
# ---------------------------------------------------------------------------


def test_tracker_marks_and_reports_progress(tmp_path):
    settings = _settings(tmp_path)
    updates: list[dict[str, Any]] = []
    tracker = IngestProgressTracker(settings, "survey", total=2, on_update=updates.append, save_every=100)

    tracker.mark("A", ok=True)
    tracker.mark("B", ok=False, error="loader exploded")

    assert updates[0] == {"processed": 1, "total": 2, "failed": 0, "current_idno": "A", "percent": 50.0}
    assert updates[1] == {"processed": 2, "total": 2, "failed": 1, "current_idno": "B", "percent": 100.0}
    assert tracker.checkpoint.completed_idnos == {"A"}
    assert tracker.checkpoint.failed == {"B": "loader exploded"}


def test_tracker_saves_checkpoint_at_save_every_boundary(tmp_path):
    settings = _settings(tmp_path)
    tracker = IngestProgressTracker(settings, "survey", total=5, save_every=2)

    tracker.mark("A", ok=True)
    assert load_checkpoint(settings, "survey") is None  # not yet at the boundary

    tracker.mark("B", ok=True)
    loaded = load_checkpoint(settings, "survey")
    assert loaded is not None
    assert loaded.completed_idnos == {"A", "B"}


def test_tracker_seeds_counts_from_a_resumed_checkpoint(tmp_path):
    settings = _settings(tmp_path)
    existing = IngestCheckpoint(catalog_type="survey", total=5, completed_idnos={"A", "B"}, failed={"C": "x"})
    tracker = IngestProgressTracker(settings, "survey", total=5, checkpoint=existing)

    assert tracker.processed == 3  # 2 completed + 1 failed already accounted for
    assert tracker.failed_count == 1


def test_finalize_completed_clears_checkpoint(tmp_path):
    settings = _settings(tmp_path)
    tracker = IngestProgressTracker(settings, "survey", total=1, save_every=1)
    tracker.mark("A", ok=True)
    assert load_checkpoint(settings, "survey") is not None

    tracker.finalize(completed=True)
    assert load_checkpoint(settings, "survey") is None


def test_finalize_not_completed_keeps_checkpoint(tmp_path):
    settings = _settings(tmp_path)
    tracker = IngestProgressTracker(settings, "survey", total=5, save_every=1)
    tracker.mark("A", ok=True)

    tracker.finalize(completed=False)
    loaded = load_checkpoint(settings, "survey")
    assert loaded is not None
    assert loaded.completed_idnos == {"A"}


# ---------------------------------------------------------------------------
# CancelToken
# ---------------------------------------------------------------------------


def test_cancel_token_starts_unset():
    token = CancelToken()
    assert token.is_set() is False
    token.set()
    assert token.is_set() is True


# ---------------------------------------------------------------------------
# iter_langdoc_records: cancellation + load-failure capture
# ---------------------------------------------------------------------------


class _FakeDoc:
    def __init__(self, page_content: str, metadata: dict[str, Any]) -> None:
        self.page_content = page_content
        self.metadata = metadata


class _FakeHandler:
    def __init__(self, docs: list[_FakeDoc]) -> None:
        self._docs = docs

    def get_langdocs(self) -> list[_FakeDoc]:
        return self._docs


class _FakeLoader:
    """Stand-in for ai4data.discovery.metadata.handler.MetadataLoader."""

    _by_idno: dict[str, list[_FakeDoc]] = {}
    _raises: set[str] = set()

    def __init__(self, idno: str, metadata_type: str, force: bool = False, include_resources: bool = True) -> None:
        if idno in self._raises:
            raise RuntimeError(f"metadata fetch failed for {idno}")
        self.idno = idno
        self.metadata_type = metadata_type
        self.metadata = {}

    def get_metadata_handler(self) -> _FakeHandler:
        return _FakeHandler(self._by_idno.get(self.idno, []))


class _FakeVec:
    def tolist(self) -> list[float]:
        return [0.1, 0.2]


class _FakeEmbedding:
    def encode_corpus(self, texts: list[str], show_progress_bar: bool = True) -> list[_FakeVec]:
        return [_FakeVec() for _ in texts]


def _doc(idno: str) -> _FakeDoc:
    return _FakeDoc(f"a perfectly fine and long enough description for {idno}", {"idno": idno, "type": "document"})


def test_iter_langdoc_records_records_load_errors_instead_of_swallowing_them(tmp_path):
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {"OK-1": [_doc("OK-1")]}
    _FakeLoader._raises = {"BAD-1"}

    with (
        patch.object(pipeline_module, "MetadataLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
    ):
        settings = _settings(tmp_path, search_backend="opensearch")
        load_errors: list[dict[str, Any]] = []
        results = list(
            pipeline_module.iter_langdoc_records(
                settings,
                _FakeEmbedding(),
                [("BAD-1", "document"), ("OK-1", "document")],
                show_progress_bar=False,
                load_errors=load_errors,
            )
        )

    assert len(results) == 1  # only OK-1 produced a document
    assert len(load_errors) == 1
    assert load_errors[0]["idno"] == "BAD-1"
    assert "metadata fetch failed" in load_errors[0]["error"]


def test_iter_langdoc_records_records_empty_docs_instead_of_silently_dropping_them(tmp_path):
    """Regression test: an idno that loads without error but produces zero
    langdocs (or all-empty-content langdocs) previously vanished with no trace
    anywhere — `indexed` could be well below `rows` with both `errors` and
    `load_errors` empty. See the real geospatial catalog run that surfaced
    this: 30 rows, 4 load_errors, only 6 indexed, 20 completely unexplained."""
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {
        "OK-1": [_doc("OK-1")],
        "NO-DOCS": [],
        "EMPTY-CONTENT": [_FakeDoc("   ", {"idno": "EMPTY-CONTENT", "type": "document"})],
    }
    _FakeLoader._raises = set()

    with (
        patch.object(pipeline_module, "MetadataLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
    ):
        settings = _settings(tmp_path, search_backend="opensearch")
        empty_docs: list[dict[str, Any]] = []
        results = list(
            pipeline_module.iter_langdoc_records(
                settings,
                _FakeEmbedding(),
                [("NO-DOCS", "document"), ("OK-1", "document"), ("EMPTY-CONTENT", "document")],
                show_progress_bar=False,
                empty_docs=empty_docs,
            )
        )

    assert len(results) == 1  # only OK-1 produced a document
    assert len(empty_docs) == 2
    by_idno = {e["idno"]: e["reason"] for e in empty_docs}
    assert by_idno == {"NO-DOCS": "no_langdocs", "EMPTY-CONTENT": "empty_page_content"}


def test_iter_langdoc_records_stops_when_cancel_token_is_set(tmp_path):
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {"A": [_doc("A")], "B": [_doc("B")], "C": [_doc("C")]}
    _FakeLoader._raises = set()

    token = CancelToken()
    token.set()  # already cancelled before the loop starts

    with (
        patch.object(pipeline_module, "MetadataLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
    ):
        settings = _settings(tmp_path, search_backend="opensearch")
        results = list(
            pipeline_module.iter_langdoc_records(
                settings,
                _FakeEmbedding(),
                [("A", "document"), ("B", "document"), ("C", "document")],
                show_progress_bar=False,
                cancel_token=token,
            )
        )

    assert results == []


def test_iter_langdoc_records_marks_progress_per_idno(tmp_path):
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {"A": [_doc("A")], "B": [_doc("B")]}
    _FakeLoader._raises = set()

    with (
        patch.object(pipeline_module, "MetadataLoader", _FakeLoader),
        patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]),
    ):
        settings = _settings(tmp_path, search_backend="opensearch")
        tracker = IngestProgressTracker(settings, "document", total=2)
        list(
            pipeline_module.iter_langdoc_records(
                settings,
                _FakeEmbedding(),
                [("A", "document"), ("B", "document")],
                show_progress_bar=False,
                progress=tracker,
            )
        )

    assert tracker.processed == 2
    assert tracker.checkpoint.completed_idnos == {"A", "B"}
