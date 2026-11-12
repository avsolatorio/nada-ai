"""Tests for checkpointing/resume and live progress reporting (ingest/progress.py),
and their wiring into iter_langdoc_records (ingest/pipeline.py) for cancellation
and per-study load-failure capture. Studies are keyed by their NADA internal id (sid).
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
from nada_ai.ingest.study_sources import StudySource
from nada_ai.settings import Settings


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(ingest_checkpoint_dir=str(tmp_path / "checkpoints"), **overrides)


# ---------------------------------------------------------------------------
# checkpoint file round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    settings = _settings(tmp_path)
    cp = IngestCheckpoint(catalog_type="survey", total=10, completed_sids={1, 2}, failed={3: "boom"})
    save_checkpoint(settings, cp)

    loaded = load_checkpoint(settings, "survey")
    assert loaded is not None
    assert loaded.total == 10
    assert loaded.completed_sids == {1, 2}
    assert loaded.failed == {3: "boom"}


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

    tracker.mark(1, ok=True)
    tracker.mark(2, ok=False, error="loader exploded")

    assert updates[0] == {"processed": 1, "total": 2, "failed": 0, "current_sid": 1, "percent": 50.0}
    assert updates[1] == {"processed": 2, "total": 2, "failed": 1, "current_sid": 2, "percent": 100.0}
    assert tracker.checkpoint.completed_sids == {1}
    assert tracker.checkpoint.failed == {2: "loader exploded"}


def test_tracker_saves_checkpoint_at_save_every_boundary(tmp_path):
    settings = _settings(tmp_path)
    tracker = IngestProgressTracker(settings, "survey", total=5, save_every=2)

    tracker.mark(1, ok=True)
    assert load_checkpoint(settings, "survey") is None  # not yet at the boundary

    tracker.mark(2, ok=True)
    loaded = load_checkpoint(settings, "survey")
    assert loaded is not None
    assert loaded.completed_sids == {1, 2}


def test_tracker_seeds_counts_from_a_resumed_checkpoint(tmp_path):
    settings = _settings(tmp_path)
    existing = IngestCheckpoint(catalog_type="survey", total=5, completed_sids={1, 2}, failed={3: "x"})
    tracker = IngestProgressTracker(settings, "survey", total=5, checkpoint=existing)

    assert tracker.processed == 3  # 2 completed + 1 failed already accounted for
    assert tracker.failed_count == 1


def test_finalize_completed_clears_checkpoint(tmp_path):
    settings = _settings(tmp_path)
    tracker = IngestProgressTracker(settings, "survey", total=1, save_every=1)
    tracker.mark(1, ok=True)
    assert load_checkpoint(settings, "survey") is not None

    tracker.finalize(completed=True)
    assert load_checkpoint(settings, "survey") is None


def test_finalize_not_completed_keeps_checkpoint(tmp_path):
    settings = _settings(tmp_path)
    tracker = IngestProgressTracker(settings, "survey", total=5, save_every=1)
    tracker.mark(1, ok=True)

    tracker.finalize(completed=False)
    loaded = load_checkpoint(settings, "survey")
    assert loaded is not None
    assert loaded.completed_sids == {1}


def test_a_study_is_marked_only_once_every_chunk_is_confirmed_written(tmp_path):
    tracker = IngestProgressTracker(_settings(tmp_path), "survey", total=2, save_every=1)
    tracker.expect(11, 2)
    tracker.confirm(11, True)
    assert tracker.checkpoint.completed_sids == set()  # one chunk still unconfirmed
    tracker.confirm(11, True)
    assert tracker.checkpoint.completed_sids == {11}


def test_a_study_with_a_rejected_chunk_is_marked_failed_so_a_resume_retries_it(tmp_path):
    tracker = IngestProgressTracker(_settings(tmp_path), "survey", total=1, save_every=1)
    tracker.expect(11, 2)
    tracker.confirm(11, False, "mapper_parsing_exception")
    tracker.confirm(11, True)
    assert tracker.checkpoint.completed_sids == set()
    assert tracker.checkpoint.failed == {11: "write failed: mapper_parsing_exception"}


def test_a_crash_leaves_unconfirmed_studies_out_of_the_checkpoint(tmp_path):
    """Chunks handed to the writer but never answered (the run crashed) are not written: a resume must redo them."""
    settings = _settings(tmp_path)
    tracker = IngestProgressTracker(settings, "survey", total=2, save_every=1)
    tracker.expect(11, 1)
    tracker.confirm(11, True)
    tracker.expect(22, 3)  # sent, never answered
    tracker.finalize(completed=False)

    loaded = load_checkpoint(settings, "survey")
    assert loaded is not None
    assert loaded.completed_sids == {11}


def test_with_study_documents_a_study_waits_for_its_study_document_too(tmp_path):
    """OpenSearch writes one study document per study: a study (even one with no chunks) is done only with it."""
    _FakeLoader._docs = {1: [_doc(1)], 2: []}
    _FakeLoader._raises = set()
    studies: list[Any] = []

    with _fake_loading():
        import nada_ai.ingest.pipeline as pipeline_module

        settings = _settings(tmp_path, search_backend="opensearch")
        tracker = IngestProgressTracker(settings, "document", total=2)
        records = list(
            pipeline_module.iter_langdoc_records(
                settings,
                _FakeEmbedding(),
                [_source(1), _source(2)],
                show_progress_bar=False,
                progress=tracker,
                studies=studies,
                study_documents=True,
            )
        )

    assert [(s.sid, s.chunks) for s in studies] == [(1, 1), (2, 0)]
    for _id, _vec, source in records:
        tracker.confirm(source["metadata"]["sid"], True)
    assert tracker.checkpoint.completed_sids == set()  # chunks written, study documents not yet
    for study in studies:
        tracker.confirm(study.sid, True)
    assert tracker.checkpoint.completed_sids == {1, 2}


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
    """Stand-in for the pipeline's ``_RecordLoader`` (ai4data's loader handed the study's record)."""

    _docs: dict[int, list[_FakeDoc]] = {}
    _raises: set[int] = set()

    def __init__(self, source: StudySource) -> None:
        if source.sid in self._raises:
            raise RuntimeError(f"metadata handler failed for study {source.sid}")
        self.sid = source.sid
        self.metadata = {
            "_extract_filters": {},
            "_extract_core_fields": {"catalog_id": source.sid, "idno": source.idno, "all_text": []},
        }

    def get_metadata_handler(self) -> _FakeHandler:
        return _FakeHandler(self._docs.get(self.sid, []))


class _FakeVec:
    def tolist(self) -> list[float]:
        return [0.1, 0.2]


class _FakeEmbedding:
    input_cap = 1024

    def encode_corpus(self, texts: list[str], show_progress_bar: bool = True) -> list[_FakeVec]:
        return [_FakeVec() for _ in texts]


def _source(sid: int) -> StudySource:
    return StudySource(sid=sid, idno=f"IDNO-{sid}", metadata_type="document", metadata={})


def _doc(sid: int) -> _FakeDoc:
    return _FakeDoc(
        f"a perfectly fine and long enough description for {sid}", {"idno": f"IDNO-{sid}", "type": "document"}
    )


def _fake_loading():
    import contextlib

    import nada_ai.ingest.pipeline as pipeline_module

    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(pipeline_module, "_RecordLoader", _FakeLoader))
    stack.enter_context(patch.object(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"]))
    return stack


def _records(tmp_path, sources, **kwargs):
    import nada_ai.ingest.pipeline as pipeline_module

    with _fake_loading():
        settings = _settings(tmp_path, search_backend="opensearch")
        return list(
            pipeline_module.iter_langdoc_records(settings, _FakeEmbedding(), sources, show_progress_bar=False, **kwargs)
        )


def test_iter_langdoc_records_records_load_errors_instead_of_swallowing_them(tmp_path):
    _FakeLoader._docs = {2: [_doc(2)]}
    _FakeLoader._raises = {1}
    load_errors: list[dict[str, Any]] = []

    results = _records(tmp_path, [_source(1), _source(2)], load_errors=load_errors)

    assert len(results) == 1  # only study 2 produced a document
    assert [(e["sid"], e["idno"], e["stage"]) for e in load_errors] == [(1, "IDNO-1", "load")]
    assert "metadata handler failed" in load_errors[0]["error"]


def test_iter_langdoc_records_records_empty_docs_instead_of_silently_dropping_them(tmp_path):
    """Regression test: a study that loads without error but produces zero
    langdocs (or all-empty-content langdocs) previously vanished with no trace
    anywhere — `indexed` could be well below `rows` with both `errors` and
    `load_errors` empty."""
    _FakeLoader._docs = {
        1: [_doc(1)],
        2: [],
        3: [_FakeDoc("   ", {"idno": "IDNO-3", "type": "document"})],
    }
    _FakeLoader._raises = set()
    empty_docs: list[dict[str, Any]] = []

    results = _records(tmp_path, [_source(2), _source(1), _source(3)], empty_docs=empty_docs)

    assert len(results) == 1  # only study 1 produced a document
    assert {e["sid"]: e["reason"] for e in empty_docs} == {2: "no_langdocs", 3: "empty_page_content"}


def test_iter_langdoc_records_stops_when_cancel_token_is_set(tmp_path):
    _FakeLoader._docs = {1: [_doc(1)], 2: [_doc(2)], 3: [_doc(3)]}
    _FakeLoader._raises = set()
    token = CancelToken()
    token.set()  # already cancelled before the loop starts

    assert _records(tmp_path, [_source(1), _source(2), _source(3)], cancel_token=token) == []


def test_iter_langdoc_records_marks_progress_per_study(tmp_path):
    _FakeLoader._docs = {1: [_doc(1)], 2: [_doc(2)]}
    _FakeLoader._raises = set()
    tracker = IngestProgressTracker(_settings(tmp_path, search_backend="opensearch"), "document", total=2)

    records = _records(tmp_path, [_source(1), _source(2)], progress=tracker)

    # Handed to the writer is not written: nothing is done until the writer confirms each document.
    assert tracker.processed == 0
    assert tracker.checkpoint.completed_sids == set()
    for _id, _vec, source in records:
        tracker.confirm(source["metadata"]["sid"], True)
    assert tracker.processed == 2
    assert tracker.checkpoint.completed_sids == {1, 2}


def test_the_record_loader_hands_ai4data_the_record_and_stamps_nadas_idno():
    """The loader never fetches or caches by idno: it is given the record, and the idno it stamps is NADA's."""
    import nada_ai.ingest.pipeline as pipeline_module

    source = StudySource(sid=5, idno="GHA/2020:x", metadata_type="document", metadata={"title": "T", "idno": "other"})
    with patch("ai4data.discovery.catalog.http.get_metadata_json") as fetch:
        loader = pipeline_module._RecordLoader(source)
    fetch.assert_not_called()
    assert loader.metadata == {"title": "T", "idno": "GHA/2020:x"}
    assert source.metadata["idno"] == "other"  # the source's own record is not changed
