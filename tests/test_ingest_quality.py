"""Tests for the non-blocking ingest quality-report system (ingest/quality.py)
and its wiring through iter_langdoc_records / index_ids_op / index_from_catalog_op.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from nada_ai.ingest.quality import QualityReport, check_source_document


# ── pure function / accumulator tests ──────────────────────────────────────


def test_check_source_document_flags_empty_content():
    source = {"page_content": "", "metadata": {"idno": "X", "type": "indicator"}}
    assert check_source_document(source) == ["empty_page_content"]


def test_check_source_document_flags_whitespace_only_content():
    source = {"page_content": "   \n\t  ", "metadata": {"idno": "X", "type": "indicator"}}
    assert check_source_document(source) == ["empty_page_content"]


def test_check_source_document_flags_very_short_content():
    source = {"page_content": "short", "metadata": {"idno": "X", "type": "indicator"}}
    assert check_source_document(source) == ["very_short_page_content"]


def test_check_source_document_flags_missing_idno_and_type():
    source = {"page_content": "a perfectly fine and long enough description", "metadata": {}}
    issues = check_source_document(source)
    assert "missing_idno" in issues
    assert "missing_type" in issues


def test_check_source_document_clean_document_has_no_issues():
    source = {
        "page_content": "a perfectly fine and long enough description",
        "metadata": {"idno": "ABC_001", "type": "indicator"},
    }
    assert check_source_document(source) == []


def test_quality_report_accumulates_counts_and_samples():
    report = QualityReport(sample_limit=2)
    report.observe({"page_content": "", "metadata": {"idno": "A", "type": "indicator"}})
    report.observe({"page_content": "", "metadata": {"idno": "B", "type": "indicator"}})
    report.observe({"page_content": "", "metadata": {"idno": "C", "type": "indicator"}})
    report.observe({"page_content": "fine long enough description here", "metadata": {"idno": "D", "type": "indicator"}})

    assert report.checked == 4
    out = report.to_dict()
    assert out["checked"] == 4
    assert out["issues"]["empty_page_content"]["count"] == 3
    # sample_limit=2 caps the sample list even though 3 idnos triggered the issue
    assert out["issues"]["empty_page_content"]["sample_idnos"] == ["A", "B"]


def test_quality_report_empty_when_nothing_observed():
    report = QualityReport()
    assert report.to_dict() == {"checked": 0, "issues": {}}


def test_quality_report_never_double_counts_same_idno_in_sample():
    report = QualityReport(sample_limit=5)
    for _ in range(3):
        report.observe({"page_content": "", "metadata": {"idno": "REPEAT", "type": "indicator"}})
    out = report.to_dict()
    assert out["issues"]["empty_page_content"]["count"] == 3
    assert out["issues"]["empty_page_content"]["sample_idnos"] == ["REPEAT"]


# ── pipeline wiring tests ───────────────────────────────────────────────────


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

    def __init__(self, idno: str, metadata_type: str, force: bool = False, include_resources: bool = True) -> None:
        self.idno = idno
        self.metadata_type = metadata_type
        self.metadata = {}

    def get_metadata_handler(self) -> _FakeHandler:
        return _FakeHandler(self._by_idno.get(self.idno, []))


def test_iter_langdoc_records_observes_quality_report(monkeypatch):
    """Wiring test: iter_langdoc_records must call quality_report.observe() for
    every source document it builds, whether or not it's flagged as an issue.

    Note: iter_langdoc_records already filters out docs with empty page_content
    *before* quality_report ever sees them (the ``non_empty`` guard), so this
    uses a doc with content but no ``idno`` metadata to exercise a flag that
    survives that upstream filter.
    """
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {
        "GOOD": [_FakeDoc("a perfectly fine and long enough description", {"idno": "GOOD", "type": "indicator"})],
        "NOMETA": [_FakeDoc("also a perfectly fine and long enough description", {"type": "indicator"})],
    }

    monkeypatch.setattr(pipeline_module, "MetadataLoader", _FakeLoader)
    monkeypatch.setattr(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata.get("idno") or "NOMETA")

    settings = SimpleNamespace(embedding_backend="opensearch_ml")
    report = QualityReport()

    results = list(
        pipeline_module.iter_langdoc_records(
            settings,
            None,
            [("GOOD", "indicator"), ("NOMETA", "indicator")],
            show_progress_bar=False,
            quality_report=report,
        )
    )

    assert len(results) == 2
    assert report.checked == 2
    out = report.to_dict()
    assert out["issues"]["missing_idno"]["count"] == 1


def test_iter_langdoc_records_without_quality_report_is_unaffected(monkeypatch):
    """quality_report=None (the default) must not change what's yielded."""
    import nada_ai.ingest.pipeline as pipeline_module

    _FakeLoader._by_idno = {
        "GOOD": [_FakeDoc("a perfectly fine and long enough description", {"idno": "GOOD", "type": "indicator"})],
    }
    monkeypatch.setattr(pipeline_module, "MetadataLoader", _FakeLoader)
    monkeypatch.setattr(pipeline_module, "get_langdoc_uuid", lambda doc: doc.metadata["idno"])

    settings = SimpleNamespace(embedding_backend="opensearch_ml")
    results = list(
        pipeline_module.iter_langdoc_records(
            settings, None, [("GOOD", "indicator")], show_progress_bar=False
        )
    )
    assert len(results) == 1
    assert results[0][0] == "GOOD"


def test_index_ids_op_includes_quality_report(monkeypatch):
    """index_ids_op must include the accumulated quality report in its result dict."""
    import nada_ai.ingest.service as service_module

    def fake_run_bulk_index(settings, pairs, **kwargs):
        report = kwargs["quality_report"]
        for idno, _ in pairs:
            report.observe({"page_content": "", "metadata": {"idno": idno, "type": "indicator"}})
        return len(pairs), None

    monkeypatch.setattr(service_module, "run_bulk_index", fake_run_bulk_index)

    class FakeSettings:
        search_backend = "qdrant"
        qdrant_collection = "test-coll"

    result = service_module.index_ids_op(FakeSettings(), ["A", "B"], "indicator")

    assert result["indexed"] == 2
    assert "quality" in result
    assert result["quality"]["checked"] == 2
    assert result["quality"]["issues"]["empty_page_content"]["count"] == 2
