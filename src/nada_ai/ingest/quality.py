"""Lightweight, non-blocking quality checks for ingested documents.

These never reject or skip a document — indexing behavior is completely
unchanged whether or not a :class:`QualityReport` is passed through the
pipeline. They only *observe* each canonical source document as it's built
and accumulate counts/samples of common issues (empty content, missing
facet fields), so operators can see when the catalog is producing thin or
malformed content without it silently vanishing into an otherwise-healthy
ingest count.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

MIN_PAGE_CONTENT_CHARS = 20


def check_source_document(source: dict[str, Any]) -> list[str]:
    """Return a list of issue codes for one canonical source document.

    ``source`` is the ``{"page_content", "metadata", ...}`` shape produced by
    :func:`nada_ai.search.documents.langdoc_to_source`.
    """
    issues: list[str] = []
    page_content = source.get("page_content")
    stripped = str(page_content).strip() if page_content else ""
    if not stripped:
        issues.append("empty_page_content")
    elif len(stripped) < MIN_PAGE_CONTENT_CHARS:
        issues.append("very_short_page_content")

    metadata = source.get("metadata") or {}
    if not metadata.get("idno"):
        issues.append("missing_idno")
    if not metadata.get("type"):
        issues.append("missing_type")

    return issues


class QualityReport:
    """Accumulates issue counts + a bounded sample of offending idnos across a bulk run."""

    def __init__(self, *, sample_limit: int = 20) -> None:
        self._counts: Counter[str] = Counter()
        self._samples: dict[str, list[str]] = {}
        self._sample_limit = sample_limit
        self.checked = 0

    def observe(self, source: dict[str, Any]) -> None:
        self.checked += 1
        issues = check_source_document(source)
        if not issues:
            return
        idno = (source.get("metadata") or {}).get("idno") or "?"
        for issue in issues:
            self._counts[issue] += 1
            samples = self._samples.setdefault(issue, [])
            if idno not in samples and len(samples) < self._sample_limit:
                samples.append(idno)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "issues": {
                issue: {"count": count, "sample_idnos": self._samples.get(issue, [])}
                for issue, count in sorted(self._counts.items(), key=lambda kv: -kv[1])
            },
        }
